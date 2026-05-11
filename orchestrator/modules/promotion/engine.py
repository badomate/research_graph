"""
modules/promotion/engine.py — PromotionEngine orchestrator.

Slim coordinator: polls Paper Tracker for s2-read papers, runs two-pass
promotion (concepts first, edges second), syncs Zotero notes, advances
to s3-distilled.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from ..config import Config, get_config
from ..logging_utils import structured_log
from ..notion_client_wrapper import NotionClientWrapper
from ..vector_index import VectorIndexEngine
from .concept_promoter import ConceptPromoter, _get_page_title, _get_select
from .edge_promoter import EdgePromoter
from .zotero_sync import ZoteroSync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class PromotionEngine:
    """
    Module 6: Promotes verified Knowledge Inbox concepts to Second Brain
    and Edges DB.
    """

    def __init__(
        self,
        vector_index: Optional[VectorIndexEngine] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        self.notion = NotionClientWrapper(self.config)
        self.paper_tracker_db   = self.config.notion_paper_tracker_db_id
        self.knowledge_inbox_db = self.config.notion_knowledge_inbox_db_id
        self.second_brain_db    = self.config.notion_second_brain_db_id
        self.edges_db           = self.config.notion_edges_db_id
        self.deferred_edges_db  = self.config.notion_deferred_edges_db_id
        self.zotero_user_id     = self.config.zotero_user_id
        self.zotero_api_key     = self.config.zotero_api_key
        self._vector_index: VectorIndexEngine | None = vector_index or None

        self._concept_promoter = ConceptPromoter(
            self.notion, self.second_brain_db, self._vector_index
        )
        self._edge_promoter = EdgePromoter(
            self.notion, self.edges_db, self.deferred_edges_db
        )
        self._zotero_sync = ZoteroSync(
            self.notion, self.zotero_user_id, self.zotero_api_key
        )

        # title → SB page_id; seeded once per run(), augmented during promotion.
        self._sb_title_cache: dict[str, str] = {}

    # -- Entry point -----------------------------------------------------------

    def run(self) -> None:
        if not self.edges_db:
            logger.warning(
                "PromotionEngine: NOTION_EDGES_DB_ID is not set — skipping promotion run."
            )
            return
        if not self.paper_tracker_db:
            logger.warning(
                "PromotionEngine: NOTION_PAPER_TRACKER_DB_ID is not set — skipping promotion run."
            )
            return

        logger.info("PromotionEngine: polling Paper Tracker for s2-read papers ...")
        papers = self.notion.query_database(
            self.paper_tracker_db,
            filter={"property": "Status", "status": {"equals": "s2-read"}},
        )
        if not papers:
            logger.info("PromotionEngine: no papers at s2-read.")
            return

        logger.info("PromotionEngine: %d paper(s) ready for promotion.", len(papers))

        self._sb_title_cache = self._concept_promoter.build_sb_title_cache()
        logger.info(
            "PromotionEngine: Second Brain cache seeded with %d concept(s).",
            len(self._sb_title_cache),
        )

        for paper_page in papers:
            try:
                self._promote_paper(paper_page)
            except Exception as exc:
                logger.exception(
                    "PromotionEngine: failed to promote paper %s; error_type=%s",
                    paper_page["id"], type(exc).__name__,
                )

    # -- Per-paper promotion ---------------------------------------------------

    def _promote_paper(self, paper_page: dict) -> None:
        paper_id    = paper_page["id"]
        paper_props = paper_page["properties"]
        paper_title = _get_page_title(paper_page) or paper_id

        all_ki = self._fetch_ki_concepts_for_paper(paper_id)
        verified_ki = [
            p for p in all_ki
            if _get_select(p["properties"], "verification_status") == "verified"
        ]
        rejected_ki = [
            p for p in all_ki
            if _get_select(p["properties"], "verification_status") == "rejected"
        ]
        total_ki     = len(all_ki)
        n_verified   = len(verified_ki)
        n_rejected   = len(rejected_ki)
        n_unreviewed = total_ki - n_verified - n_rejected

        logger.info(
            "PromotionEngine: paper '%s' — %d total / %d verified / %d rejected / %d unreviewed",
            paper_title, total_ki, n_verified, n_rejected, n_unreviewed,
        )

        if n_verified == 0:
            logger.info(
                "PromotionEngine: paper '%s' has no verified concepts — "
                "advancing to s3-distilled without promotion.",
                paper_title,
            )
            try:
                self._zotero_sync.sync_zotero_notes(paper_id, paper_props)
            except Exception:
                logger.warning(
                    "PromotionEngine: Zotero note sync failed for paper '%s' — continuing.",
                    paper_title,
                )
            self.notion.update_page(
                page_id=paper_id,
                properties={"Status": self.notion.status_prop("s3-distilled")},
            )
            return

        try:
            # Pass 1: promote concept nodes
            promoted: dict[str, str] = {}
            for item in verified_ki:
                try:
                    sb_page_id = self._concept_promoter.promote_concept_node(
                        item, self._sb_title_cache, self._edge_promoter
                    )
                    self._edge_promoter.resolve_all_deferred(self._sb_title_cache)
                    if sb_page_id:
                        promoted[item["id"]] = sb_page_id
                except Exception as exc:
                    logger.exception(
                        "PromotionEngine [pass 1]: failed for KI page %s; error_type=%s",
                        item["id"], type(exc).__name__,
                    )

            structured_log(
                logger, "info", "Pass 1 complete",
                promoted=len(promoted), of=n_verified,
            )

            # Pass 2: create edges
            total_edges = 0
            for item in verified_ki:
                ki_page_id = item["id"]
                sb_page_id = promoted.get(ki_page_id)
                if not sb_page_id:
                    continue
                try:
                    n = self._edge_promoter.promote_concept_edges(
                        item, sb_page_id, self._sb_title_cache
                    )
                    total_edges += n
                except Exception as exc:
                    logger.exception(
                        "PromotionEngine [pass 2]: edge promotion failed for KI page %s; error_type=%s",
                        ki_page_id, type(exc).__name__,
                    )

            structured_log(logger, "info", "Pass 2 complete", edges_created=total_edges)

            # Mark promoted KI pages
            for ki_page_id in promoted:
                try:
                    self.notion.update_page(
                        page_id=ki_page_id,
                        properties={"Status": self.notion.select_prop("Promoted")},
                    )
                except Exception:
                    logger.warning(
                        "PromotionEngine: could not mark KI page %s as Promoted.", ki_page_id
                    )

            # Zotero note sync — failure must never block promotion
            try:
                self._zotero_sync.sync_zotero_notes(paper_id, paper_props)
            except Exception:
                logger.warning(
                    "PromotionEngine: Zotero note sync failed for paper '%s' — continuing.",
                    paper_title,
                )

        finally:
            # Advance paper to s3-distilled — always, even on partial failure
            try:
                self.notion.update_page(
                    page_id=paper_id,
                    properties={"Status": self.notion.status_prop("s3-distilled")},
                )
                structured_log(logger, "info", "Paper advanced to s3-distilled", title=paper_title)
            except Exception as exc:
                logger.exception(
                    "PromotionEngine: could not advance paper '%s' to s3-distilled; error_type=%s",
                    paper_title, type(exc).__name__,
                )

    def _fetch_ki_concepts_for_paper(self, paper_page_id: str) -> list[dict]:
        return self.notion.query_database(
            self.knowledge_inbox_db,
            filter={
                "property": "Source Paper",
                "relation": {"contains": paper_page_id},
            },
        )

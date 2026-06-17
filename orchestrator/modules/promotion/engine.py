"""
modules/promotion/engine.py — PromotionEngine (SQLite Store backend).

Polls papers at s2-read, promotes their verified inbox concepts into the Second
Brain (a state flip with cross-paper title de-dup), verifies auto-channel edges
whose endpoints are now both promoted, syncs Zotero notes (best-effort), and
advances the paper to s3-distilled.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from ..config import Config, get_config
from ..logging_utils import structured_log
from ..store import (
    ConceptState,
    PaperStatus,
    Store,
    VerificationStatus,
    make_engine,
)
from ..vector_index import VectorIndexEngine
from .zotero_sync import ZoteroSync

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class PromotionEngine:
    """Module 6: promote verified Knowledge Inbox concepts to the Second Brain."""

    def __init__(
        self,
        vector_index: Optional[VectorIndexEngine] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        self.store = Store(make_engine(self.config.database_url))
        self.store.create_all()
        self._vector_index: VectorIndexEngine | None = vector_index or None
        self._zotero = ZoteroSync(
            self.store, self.config.zotero_user_id, self.config.zotero_api_key
        )
        self._ingestion = None  # lazily built IngestionEngine for edge proposal

    def _linker_engine(self):
        """Lazily build (once per run) the IngestionEngine used to propose edges."""
        if self._ingestion is None:
            from ..ingestion.engine import IngestionEngine
            self._ingestion = IngestionEngine(vector_index=self._vector_index, config=self.config)
        return self._ingestion

    def run(self) -> None:
        logger.info("PromotionEngine: polling for s2-read papers ...")
        papers = self.store.get_papers_by_status(PaperStatus.S2_READ.value)
        if not papers:
            logger.info("PromotionEngine: no papers at s2-read.")
            return
        logger.info("PromotionEngine: %d paper(s) ready.", len(papers))
        for paper in papers:
            try:
                self._promote_paper(paper)
            except Exception as exc:
                logger.exception(
                    "PromotionEngine: failed for paper %s; error_type=%s",
                    paper.id, type(exc).__name__,
                )

    def _promote_paper(self, paper) -> None:
        paper_id = paper.id
        inbox = self.store.concepts_for_paper(paper_id, state=ConceptState.INBOX.value)
        verified = [c for c in inbox if c.verification_status == VerificationStatus.VERIFIED.value]
        logger.info(
            "PromotionEngine: '%s' — %d inbox / %d verified.",
            paper.title, len(inbox), len(verified),
        )

        promoted = 0
        for concept in verified:
            try:
                sb = self.store.promote_concept(concept.id)
                if sb is None:
                    continue
                promoted += 1
                if self._vector_index and self._vector_index.available:
                    try:
                        self._vector_index.promote_concept(concept.id, sb.id)
                    except Exception:
                        logger.warning("PromotionEngine: vector promote failed for %s.", concept.id)
            except Exception as exc:
                logger.exception(
                    "PromotionEngine: promote failed for %s; error_type=%s",
                    concept.id, type(exc).__name__,
                )

        # Propose edges for the just-promoted concepts against the accepted graph.
        # This is where edges are born — a concept links to everything you've
        # accepted, not a stale snapshot from extraction time.
        promoted_ids = [
            c.id for c in self.store.concepts_for_paper(paper_id, state=ConceptState.PROMOTED.value)
        ]
        if promoted_ids:
            try:
                self._linker_engine().link_concepts_against_brain(promoted_ids)
            except Exception:
                logger.warning(
                    "PromotionEngine: edge proposal failed for %s — promoted without edges.",
                    paper_id, exc_info=True,
                )

        edges_verified = self.store.verify_auto_edges_between_promoted()
        structured_log(logger, "info", "Promotion done",
                       promoted=promoted, of=len(verified), edges_verified=edges_verified)

        try:
            self._zotero.sync_zotero_notes(paper)
        except Exception:
            logger.warning("PromotionEngine: Zotero note sync failed — continuing.")

        self.store.set_paper_status(paper_id, PaperStatus.S3_DISTILLED.value)
        structured_log(logger, "info", "Paper advanced to s3-distilled", title=paper.title)

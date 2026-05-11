"""
modules/promotion/concept_promoter.py — Second Brain concept page creation and patching.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from ..notion_client_wrapper import NotionClientWrapper
from ..vector_index import VectorIndexEngine

logger = logging.getLogger(__name__)

_SB_CONCEPT_LEVEL = os.environ.get("SB_CONCEPT_LEVEL", "Concept")


class ConceptPromoter:
    """Creates and patches Second Brain concept pages."""

    def __init__(
        self,
        notion: NotionClientWrapper,
        second_brain_db: str,
        vector_index: Optional[VectorIndexEngine] = None,
    ) -> None:
        self.notion = notion
        self.second_brain_db = second_brain_db
        self._vector_index = vector_index

    def build_sb_title_cache(self) -> dict[str, str]:
        """Return a dict mapping Second Brain concept title → page_id (all variants)."""
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={"property": "Note Level", "select": {"equals": _SB_CONCEPT_LEVEL}},
        )
        _hub_suffix_re = re.compile(r'\s*\[[^\]]+\]\s*$')
        cache: dict[str, str] = {}
        for page in pages:
            title = _get_page_title(page)
            if not title:
                continue
            stripped_hub  = _hub_suffix_re.sub("", title).strip()
            stripped_type = re.sub(r'^\[[^\]]+\]\s*', '', title).strip()
            stripped_both = re.sub(r'^\[[^\]]+\]\s*', '', stripped_hub).strip()
            for t in (title, stripped_hub, stripped_type, stripped_both):
                if t:
                    cache[t] = page["id"]
        return cache

    def promote_concept_node(
        self,
        item: dict,
        sb_title_cache: dict[str, str],
        edge_promoter,
    ) -> str | None:
        """
        Get or create the Second Brain page for a KI item.

        Augments sb_title_cache with the new mapping.
        Resolves deferred edges that target this concept by title.
        Returns the SB page ID on success, None on failure.
        """
        ki_page_id       = item["id"]
        props            = item["properties"]
        corrected_title  = _get_text(props, "Corrected Title").strip()
        title            = corrected_title if corrected_title else _get_page_title(item)
        source_paper_ids = _get_relation(props, "Source Paper")

        sb_page_id = _get_promotion_target(props)

        if not sb_page_id:
            sb_page_id = self._create_sb_concept(item)
            if not sb_page_id:
                logger.error(
                    "ConceptPromoter: could not create SB page for '%s' (%s).",
                    title, ki_page_id,
                )
                return None
            self.notion.update_page(
                page_id=ki_page_id,
                properties={"Promotion Target": self.notion.relation_prop([sb_page_id])},
            )
            logger.info(
                "ConceptPromoter [pass 1]: '%s' → new SB page %s.", title, sb_page_id
            )
        else:
            self._patch_sb_concept(sb_page_id, source_paper_ids)
            logger.info(
                "ConceptPromoter [pass 1]: patched existing SB page %s for '%s'.",
                sb_page_id, title,
            )

        # Inject all title variants into cache for pass 2 edge resolution.
        _hub_re = re.compile(r'\s*\[[^\]]+\]\s*$')
        stripped_hub  = _hub_re.sub("", title).strip()
        stripped_type = re.sub(r'^\[[^\]]+\]\s*', '', title).strip()
        stripped_both = re.sub(r'^\[[^\]]+\]\s*', '', stripped_hub).strip()
        for t in (title, stripped_hub, stripped_type, stripped_both):
            if t:
                sb_title_cache[t] = sb_page_id

        # Resolve any deferred edges that were waiting for this concept.
        for t in (title, stripped_hub, stripped_type, stripped_both):
            if t:
                edge_promoter.resolve_deferred_for_title(t, sb_page_id)

        if self._vector_index and self._vector_index.available:
            try:
                self._vector_index.promote_concept(ki_page_id, sb_page_id)
            except Exception:
                logger.warning(
                    "ConceptPromoter: vector promote failed for KI %s → SB %s.",
                    ki_page_id, sb_page_id,
                )

        return sb_page_id

    def _create_sb_concept(self, ki_item: dict) -> str | None:
        props         = ki_item["properties"]
        corrected     = _get_text(props, "Corrected Title").strip()
        title         = corrected if corrected else _get_page_title(ki_item)
        concept_type  = _get_select(props, "Type")
        source_ids    = _get_relation(props, "Source Paper")
        source_pages  = _get_text(props, "Source Pages")
        source_anchors = _get_text(props, "Source Anchors")

        sb_props: dict[str, Any] = {
            "Name": {"title": [{"type": "text", "text": {"content": title[:2000]}}]},
            "Note Level":       self.notion.select_prop(_SB_CONCEPT_LEVEL),
            "Verified":         self.notion.checkbox_prop(True),
            "Last Verified At": {"date": {"start": datetime.now(tz=timezone.utc).isoformat()}},
            "Source Pages":     {"rich_text": self.notion.rich_text(source_pages)},
            "Source Anchors":   {"rich_text": self.notion.rich_text(source_anchors)},
            "Assumptions":      {"rich_text": self.notion.rich_text(_get_text(props, "Assumptions"))},
            "Statement LaTeX":  {"rich_text": self.notion.rich_text(_get_text(props, "Statement LaTeX"))},
        }
        if source_ids:
            sb_props["Sources"] = self.notion.relation_prop(source_ids)
        if concept_type:
            sb_props["Type"] = self.notion.select_prop(concept_type)

        for ki_key, sb_key in [
            ("Interpretation", "Interpretation"),
            ("Proof Idea",     "Proof Idea"),
            ("Aliases",        "Aliases"),
        ]:
            text = _get_text(props, ki_key)
            if text:
                sb_props[sb_key] = {
                    "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
                }

        for ki_key, sb_key in [
            ("Keywords",            "Keywords"),
            ("Prereq Keywords",     "Prereq Keywords"),
            ("Downstream Keywords", "Downstream Keywords"),
            ("Named Tools",         "Named Tools"),
        ]:
            kw = _get_multi_select(props, ki_key)
            if kw:
                sb_props[sb_key] = self.notion.multi_select_prop(kw)

        try:
            page = self.notion.create_page(
                parent={"database_id": self.second_brain_db},
                properties=sb_props,
            )
            sb_page_id = page["id"]
        except Exception:
            logger.exception(
                "ConceptPromoter: failed to create SB concept for '%s'", title
            )
            return None

        try:
            self._copy_blocks(ki_item["id"], sb_page_id)
        except Exception:
            logger.warning(
                "ConceptPromoter: block copy failed for KI %s → SB %s — body empty.",
                ki_item["id"], sb_page_id,
            )

        return sb_page_id

    def _patch_sb_concept(self, sb_page_id: str, source_paper_ids: list[str]) -> None:
        if not source_paper_ids:
            return
        try:
            page     = self.notion.get_page(sb_page_id)
            existing = _get_relation(page.get("properties", {}), "Sources")
            merged   = list(set(existing) | set(source_paper_ids))
            self.notion.update_page(
                page_id=sb_page_id,
                properties={
                    "Sources": self.notion.relation_prop(merged),
                    "Last Verified At": {
                        "date": {"start": datetime.now(tz=timezone.utc).isoformat()}
                    },
                },
            )
        except Exception:
            logger.warning(
                "ConceptPromoter: could not patch existing SB page %s", sb_page_id
            )

    def _copy_blocks(self, src_id: str, dst_id: str) -> None:
        blocks = self.notion.get_block_children(src_id)
        if not blocks:
            return
        clean = [b for b in (_strip_block(raw) for raw in blocks) if b]
        if not clean:
            return
        for i in range(0, len(clean), 100):
            self.notion.append_block_children(dst_id, clean[i:i + 100])
        logger.debug(
            "ConceptPromoter: copied %d block(s) from %s → %s.",
            len(clean), src_id, dst_id,
        )


# -- Helpers -------------------------------------------------------------------

def _strip_block(block: dict) -> dict | None:
    btype = block.get("type")
    if not btype or btype in ("child_page", "child_database", "unsupported"):
        return None
    inner = block.get(btype)
    if inner is None:
        return None
    inner_clean = {k: v for k, v in inner.items() if k != "children"}
    return {"object": "block", "type": btype, btype: inner_clean}


def _get_page_title(page: dict) -> str:
    for value in page.get("properties", {}).values():
        if value.get("type") == "title":
            try:
                return value["title"][0]["plain_text"]
            except (KeyError, IndexError):
                return ""
    return ""


def _get_text(props: dict, key: str) -> str:
    try:
        segments = props[key]["rich_text"]
        return "".join(seg.get("plain_text", "") for seg in segments)
    except (KeyError, TypeError):
        return ""


def _get_select(props: dict, key: str) -> str:
    try:
        return props[key]["select"]["name"]
    except (KeyError, TypeError):
        return ""


def _get_relation(props: dict, key: str) -> list[str]:
    try:
        return [r["id"] for r in props[key]["relation"]]
    except (KeyError, TypeError):
        return []


def _get_multi_select(props: dict, key: str) -> list[str]:
    try:
        return [opt["name"] for opt in props[key]["multi_select"]]
    except (KeyError, TypeError):
        return []


def _get_promotion_target(props: dict) -> str | None:
    try:
        targets = props["Promotion Target"]["relation"]
        if targets:
            return targets[0]["id"]
    except (KeyError, TypeError):
        pass
    return None

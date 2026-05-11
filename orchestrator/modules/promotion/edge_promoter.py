"""
modules/promotion/edge_promoter.py — Edges DB creation and deferred edge resolution.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..exceptions import PromotionError
from ..notion_client_wrapper import NotionClientWrapper
from .edge_parser import parse_edge_suggestions

logger = logging.getLogger(__name__)


class EdgePromoter:
    """Creates Edges DB rows and manages deferred edges."""

    def __init__(
        self,
        notion: NotionClientWrapper,
        edges_db: str,
        deferred_edges_db: str,
    ) -> None:
        self.notion = notion
        self.edges_db = edges_db
        self.deferred_edges_db = deferred_edges_db

    def create_edge(
        self,
        from_sb_id: str,
        to_sb_id: str,
        relation_type: str,
        rationale: str,
        confidence: float,
        source_paper_ids: list[str],
        needs_review: bool = False,
        driving_fields: list | None = None,
        pre_filter_signal: str = "",
        justification: str = "",
        falsifiability: str = "",
        channel: str = "auto",
        temperature_stable: bool = False,
    ) -> None:
        edge_title = f"{relation_type}: {to_sb_id[:8]}"
        edge_props: dict[str, Any] = {
            "Name": {
                "title": [{"type": "text", "text": {"content": edge_title[:2000]}}]
            },
            "From Concept":  self.notion.relation_prop([from_sb_id]),
            "To Concept":    self.notion.relation_prop([to_sb_id]),
            "Relation Type": self.notion.select_prop(relation_type),
            "Created By":    self.notion.select_prop("AI-suggested"),
            "Status":        self.notion.select_prop("pending"),
            "AI Confidence": {"number": confidence},
        }
        if rationale or justification:
            edge_props["Rationale"] = {
                "rich_text": [
                    {"type": "text", "text": {"content": (rationale or justification)[:2000]}}
                ]
            }
        if source_paper_ids:
            edge_props["Source Papers"] = self.notion.relation_prop(source_paper_ids)
        edge_props["needs_review"] = self.notion.checkbox_prop(needs_review)
        edge_props["channel"] = self.notion.select_prop(channel)
        edge_props["temperature_stable"] = self.notion.checkbox_prop(temperature_stable)
        if driving_fields:
            edge_props["driving_fields"] = {
                "rich_text": [{"type": "text", "text": {
                    "content": ", ".join(driving_fields)[:2000]
                }}]
            }
        if falsifiability:
            edge_props["falsifiability"] = {
                "rich_text": [{"type": "text", "text": {"content": falsifiability[:2000]}}]
            }
        if justification:
            edge_props["justification"] = {
                "rich_text": [{"type": "text", "text": {"content": justification[:2000]}}]
            }
        if pre_filter_signal:
            edge_props["pre_filter_signal"] = self.notion.select_prop(pre_filter_signal)

        self.notion.create_page(
            parent={"database_id": self.edges_db},
            properties=edge_props,
        )

    def defer_edge(
        self,
        from_sb_id: str,
        target_title: str,
        relation_type: str,
        rationale: str,
        confidence: float,
        source_paper_ids: list[str],
        needs_review: bool = False,
        driving_fields: list | None = None,
        pre_filter_signal: str = "",
        justification: str = "",
        falsifiability: str = "",
    ) -> None:
        if not self.deferred_edges_db:
            logger.warning(
                "EdgePromoter: NOTION_DEFERRED_EDGES_DB_ID not set — "
                "edge -[%s]-> '%s' will be permanently lost.",
                relation_type, target_title,
            )
            return

        props: dict[str, Any] = {
            "Name": {
                "title": [{"type": "text", "text": {
                    "content": f"[{relation_type}] → {target_title}"[:2000]
                }}]
            },
            "From Concept":  self.notion.relation_prop([from_sb_id]),
            "Target Title":  {"rich_text": [{"type": "text", "text": {"content": target_title[:2000]}}]},
            "Relation Type": self.notion.select_prop(relation_type),
            "Status":        self.notion.select_prop("pending"),
            "AI Confidence": {"number": confidence},
            "Created At":    {"date": {"start": datetime.now(tz=timezone.utc).isoformat()}},
        }
        if rationale:
            props["Rationale"] = {
                "rich_text": [{"type": "text", "text": {"content": rationale[:2000]}}]
            }
        if source_paper_ids:
            props["Source Papers"] = self.notion.relation_prop(source_paper_ids)

        try:
            self.notion.create_page(
                parent={"database_id": self.deferred_edges_db},
                properties=props,
            )
            logger.info(
                "EdgePromoter: deferred edge -[%s]-> '%s' written.", relation_type, target_title
            )
        except Exception as exc:
            raise PromotionError(
                f"EdgePromoter: failed to write deferred edge -[{relation_type}]-> '{target_title}'"
            ) from exc

    def promote_concept_edges(
        self, item: dict, from_sb_id: str, sb_title_cache: dict[str, str]
    ) -> int:
        """
        Create Edges DB rows for all auto-channel edge suggestions on a KI item.
        Defers edges whose targets are not yet in sb_title_cache.
        Returns the number of edges successfully created.
        """
        ki_page_id       = item["id"]
        props            = item["properties"]
        source_paper_ids = _get_relation(props, "Source Paper")
        edges            = parse_edge_suggestions(props, ki_page_id)

        edges_created = 0
        for edge_entry in edges:
            if edge_entry.get("channel", "auto") == "suggest":
                logger.debug(
                    "EdgePromoter [pass 2]: skipping suggest-channel edge "
                    "'%s' → '%s' on KI page %s.",
                    edge_entry.get("relation_type"), edge_entry.get("target_title"),
                    ki_page_id,
                )
                continue

            rel_type          = edge_entry["relation_type"]
            target_title      = edge_entry["target_title"]
            rationale         = edge_entry["rationale"]
            confidence        = edge_entry["confidence"]
            needs_review      = edge_entry.get("needs_review", False)
            driving_fields    = edge_entry.get("driving_fields", [])
            pre_filter_signal = edge_entry.get("pre_filter_signal", "")
            justification     = edge_entry.get("justification", rationale)
            target_page_id    = edge_entry.get("target_notion_page_id", "")
            falsifiability    = edge_entry.get("falsifiability", "")

            target_sb_id = None
            if target_page_id and target_page_id in sb_title_cache.values():
                target_sb_id = target_page_id
            if not target_sb_id:
                target_sb_id = sb_title_cache.get(target_title)

            effective_rationale = rationale or justification
            if not target_sb_id:
                self.defer_edge(
                    from_sb_id=from_sb_id,
                    target_title=target_title,
                    relation_type=rel_type,
                    rationale=effective_rationale,
                    confidence=confidence,
                    source_paper_ids=source_paper_ids,
                    needs_review=needs_review,
                    driving_fields=driving_fields,
                    pre_filter_signal=pre_filter_signal,
                    justification=justification,
                    falsifiability=falsifiability,
                )
                continue

            try:
                self.create_edge(
                    from_sb_id=from_sb_id,
                    to_sb_id=target_sb_id,
                    relation_type=rel_type,
                    rationale=effective_rationale,
                    confidence=confidence,
                    source_paper_ids=source_paper_ids,
                    needs_review=needs_review,
                    driving_fields=driving_fields,
                    pre_filter_signal=pre_filter_signal,
                    justification=justification,
                    falsifiability=falsifiability,
                )
                edges_created += 1
            except Exception as exc:
                logger.exception(
                    "EdgePromoter [pass 2]: failed to create edge %s -[%s]-> %s; error_type=%s",
                    from_sb_id, rel_type, target_sb_id, type(exc).__name__,
                )

        logger.info(
            "EdgePromoter [pass 2]: %d edge(s) created for KI page %s.",
            edges_created, ki_page_id,
        )
        return edges_created

    def resolve_all_deferred(self, sb_title_cache: dict[str, str]) -> None:
        """Sweep all pending deferred edges against the current SB title cache."""
        if not self.deferred_edges_db:
            return
        all_pending = self.notion.query_database(
            self.deferred_edges_db,
            filter={"property": "Status", "select": {"equals": "pending"}},
        )
        if not all_pending:
            return

        logger.info(
            "EdgePromoter: sweeping %d pending deferred edge(s) from previous runs.",
            len(all_pending),
        )
        for row in all_pending:
            props        = row["properties"]
            target_title = _get_text(props, "Target Title")
            if not target_title:
                self._mark_deferred_stale(row["id"])
                continue

            _hub_re = re.compile(r'\s*\[[^\]]+\]\s*$')
            candidates = []
            for t in (
                target_title,
                _hub_re.sub("", target_title).strip(),
                re.sub(r'^\[[^\]]+\]\s*', '', target_title).strip(),
                re.sub(r'^\[[^\]]+\]\s*', '', _hub_re.sub("", target_title)).strip(),
            ):
                if t and t not in candidates:
                    candidates.append(t)

            to_sb_id = next(
                (sb_title_cache[c] for c in candidates if c in sb_title_cache), None
            )
            if not to_sb_id:
                continue

            from_ids   = _get_relation(props, "From Concept")
            rel_type   = _get_select(props, "Relation Type")
            rationale  = _get_text(props, "Rationale")
            confidence = props.get("AI Confidence", {}).get("number") or 0.0
            paper_ids  = _get_relation(props, "Source Papers")

            if not from_ids:
                self._mark_deferred_stale(row["id"])
                continue

            try:
                self.create_edge(
                    from_sb_id=from_ids[0],
                    to_sb_id=to_sb_id,
                    relation_type=rel_type,
                    rationale=rationale,
                    confidence=confidence,
                    source_paper_ids=paper_ids,
                )
                self._mark_deferred_resolved(row["id"], to_sb_id)
                logger.info(
                    "EdgePromoter: resolved stale deferred edge → '%s'.", target_title
                )
            except Exception as exc:
                logger.exception(
                    "EdgePromoter: failed to resolve stale deferred edge %s; error_type=%s",
                    row["id"], type(exc).__name__,
                )

    def resolve_deferred_for_title(
        self, target_title: str, to_sb_id: str
    ) -> None:
        """Find and resolve all pending deferred edges matching target_title."""
        if not self.deferred_edges_db:
            return
        rows = self.notion.query_database(
            self.deferred_edges_db,
            filter={
                "and": [
                    {"property": "Target Title", "rich_text": {"equals": target_title}},
                    {"property": "Status",       "select":    {"equals": "pending"}},
                ]
            },
        )
        if not rows:
            return

        logger.info(
            "EdgePromoter: resolving %d deferred edge(s) → '%s'.",
            len(rows), target_title,
        )
        for row in rows:
            row_id     = row["id"]
            props      = row["properties"]
            rel_type   = _get_select(props, "Relation Type")
            rationale  = _get_text(props, "Rationale")
            confidence = props.get("AI Confidence", {}).get("number") or 0.0
            from_ids   = _get_relation(props, "From Concept")
            paper_ids  = _get_relation(props, "Source Papers")

            if not from_ids:
                logger.warning(
                    "EdgePromoter: deferred edge %s has no From Concept — skipping.", row_id
                )
                self._mark_deferred_stale(row_id)
                continue

            try:
                self.create_edge(
                    from_sb_id=from_ids[0],
                    to_sb_id=to_sb_id,
                    relation_type=rel_type,
                    rationale=rationale,
                    confidence=confidence,
                    source_paper_ids=paper_ids,
                )
                self._mark_deferred_resolved(row_id, to_sb_id)
            except Exception as exc:
                logger.exception(
                    "EdgePromoter: failed to resolve deferred edge %s; error_type=%s",
                    row_id, type(exc).__name__,
                )

    def _mark_deferred_resolved(self, page_id: str, to_sb_id: str) -> None:
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status":      self.notion.select_prop("resolved"),
                "Resolved To": self.notion.relation_prop([to_sb_id]),
                "Resolved At": {"date": {"start": datetime.now(tz=timezone.utc).isoformat()}},
            },
        )

    def _mark_deferred_stale(self, page_id: str) -> None:
        self.notion.update_page(
            page_id=page_id,
            properties={"Status": self.notion.select_prop("stale")},
        )


# -- Module-level property helpers ---------------------------------------------

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

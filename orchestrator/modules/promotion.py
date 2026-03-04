"""
modules/promotion.py — Module 6: Promotion Engine
──────────────────────────────────────────────────
Polls the Knowledge Inbox for human-verified concepts and promotes them to
the Second Brain and the Edges DB.

Query trigger
-------------
Knowledge Inbox pages where:
  - ``verification_status`` = ``verified``
  - ``graph_link_status``   = ``needs-review``

Per-concept logic
-----------------
1. **Concept promotion** — create a new Second Brain ``"Concept"`` page (or
   patch an existing one if ``Promotion Target`` is already set).
2. **Edge promotion** — for each entry in the ``Edge Suggestions`` JSON
   property, create an Edges DB row linking ``From Concept`` → ``To Concept``
   with the recorded relation type, rationale, and confidence.
3. Update the Knowledge Inbox page:
   ``graph_link_status`` → ``verified-links``, ``Status`` → ``Promoted``.

Prerequisites (manual Notion setup — Group 1)
--------------------------------------------
- ``NOTION_EDGES_DB_ID`` env var must be set.
- Second Brain DB must have ``Sources`` (Relation), ``Interpretation``
  (Rich Text), ``Proof Idea`` (Rich Text), ``Named Tools`` (Multi-select),
  ``Aliases`` (Rich Text), ``Verified`` (Checkbox),
  ``Last Verified At`` (Date) properties.
- Edges DB must exist with the schema defined in Group 1d of the plan.

If ``NOTION_EDGES_DB_ID`` is not set the engine logs a warning and exits
cleanly — other pipeline jobs are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

# Note Level value to use when creating Second Brain Concept pages.
# Mirror of ingestion.py — both read from the same env var.
_SB_CONCEPT_LEVEL = os.environ.get("SB_CONCEPT_LEVEL", "Concept")


class PromotionEngine:
    """
    Module 6: Promotes verified Knowledge Inbox concepts to Second Brain
    and Edges DB.
    """

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.edges_db: str = os.environ.get("NOTION_EDGES_DB_ID", "")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Promote verified KI concepts to Second Brain and Edges DB."""
        if not self.edges_db:
            logger.warning(
                "PromotionEngine: NOTION_EDGES_DB_ID is not set — "
                "skipping promotion run. Set it in .env to enable promotion."
            )
            return

        logger.info("PromotionEngine: polling for verified concepts ...")
        items = self.notion.query_database(
            self.knowledge_inbox_db,
            filter={
                "and": [
                    {
                        "property": "verification_status",
                        "select": {"equals": "verified"},
                    },
                    {
                        "property": "graph_link_status",
                        "select": {"equals": "needs-review"},
                    },
                ]
            },
        )
        logger.info("PromotionEngine: found %d item(s) to promote.", len(items))

        for item in items:
            try:
                self._promote_concept(item)
            except Exception:
                logger.exception(
                    "PromotionEngine: failed for KI page %s", item["id"]
                )

    # ── Per-concept promotion ─────────────────────────────────────────────────

    def _promote_concept(self, item: dict) -> None:
        """Promote a single Knowledge Inbox item to Second Brain and Edges DB."""
        ki_page_id = item["id"]
        props = item["properties"]

        title = self._get_page_title(item)
        source_paper_ids = self._get_relation(props, "Source Paper")
        edge_suggestions_raw = self._get_text(props, "Edge Suggestions")

        # Parse edge suggestions JSON (written by ingestion LINK stage).
        edge_suggestions: list[dict] = []
        if edge_suggestions_raw:
            try:
                parsed = json.loads(edge_suggestions_raw)
                if isinstance(parsed, list):
                    edge_suggestions = parsed
            except json.JSONDecodeError:
                logger.warning(
                    "PromotionEngine: invalid Edge Suggestions JSON on page %s",
                    ki_page_id,
                )

        # Step 1: Get or create Second Brain concept page.
        sb_page_id = self._get_promotion_target(props)
        if not sb_page_id:
            sb_page_id = self._create_sb_concept(item)
            if not sb_page_id:
                logger.error(
                    "PromotionEngine: could not create SB page for '%s' (%s) — skipping.",
                    title,
                    ki_page_id,
                )
                return
            # Write Promotion Target back to KI page.
            self.notion.update_page(
                page_id=ki_page_id,
                properties={
                    "Promotion Target": self.notion.relation_prop([sb_page_id])
                },
            )
            logger.info(
                "PromotionEngine: promoted '%s' → new SB page %s.", title, sb_page_id
            )
        else:
            # Patch existing SB page with fresh source references.
            self._patch_sb_concept(sb_page_id, source_paper_ids)
            logger.info(
                "PromotionEngine: updated existing SB page %s for '%s'.",
                sb_page_id,
                title,
            )

        # Step 2: Create Edges DB rows from edge suggestions.
        edges_created = 0
        for edge in edge_suggestions:
            target_id = edge.get("target_concept_id", "")
            if not target_id:
                continue
            try:
                self._create_edge(sb_page_id, edge, source_paper_ids)
                edges_created += 1
            except Exception:
                logger.exception(
                    "PromotionEngine: failed to create edge %s → %s",
                    sb_page_id,
                    target_id,
                )
        logger.info(
            "PromotionEngine: created %d edge(s) for '%s'.", edges_created, title
        )

        # Step 3: Mark KI page as promoted.
        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "graph_link_status": self.notion.select_prop("verified-links"),
                "Status": self.notion.select_prop("Promoted"),
            },
        )

    # ── Second Brain helpers ──────────────────────────────────────────────────

    def _create_sb_concept(self, ki_item: dict) -> str | None:
        """
        Create a new Second Brain Concept page from a Knowledge Inbox item.
        Returns the new page ID, or None on failure.
        """
        props = ki_item["properties"]
        title = self._get_page_title(ki_item)
        concept_type = self._get_select(props, "Type")
        source_paper_ids = self._get_relation(props, "Source Paper")

        sb_properties: dict[str, Any] = {
            "Name": {
                "title": [{"type": "text", "text": {"content": title[:2000]}}]
            },
            "Note Level": self.notion.select_prop(_SB_CONCEPT_LEVEL),
            "Verified": self.notion.checkbox_prop(True),
            "Last Verified At": {
                "date": {"start": datetime.now(tz=timezone.utc).isoformat()}
            },
        }

        if source_paper_ids:
            sb_properties["Sources"] = self.notion.relation_prop(source_paper_ids)
        if concept_type:
            sb_properties["Type"] = self.notion.select_prop(concept_type)

        interpretation = self._get_text(props, "Interpretation")
        if interpretation:
            sb_properties["Interpretation"] = {
                "rich_text": [
                    {"type": "text", "text": {"content": interpretation[:2000]}}
                ]
            }
        proof_idea = self._get_text(props, "Proof Idea")
        if proof_idea:
            sb_properties["Proof Idea"] = {
                "rich_text": [
                    {"type": "text", "text": {"content": proof_idea[:2000]}}
                ]
            }
        aliases = self._get_text(props, "Aliases")
        if aliases:
            sb_properties["Aliases"] = {
                "rich_text": [
                    {"type": "text", "text": {"content": aliases[:2000]}}
                ]
            }
        named_tools = self._get_multi_select(props, "Named Tools")
        if named_tools:
            sb_properties["Named Tools"] = self.notion.multi_select_prop(named_tools)

        try:
            page = self.notion.create_page(
                parent={"database_id": self.second_brain_db},
                properties=sb_properties,
            )
            return page["id"]
        except Exception:
            logger.exception(
                "PromotionEngine: failed to create SB concept for '%s'", title
            )
            return None

    def _patch_sb_concept(
        self,
        sb_page_id: str,
        source_paper_ids: list[str],
    ) -> None:
        """Merge new source papers into an existing Second Brain concept page."""
        if not source_paper_ids:
            return
        try:
            page = self.notion.get_page(sb_page_id)
            existing = self._get_relation(page.get("properties", {}), "Sources")
            # Union of existing and new IDs, deduplicated.
            merged = list(set(existing) | set(source_paper_ids))
            self.notion.update_page(
                page_id=sb_page_id,
                properties={
                    "Sources": self.notion.relation_prop(merged),
                    "Last Verified At": {
                        "date": {
                            "start": datetime.now(tz=timezone.utc).isoformat()
                        }
                    },
                },
            )
        except Exception:
            logger.warning(
                "PromotionEngine: could not patch existing SB page %s", sb_page_id
            )

    # ── Edges DB helpers ──────────────────────────────────────────────────────

    def _create_edge(
        self,
        from_sb_id: str,
        edge: dict,
        source_paper_ids: list[str],
    ) -> None:
        """Create a single row in the Edges DB for one edge suggestion."""
        target_id: str = edge.get("target_concept_id", "")
        relation_type: str = edge.get("relation_type", "related")
        rationale: str = edge.get("rationale", "")
        confidence: float = float(edge.get("confidence", 0.0))
        target_name: str = edge.get("target_name", "?")

        # Resolve from-concept name for a human-readable edge title.
        try:
            from_page = self.notion.get_page(from_sb_id)
            from_name = self._get_page_title(from_page)
        except Exception:
            from_name = "?"

        edge_title = f"{from_name} —{relation_type}→ {target_name}"

        edge_properties: dict[str, Any] = {
            "Name": {
                "title": [
                    {"type": "text", "text": {"content": edge_title[:2000]}}
                ]
            },
            "From Concept": self.notion.relation_prop([from_sb_id]),
            "To Concept": self.notion.relation_prop([target_id]),
            "Relation Type": self.notion.select_prop(relation_type),
            "Created By": self.notion.select_prop("AI-suggested"),
            "Status": self.notion.select_prop("suggested"),
            "Confidence": {"number": confidence},
        }

        if rationale:
            edge_properties["Rationale"] = {
                "rich_text": [
                    {"type": "text", "text": {"content": rationale[:2000]}}
                ]
            }
        if source_paper_ids:
            edge_properties["Source Papers"] = self.notion.relation_prop(
                source_paper_ids
            )

        self.notion.create_page(
            parent={"database_id": self.edges_db},
            properties=edge_properties,
        )

    # ── Property helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_page_title(page: dict) -> str:
        """Extract plain-text title from a raw Notion page object."""
        for value in page.get("properties", {}).values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return ""
        return ""

    @staticmethod
    def _get_text(props: dict, key: str) -> str:
        """Extract plain text from a Notion rich_text property."""
        try:
            return props[key]["rich_text"][0]["plain_text"]
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _get_select(props: dict, key: str) -> str:
        """Extract option name from a Notion select property."""
        try:
            return props[key]["select"]["name"]
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def _get_relation(props: dict, key: str) -> list[str]:
        """Extract page IDs from a Notion relation property."""
        try:
            return [r["id"] for r in props[key]["relation"]]
        except (KeyError, TypeError):
            return []

    @staticmethod
    def _get_multi_select(props: dict, key: str) -> list[str]:
        """Extract option names from a Notion multi_select property."""
        try:
            return [opt["name"] for opt in props[key]["multi_select"]]
        except (KeyError, TypeError):
            return []

    @staticmethod
    def _get_promotion_target(props: dict) -> str | None:
        """Return the first Promotion Target page ID, or None if not set."""
        try:
            targets = props["Promotion Target"]["relation"]
            if targets:
                return targets[0]["id"]
        except (KeyError, TypeError):
            pass
        return None

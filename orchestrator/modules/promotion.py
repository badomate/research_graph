"""
modules/promotion.py — Module 6: Promotion Engine
──────────────────────────────────────────────────
Polls the Knowledge Inbox for human-verified concepts and promotes them to
the Second Brain and the Edges DB.

Query trigger
-------------
Knowledge Inbox pages where:
  - ``verification_status`` = ``verified``
  - ``graph_link_status``   = ``linked-ai``

Per-concept logic
-----------------
1. **Concept promotion** — create a new Second Brain ``"Concept"`` page (or
   patch an existing one if ``Promotion Target`` is already set).
2. **Edge promotion** — for each entry in the ``Edge Suggestions`` JSON
   property, create an Edges DB row linking ``From Concept`` → ``To Concept``
   with the recorded relation type, rationale, and confidence.
3. Update the Knowledge Inbox page:
   ``graph_link_status`` → ``promoted``, ``Status`` → ``Promoted``.

Edge Suggestions JSON format (written by Stage 3 of IngestionEngine)
---------------------------------------------------------------------
{
  "depends_on":    [{"target_concept_id": "...", "target_title": "...", "rationale": "...", "confidence": 0.9}],
  "enables":       [...],
  "generalizes":   [...],
  "special_case_of": [...],
  "related":       [...]
}

Prerequisites (manual Notion setup)
------------------------------------
- ``NOTION_EDGES_DB_ID`` env var must be set.
- Second Brain DB must have:
    ``Sources`` (Relation), ``Interpretation`` (Rich Text),
    ``Proof Idea`` (Rich Text), ``Named Tools`` (Multi-select),
    ``Aliases`` (Rich Text), ``Verified`` (Checkbox),
    ``Last Verified At`` (Date), ``Type`` (Select),
    ``Note Level`` (Select).
- Edges DB must have:
    ``Name`` (Title), ``From Concept`` (Relation), ``To Concept`` (Relation),
    ``Relation Type`` (Select), ``Rationale`` (Rich Text),
    ``AI Confidence`` (Number), ``Created By`` (Select),
    ``Status`` (Select), ``Source Papers`` (Relation).
- Knowledge Inbox DB must have:
    ``Promotion Target`` (Relation → Second Brain).

If ``NOTION_EDGES_DB_ID`` is not set the engine logs a warning and exits
cleanly — other pipeline jobs are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from .notion_client_wrapper import NotionClientWrapper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_SB_CONCEPT_LEVEL = os.environ.get("SB_CONCEPT_LEVEL", "Concept")

# Structured edge entry parsed from Edge Suggestions JSON.
# (relation_type, target_title, rationale, confidence)
_EdgeEntry = tuple[str, str, str, float]


class PromotionEngine:
    """
    Module 6: Promotes verified Knowledge Inbox concepts to Second Brain
    and Edges DB.
    """

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db    = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.edges_db: str      = os.environ.get("NOTION_EDGES_DB_ID", "")
        # title -> SB page_id; populated once per run(), augmented on promotion.
        self._sb_title_cache: dict[str, str] = {}

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Promote verified KI concepts to Second Brain and Edges DB.

        Two-pass design
        ───────────────
        Pass 1 — concept promotion:
            For every approved KI item, create (or patch) its Second Brain
            page and record the ki_page_id → sb_page_id mapping.  After this
            pass the full batch is represented in _sb_title_cache, so every
            edge target in the batch is resolvable.

        Pass 2 — edge promotion:
            Iterate the same items again, resolve both source and target
            concept IDs via _sb_title_cache, and write Edges DB rows.

        This guarantees that intra-batch edges (concept A → concept B where
        both A and B are in the current promotion batch) are always created,
        regardless of ordering.
        """
        if not self.edges_db:
            logger.warning(
                "PromotionEngine: NOTION_EDGES_DB_ID is not set — "
                "skipping promotion run."
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
                        "property": "Status",
                        "select": {"equals": "Approved"},
                    },
                ]
            },
        )
        logger.info("PromotionEngine: found %d item(s) to promote.", len(items))
        if not items:
            return

        # Seed cache from existing SB concepts so cross-paper edges resolve too.
        self._sb_title_cache = self._build_sb_title_cache()
        logger.info(
            "PromotionEngine: Second Brain cache seeded with %d concept(s).",
            len(self._sb_title_cache),
        )

        # ── Pass 1: promote all concepts, build complete ki_id → sb_id map ───
        # ki_page_id → sb_page_id for every successfully promoted item.
        promoted: dict[str, str] = {}

        for item in items:
            try:
                sb_page_id = self._promote_concept_node(item)
                if sb_page_id:
                    promoted[item["id"]] = sb_page_id
            except Exception:
                logger.exception(
                    "PromotionEngine [pass 1]: failed for KI page %s", item["id"]
                )

        logger.info(
            "PromotionEngine: pass 1 complete — %d/%d concept(s) promoted.",
            len(promoted), len(items),
        )

        # ── Pass 2: create edges now that the full batch is in _sb_title_cache ─
        total_edges = 0
        for item in items:
            ki_page_id = item["id"]
            sb_page_id = promoted.get(ki_page_id)
            if not sb_page_id:
                # Concept promotion failed in pass 1 — skip edges too.
                continue
            try:
                n = self._promote_concept_edges(item, sb_page_id)
                total_edges += n
            except Exception:
                logger.exception(
                    "PromotionEngine [pass 2]: edge promotion failed for KI page %s",
                    ki_page_id,
                )

        logger.info(
            "PromotionEngine: pass 2 complete — %d edge(s) created.", total_edges
        )

        # ── Finalise: mark all successfully promoted KI pages ────────────────
        for ki_page_id in promoted:
            try:
                self.notion.update_page(
                    page_id=ki_page_id,
                    properties={
                        "Status": self.notion.select_prop("Promoted"),
                    },
                )
            except Exception:
                logger.warning(
                    "PromotionEngine: could not mark KI page %s as Promoted.",
                    ki_page_id,
                )

    # ── Pass 1: concept node promotion ───────────────────────────────────────

    def _promote_concept_node(self, item: dict) -> str | None:
        """
        Get or create the Second Brain page for a KI item.

        Returns the SB page ID on success, None on failure.
        Augments _sb_title_cache with the new mapping so pass 2 can resolve
        edges that target this concept.
        """
        ki_page_id       = item["id"]
        props            = item["properties"]
        title            = self._get_page_title(item)
        source_paper_ids = self._get_relation(props, "Source Paper")

        sb_page_id = self._get_promotion_target(props)

        if not sb_page_id:
            sb_page_id = self._create_sb_concept(item)
            if not sb_page_id:
                logger.error(
                    "PromotionEngine: could not create SB page for '%s' (%s).",
                    title, ki_page_id,
                )
                return None
            # Write Promotion Target back to KI page for idempotency.
            self.notion.update_page(
                page_id=ki_page_id,
                properties={
                    "Promotion Target": self.notion.relation_prop([sb_page_id])
                },
            )
            logger.info(
                "PromotionEngine [pass 1]: '%s' → new SB page %s.", title, sb_page_id
            )
        else:
            self._patch_sb_concept(sb_page_id, source_paper_ids)
            logger.info(
                "PromotionEngine [pass 1]: patched existing SB page %s for '%s'.",
                sb_page_id, title,
            )

        # Inject into cache so pass 2 (and later items in pass 1) can resolve
        # edges pointing to this concept by title.
        self._sb_title_cache[title] = sb_page_id
        return sb_page_id

    # ── Pass 2: edge promotion ────────────────────────────────────────────────

    def _promote_concept_edges(self, item: dict, from_sb_id: str) -> int:
        """
        Create Edges DB rows for all edge suggestions on a KI item.

        Both source (from_sb_id, already resolved) and targets are SB page
        IDs resolved via _sb_title_cache.  Targets not present in the cache
        (i.e. concepts that were never promoted) are skipped with a log entry.

        Returns the number of edges successfully created.
        """
        ki_page_id       = item["id"]
        props            = item["properties"]
        source_paper_ids = self._get_relation(props, "Source Paper")
        edges            = self._parse_edge_suggestions(props, ki_page_id)

        edges_created = 0
        for rel_type, target_title, rationale, confidence in edges:
            target_sb_id = self._sb_title_cache.get(target_title)
            if not target_sb_id:
                logger.info(
                    "PromotionEngine [pass 2]: target '%s' not in SB cache — "
                    "skipping edge (concept may not be promoted yet).",
                    target_title,
                )
                continue
            try:
                self._create_edge(
                    from_sb_id=from_sb_id,
                    to_sb_id=target_sb_id,
                    relation_type=rel_type,
                    rationale=rationale,
                    confidence=confidence,
                    source_paper_ids=source_paper_ids,
                )
                edges_created += 1
            except Exception:
                logger.exception(
                    "PromotionEngine [pass 2]: failed to create edge %s -[%s]-> %s",
                    from_sb_id, rel_type, target_sb_id,
                )

        logger.info(
            "PromotionEngine [pass 2]: %d edge(s) created for KI page %s.",
            edges_created, ki_page_id,
        )
        return edges_created

    # ── Edge Suggestions parser ───────────────────────────────────────────────

    def _parse_edge_suggestions(
        self, props: dict, ki_page_id: str
    ) -> list[_EdgeEntry]:
        """
        Parse the Edge Suggestions JSON property into a flat list of
        (relation_type, target_title, rationale, confidence) tuples.

        Handles the ConceptLinkResult format written by Stage 3:
            {
              "depends_on": [
                {"target_concept_id": "...", "target_title": "...",
                 "rationale": "...", "confidence": 0.9},
                ...
              ],
              ...
            }

        Also handles the legacy flat-list format for backward compatibility:
            [{"relation_type": "...", "target_name": "..."}]
        """
        raw = self._get_text(props, "Edge Suggestions")
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "PromotionEngine: invalid Edge Suggestions JSON on page %s",
                ki_page_id,
            )
            return []

        edges: list[_EdgeEntry] = []

        if isinstance(parsed, dict):
            for rel_type, targets in parsed.items():
                if not isinstance(targets, list):
                    continue
                for entry in targets:
                    if isinstance(entry, str):
                        # Bare string — no rationale or confidence available.
                        t = entry.strip()
                        if t:
                            edges.append((rel_type, t, "", 0.0))
                    elif isinstance(entry, dict):
                        target_title = entry.get("target_title", "").strip()
                        rationale    = entry.get("rationale", "")
                        confidence   = float(entry.get("confidence", 0.0))
                        if target_title:
                            edges.append((rel_type, target_title, rationale, confidence))

        elif isinstance(parsed, list):
            # Legacy flat-list fallback.
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                rel_type = entry.get("relation_type", "related")
                target   = entry.get("target_name", "") or entry.get("target_title", "")
                rationale  = entry.get("rationale", "")
                confidence = float(entry.get("confidence", 0.0))
                if target:
                    edges.append((rel_type, target.strip(), rationale, confidence))

        return edges

    # ── Second Brain helpers ──────────────────────────────────────────────────

    def _build_sb_title_cache(self) -> dict[str, str]:
        """Return a dict mapping Second Brain concept title -> page_id."""
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Note Level",
                "select": {"equals": _SB_CONCEPT_LEVEL},
            },
        )
        return {
            title: page["id"]
            for page in pages
            if (title := self._get_page_title(page))
        }

    def _create_sb_concept(self, ki_item: dict) -> str | None:
        """
        Create a new Second Brain Concept page from a Knowledge Inbox item.
        Returns the new page ID, or None on failure.
        """
        props        = ki_item["properties"]
        title        = self._get_page_title(ki_item)
        concept_type = self._get_select(props, "Type")
        source_ids   = self._get_relation(props, "Source Paper")
        source_pages = self._get_text(props, "Source Pages")
        source_anchors = self._get_text(props, "Source Anchors")

        sb_props: dict[str, Any] = {
            "Name": {
                "title": [{"type": "text", "text": {"content": title[:2000]}}]
            },
            "Note Level":      self.notion.select_prop(_SB_CONCEPT_LEVEL),
            "Verified":        self.notion.checkbox_prop(True),
            "Last Verified At": {
                "date": {"start": datetime.now(tz=timezone.utc).isoformat()}
            },
            "Source Pages": {
                "rich_text": self.notion.rich_text(source_pages)
            },
            "Source Anchors":{
                "rich_text": self.notion.rich_text(source_anchors)
            }
        }

        if source_ids:
            sb_props["Source Paper"] = self.notion.relation_prop(source_ids)
        if concept_type:
            sb_props["Type"] = self.notion.select_prop(concept_type)

        for ki_key, sb_key in [
            ("Interpretation", "Interpretation"),
            ("Proof Idea",     "Proof Idea"),
            ("Aliases",        "Aliases"),
        ]:
            text = self._get_text(props, ki_key)
            if text:
                sb_props[sb_key] = {
                    "rich_text": [
                        {"type": "text", "text": {"content": text[:2000]}}
                    ]
                }

        for ki_key, sb_key in [
            ("Keywords", "Keywords"),
            ("Prereq Keywords",     "Prereq Keywords"),
            ("Downstream Keywords",        "Downstream Keywords"),
            ("Named Tools", "Named Tools")
        ]:
            keyword = self._get_multi_select(props, ki_key)
            if keyword:
                sb_props[sb_key] = self.notion.multi_select_prop(keyword)

        try:
            page = self.notion.create_page(
                parent={"database_id": self.second_brain_db},
                properties=sb_props,
            )
            sb_page_id = page["id"]
        except Exception:
            logger.exception(
                "PromotionEngine: failed to create SB concept for '%s'", title
            )
            return None

        # Copy block body (Assumptions, Statement, Variables, etc.) from KI page.
        try:
            self._copy_blocks(ki_item["id"], sb_page_id)
        except Exception:
            logger.warning(
                "PromotionEngine: block copy failed for KI page %s → SB page %s — "
                "page created but body is empty.",
                ki_item["id"], sb_page_id,
            )

        return sb_page_id
    

    def _patch_sb_concept(
        self,
        sb_page_id: str,
        source_paper_ids: list[str],
    ) -> None:
        """Merge new source papers into an existing Second Brain concept page."""
        if not source_paper_ids:
            return
        try:
            page     = self.notion.get_page(sb_page_id)
            existing = self._get_relation(page.get("properties", {}), "Sources")
            merged   = list(set(existing) | set(source_paper_ids))
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
        to_sb_id: str,
        relation_type: str,
        rationale: str,
        confidence: float,
        source_paper_ids: list[str],
    ) -> None:
        """Create a single row in the Edges DB."""
        edge_title = f"{relation_type}: {to_sb_id[:8]}"

        edge_props: dict[str, Any] = {
            "Name": {
                "title": [{"type": "text", "text": {"content": edge_title[:2000]}}]
            },
            "From Concept":  self.notion.relation_prop([from_sb_id]),
            "To Concept":    self.notion.relation_prop([to_sb_id]),
            "Relation Type": self.notion.select_prop(relation_type),
            "Created By":    self.notion.select_prop("AI-suggested"),
            "Status":        self.notion.select_prop("suggested"),
            "AI Confidence": {"number": confidence},
        }

        if rationale:
            edge_props["Rationale"] = {
                "rich_text": [
                    {"type": "text", "text": {"content": rationale[:2000]}}
                ]
            }
        if source_paper_ids:
            edge_props["Source Papers"] = self.notion.relation_prop(source_paper_ids)

        self.notion.create_page(
            parent={"database_id": self.edges_db},
            properties=edge_props,
        )
    # ── Block copy ────────────────────────────────────────────────────────────

    def _copy_blocks(self, src_id: str, dst_id: str) -> None:
        """
        Copy all block children from src page to dst page.

        Notion read-only metadata fields (id, created_time, last_edited_time,
        created_by, last_edited_by, has_children, archived) are stripped before
        re-posting. Unsupported block types (child_page, child_database) are
        skipped — they cannot be created via the blocks API.

        Blocks are posted in batches of 100 (Notion API limit).
        """
        blocks = self.notion.get_block_children(src_id)
        if not blocks:
            return
        clean = [b for b in (self._strip_block(raw) for raw in blocks) if b]
        if not clean:
            return
        for i in range(0, len(clean), 100):
            self.notion.append_block_children(dst_id, clean[i:i + 100])
        logger.debug(
            "PromotionEngine: copied %d block(s) from %s → %s.",
            len(clean), src_id, dst_id,
        )

    @staticmethod
    def _strip_block(block: dict) -> dict | None:
        """
        Remove Notion read-only fields from a block dict so it can be
        re-posted via the blocks.children.append endpoint.

        Returns None for block types that cannot be created via the API
        (child_page, child_database, unsupported).
        """
        btype = block.get("type")
        if not btype or btype in ("child_page", "child_database", "unsupported"):
            return None
        inner = block.get(btype)
        if inner is None:
            return None
        # Strip children recursively — nested blocks must be appended separately
        # after the parent is created. For now we copy only the top-level content.
        inner_clean = {k: v for k, v in inner.items() if k != "children"}
        return {"object": "block", "type": btype, btype: inner_clean}
    
    # ── Property helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_page_title(page: dict) -> str:
        for value in page.get("properties", {}).values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return ""
        return ""

    @staticmethod
    def _get_text(props: dict, key: str) -> str:
        try:
            segments = props[key]["rich_text"]
            return "".join(seg.get("plain_text", "") for seg in segments)
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def _get_select(props: dict, key: str) -> str:
        try:
            return props[key]["select"]["name"]
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def _get_relation(props: dict, key: str) -> list[str]:
        try:
            return [r["id"] for r in props[key]["relation"]]
        except (KeyError, TypeError):
            return []

    @staticmethod
    def _get_multi_select(props: dict, key: str) -> list[str]:
        try:
            return [opt["name"] for opt in props[key]["multi_select"]]
        except (KeyError, TypeError):
            return []

    @staticmethod
    def _get_promotion_target(props: dict) -> str | None:
        try:
            targets = props["Promotion Target"]["relation"]
            if targets:
                return targets[0]["id"]
        except (KeyError, TypeError):
            pass
        return None
    

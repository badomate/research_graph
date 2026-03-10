"""
modules/promotion.py — Module 6: Promotion Engine
──────────────────────────────────────────────────
Polls the Paper Tracker for papers at s2-read and promotes all verified
Knowledge Inbox concepts for each paper to the Second Brain and Edges DB.

New trigger (v2)
----------------
PromotionEngine polls **Paper Tracker** for ``Status = s2-read``.
For each such paper it:
  1. Fetches all KI concepts where ``Source Paper = paper_id AND
     verification_status = verified``.
  2. Runs the two-pass promotion (concept nodes first, edges second).
  3. Syncs Zotero reading notes and annotations to the paper page.
  4. Advances the paper to ``s3-distilled``.

Two-pass design
---------------
Pass 1 — concept node promotion:
    For every verified KI item, create (or patch) its Second Brain page
    and record the ki_page_id → sb_page_id mapping.

Pass 2 — edge promotion:
    Resolve both source and target concept IDs via ``_sb_title_cache``
    and write Edges DB rows.

This guarantees that intra-batch edges (concept A → concept B where both
are in the current promotion batch) are always created regardless of ordering.

Corrected Title (REQ-8)
-----------------------
If the KI page has a non-empty ``Corrected Title`` property, it is used
as the Second Brain page name instead of the AI-generated ``Name``.

Edge Suggestions JSON format
-----------------------------
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
- ``NOTION_PAPER_TRACKER_DB_ID`` env var must be set.
- Second Brain DB must have:
    ``Sources`` (Relation), ``Interpretation`` (Rich Text),
    ``Proof Idea`` (Rich Text), ``Named Tools`` (Multi-select),
    ``Aliases`` (Rich Text), ``Assumptions`` (Rich Text),
    ``Statement LaTeX`` (Rich Text), ``Verified`` (Checkbox),
    ``Last Verified At`` (Date), ``Type`` (Select),
    ``Note Level`` (Select).
- Edges DB must have:
    ``Name`` (Title), ``From Concept`` (Relation), ``To Concept`` (Relation),
    ``Relation Type`` (Select), ``Rationale`` (Rich Text),
    ``AI Confidence`` (Number), ``Created By`` (Select),
    ``Status`` (Select), ``Source Papers`` (Relation).
- Knowledge Inbox DB must have:
    ``Promotion Target`` (Relation → Second Brain),
    ``Corrected Title`` (Rich Text).

If ``NOTION_EDGES_DB_ID`` is not set the engine logs a warning and exits
cleanly — other pipeline jobs are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Optional

import requests

from .notion_client_wrapper import NotionClientWrapper
from .vector_index import VectorIndexEngine
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

    def __init__(self, vector_index: Optional[VectorIndexEngine]) -> None:
        self.notion = NotionClientWrapper()
        self.paper_tracker_db   = os.environ.get("NOTION_PAPER_TRACKER_DB_ID", "")
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db    = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.deferred_edges_db = os.environ.get("NOTION_DEFERRED_EDGES_DB_ID", "")
        self.edges_db: str      = os.environ.get("NOTION_EDGES_DB_ID", "")
        self.zotero_user_id     = os.environ.get("ZOTERO_USER_ID", "")
        self.zotero_api_key     = os.environ.get("ZOTERO_API_KEY", "")
        # title -> SB page_id; populated once per run(), augmented on promotion.
        self._sb_title_cache: dict[str, str] = {}
        # Module 7: VectorIndexEngine — only active when VECTOR_INDEX_ENABLED is set.
        self._vector_index: VectorIndexEngine | None = vector_index if vector_index else None
    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Poll Paper Tracker for papers at s2-read.

        For each such paper: promotes all verified KI concepts as a batch,
        syncs Zotero notes, then advances the paper to s3-distilled.

        PromotionEngine no longer polls the Knowledge Inbox independently.
        """
        if not self.edges_db:
            logger.warning(
                "PromotionEngine: NOTION_EDGES_DB_ID is not set — "
                "skipping promotion run."
            )
            return
        if not self.paper_tracker_db:
            logger.warning(
                "PromotionEngine: NOTION_PAPER_TRACKER_DB_ID is not set — "
                "skipping promotion run."
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

        logger.info(
            "PromotionEngine: %d paper(s) ready for promotion.", len(papers)
        )

        # Seed the title cache once so all papers in the batch share it.
        self._sb_title_cache = self._build_sb_title_cache()
        logger.info(
            "PromotionEngine: Second Brain cache seeded with %d concept(s).",
            len(self._sb_title_cache),
        )


        for paper_page in papers:
            try:
                self._promote_paper(paper_page)
            except Exception:
                logger.exception(
                    "PromotionEngine: failed to promote paper %s",
                    paper_page["id"],
                )
        

    def _defer_edge(
        self,
        from_sb_id: str,
        target_title: str,
        relation_type: str,
        rationale: str,
        confidence: float,
        source_paper_ids: list[str],
    ) -> None:
        if not self.deferred_edges_db:
            logger.warning(
                "PromotionEngine: NOTION_DEFERRED_EDGES_DB_ID not set — "
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
                "PromotionEngine: deferred edge -[%s]-> '%s' written to Notion.",
                relation_type, target_title,
            )
        except Exception:
            logger.exception(
                "PromotionEngine: failed to write deferred edge -[%s]-> '%s'.",
                relation_type, target_title,
            )


    def _get_deferred_edges_for_target(self, target_title: str) -> list[dict]:
        """Query Deferred Edges DB for pending rows whose Target Title matches."""
        if not self.deferred_edges_db:
            return []
        return self.notion.query_database(
            self.deferred_edges_db,
            filter={
                "and": [
                    {"property": "Target Title", "rich_text": {"equals": target_title}},
                    {"property": "Status",       "select":    {"equals": "pending"}},
                ]
            },
        )


    def _resolve_deferred_edges(self, target_title: str, to_sb_id: str) -> None:
        """
        Find all pending deferred edges whose Target Title matches the
        newly promoted concept and create Edges DB rows for them.
        """
        rows = self._get_deferred_edges_for_target(target_title)
        if not rows:
            return

        logger.info(
            "PromotionEngine: resolving %d deferred edge(s) → '%s'.",
            len(rows), target_title,
        )
        for row in rows:
            row_id   = row["id"]
            props    = row["properties"]
            rel_type = self._get_select(props, "Relation Type")
            rationale   = self._get_text(props, "Rationale")
            confidence  = props.get("AI Confidence", {}).get("number") or 0.0
            from_ids    = self._get_relation(props, "From Concept")
            paper_ids   = self._get_relation(props, "Source Papers")

            if not from_ids:
                logger.warning(
                    "PromotionEngine: deferred edge %s has no From Concept — skipping.",
                    row_id,
                )
                self._mark_deferred_stale(row_id)
                continue

            try:
                self._create_edge(
                    from_sb_id=from_ids[0],
                    to_sb_id=to_sb_id,
                    relation_type=rel_type,
                    rationale=rationale,
                    confidence=confidence,
                    source_paper_ids=paper_ids,
                )
                self._mark_deferred_resolved(row_id, to_sb_id)
                logger.info(
                    "PromotionEngine: resolved deferred edge %s -[%s]-> '%s'.",
                    from_ids[0], rel_type, target_title,
                )
            except Exception:
                logger.exception(
                    "PromotionEngine: failed to resolve deferred edge %s.", row_id
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


    def _resolve_all_deferred(self) -> None:
        """
        At the start of each run(), sweep all pending deferred edges against
        the current SB title cache. Catches edges deferred in previous runs
        whose targets have since been promoted.
        """
        if not self.deferred_edges_db:
            return
        all_pending = self.notion.query_database(
            self.deferred_edges_db,
            filter={"property": "Status", "select": {"equals": "pending"}},
        )
        if not all_pending:
            return

        logger.info(
            "PromotionEngine: sweeping %d pending deferred edge(s) from previous runs.",
            len(all_pending),
        )
        for row in all_pending:
            props        = row["properties"]
            target_title = self._get_text(props, "Target Title")
            if not target_title:
                self._mark_deferred_stale(row["id"])
                continue

            to_sb_id = self._sb_title_cache.get(target_title)
            if not to_sb_id:
                continue  # Target still not promoted — leave pending

            from_ids   = self._get_relation(props, "From Concept")
            rel_type   = self._get_select(props, "Relation Type")
            rationale  = self._get_text(props, "Rationale")
            confidence = props.get("AI Confidence", {}).get("number") or 0.0
            paper_ids  = self._get_relation(props, "Source Papers")

            if not from_ids:
                self._mark_deferred_stale(row["id"])
                continue

            try:
                self._create_edge(
                    from_sb_id=from_ids[0],
                    to_sb_id=to_sb_id,
                    relation_type=rel_type,
                    rationale=rationale,
                    confidence=confidence,
                    source_paper_ids=paper_ids,
                )
                self._mark_deferred_resolved(row["id"], to_sb_id)
                logger.info(
                    "PromotionEngine: resolved stale deferred edge → '%s'.", target_title
                )
            except Exception:
                logger.exception(
                    "PromotionEngine: failed to resolve stale deferred edge %s.", row["id"]
                )
                
    # ── Per-paper promotion ───────────────────────────────────────────────────

    def _promote_paper(self, paper_page: dict) -> None:
        """
        Promote all verified KI concepts for one paper, sync Zotero notes,
        and advance the paper to s3-distilled.

        Partial approval (some concepts unreviewed) is logged but does NOT
        block promotion — humans approve individual concepts.
        """
        paper_id    = paper_page["id"]
        paper_props = paper_page["properties"]
        paper_title = self._get_page_title(paper_page) or paper_id

        # Fetch ALL KI concepts for this paper (for logging only).
        all_ki = self._fetch_ki_concepts_for_paper(paper_id)
        verified_ki = [
            p for p in all_ki
            if self._get_select(p["properties"], "verification_status") == "verified"
        ]
        rejected_ki = [
            p for p in all_ki
            if self._get_select(p["properties"], "verification_status") == "rejected"
        ]
        total_ki = len(all_ki)
        n_verified = len(verified_ki)
        n_rejected = len(rejected_ki)
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
                self._sync_zotero_notes(paper_id, paper_props)
            except Exception:
                logger.warning(
                    "PromotionEngine: Zotero note sync failed for paper '%s' — "
                    "continuing to s3-distilled.",
                    paper_title,
                )
            self.notion.update_page(
                page_id=paper_id,
                properties={"Status": self.notion.status_prop("s3-distilled")},
            )
            return

        try:
            # ── Pass 1: promote concept nodes ────────────────────────────────────
            promoted: dict[str, str] = {}  # ki_page_id → sb_page_id
            for item in verified_ki:
                try:
                    sb_page_id = self._promote_concept_node(item)
                    self._resolve_all_deferred() 
                    if sb_page_id:
                        promoted[item["id"]] = sb_page_id
                except Exception:
                    logger.exception(
                        "PromotionEngine [pass 1]: failed for KI page %s", item["id"]
                    )

            logger.info(
                "PromotionEngine: pass 1 complete — %d/%d concept(s) promoted.",
                len(promoted), n_verified,
            )

            # ── Pass 2: create edges ─────────────────────────────────────────────
            total_edges = 0
            for item in verified_ki:
                ki_page_id = item["id"]
                sb_page_id = promoted.get(ki_page_id)
                if not sb_page_id:
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

            # ── Mark promoted KI pages ───────────────────────────────────────────
            for ki_page_id in promoted:
                try:
                    self.notion.update_page(
                        page_id=ki_page_id,
                        properties={"Status": self.notion.select_prop("Promoted")},
                    )
                except Exception:
                    logger.warning(
                        "PromotionEngine: could not mark KI page %s as Promoted.",
                        ki_page_id,
                    )

            # ── Zotero note sync (REQ-7) — failure must never block promotion ────
            try:
                self._sync_zotero_notes(paper_id, paper_props)
            except Exception:
                logger.warning(
                    "PromotionEngine: Zotero note sync failed for paper '%s' — "
                    "continuing to s3-distilled.",
                    paper_title,
                )

        finally:
            # ── Advance paper to s3-distilled — always, even on partial failure ──
            try:
                self.notion.update_page(
                    page_id=paper_id,
                    properties={"Status": self.notion.status_prop("s3-distilled")},
                )
                logger.info(
                    "PromotionEngine: paper '%s' → s3-distilled.", paper_title
                )
            except Exception:
                logger.exception(
                    "PromotionEngine: could not advance paper '%s' to s3-distilled.",
                    paper_title,
                )

    def _fetch_ki_concepts_for_paper(self, paper_page_id: str) -> list[dict]:
        """Return all KI pages whose Source Paper relation includes paper_page_id."""
        return self.notion.query_database(
            self.knowledge_inbox_db,
            filter={
                "property": "Source Paper",
                "relation": {"contains": paper_page_id},
            },
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
        # REQ-8: Prefer Corrected Title (human override) over the AI-generated name.
        corrected_title  = self._get_text(props, "Corrected Title").strip()
        title            = corrected_title if corrected_title else self._get_page_title(item)
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
        # edges pointing to this concept by title.  Store both the full title
        # (e.g. "[Theorem] Foo") and the stripped form ("Foo") so that edge
        # target_title values without a type-prefix still resolve.
        self._sb_title_cache[title] = sb_page_id
        self._sb_title_cache[re.sub(r'^\[[^\]]+\]\s*', '', title).strip()] = sb_page_id

        # Resolve any edges that were waiting for this concept
        self._resolve_deferred_edges(title, sb_page_id)
        self._resolve_deferred_edges(re.sub(r'^\[[^\]]+\]\s*', '', title).strip(), sb_page_id)

        # Module 7: migrate the vector index entry from KI to SB.
        if self._vector_index and self._vector_index.available:
            try:
                self._vector_index.promote_concept(ki_page_id, sb_page_id)
            except Exception:
                logger.warning(
                    "PromotionEngine: vector promote failed for KI %s → SB %s.",
                    ki_page_id, sb_page_id,
                )

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
                self._defer_edge(
                    from_sb_id=from_sb_id,
                    target_title=target_title,
                    relation_type=rel_type,
                    rationale=rationale,
                    confidence=confidence,
                    source_paper_ids=source_paper_ids,
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

    # ── Zotero note sync (REQ-7) ──────────────────────────────────────────────

    def _get_zotero_key(self, props: dict) -> str | None:
        """
        Resolve the Zotero item key from paper page properties.

        Checks in order:
        1. ``Item Key`` (rich text)
        2. ``Zotero URI`` (url) — parse last path segment
        3. ``Key`` (rich text)
        """
        for prop_name in ("Item Key", "Key"):
            val = self._get_text(props, prop_name).strip()
            if val and re.match(r'^[A-Z0-9]{8}$', val, re.IGNORECASE):
                return val.upper()

        # Try parsing from Zotero URI.
        uri = props.get("Zotero URI", {}).get("url") or self._get_text(props, "Zotero URI")
        if uri:
            match = re.search(r'/items/([A-Z0-9]{8})(?:/|$)', uri, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _fetch_zotero_notes(self, zotero_item_key: str) -> list[dict]:
        """
        Fetch note child items from Zotero API.
        Returns list of dicts with keys: key, title, content (HTML), date_modified.
        """
        if not self.zotero_user_id or not self.zotero_api_key:
            return []
        url = (
            f"https://api.zotero.org/users/{self.zotero_user_id}"
            f"/items/{zotero_item_key}/children"
        )
        try:
            resp = requests.get(
                url,
                headers={"Zotero-API-Key": self.zotero_api_key},
                params={"itemType": "note"},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception:
            logger.warning(
                "PromotionEngine: failed to fetch Zotero notes for key %s",
                zotero_item_key,
            )
            return []

        notes = []
        for item in items:
            data = item.get("data", {})
            if data.get("itemType") == "note":
                notes.append({
                    "key": item.get("key", ""),
                    "title": data.get("title", ""),
                    "content": data.get("note", ""),
                    "date_modified": data.get("dateModified", ""),
                })
        return notes

    def _fetch_zotero_annotations(self, zotero_item_key: str) -> list[dict]:
        """
        Fetch annotation child items (highlights + margin comments) from Zotero API.
        Returns list of dicts with keys: key, type, comment, text, pageLabel, date_modified.
        """
        if not self.zotero_user_id or not self.zotero_api_key:
            return []
        url = (
            f"https://api.zotero.org/users/{self.zotero_user_id}"
            f"/items/{zotero_item_key}/children"
        )
        try:
            resp = requests.get(
                url,
                headers={"Zotero-API-Key": self.zotero_api_key},
                params={"itemType": "annotation"},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception:
            logger.warning(
                "PromotionEngine: failed to fetch Zotero annotations for key %s",
                zotero_item_key,
            )
            return []

        annotations = []
        for item in items:
            data = item.get("data", {})
            if data.get("itemType") == "annotation":
                annotations.append({
                    "key": item.get("key", ""),
                    "type": data.get("annotationType", ""),
                    "comment": data.get("comment", ""),
                    "text": data.get("annotationText", ""),
                    "pageLabel": data.get("pageLabel", ""),
                    "date_modified": data.get("dateModified", ""),
                })
        return annotations

    def _zotero_html_to_blocks(self, html_content: str) -> list[dict]:
        """
        Convert Zotero note HTML to a list of Notion blocks.

        Handles: <p>, <h1>/<h2>/<h3>, <ul>/<ol>/<li>, <br>, <strong>, <em>, <a>.
        """

        class _Parser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.blocks: list[dict] = []
                self._text = ""
                self._list_type: str | None = None
                self._in_tag: str | None = None

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag in ("p", "h1", "h2", "h3", "li"):
                    self._text = ""
                    self._in_tag = tag
                elif tag == "br":
                    self._text += "\n"
                elif tag == "ul":
                    self._list_type = "ul"
                elif tag == "ol":
                    self._list_type = "ol"

            def handle_endtag(self, tag: str) -> None:
                text = self._text.strip()
                if not text:
                    return
                if tag == "p":
                    self.blocks.append({
                        "object": "block", "type": "paragraph",
                        "paragraph": {"rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ]},
                    })
                elif tag in ("h1", "h2", "h3"):
                    level = int(tag[1])
                    bt = f"heading_{min(level, 3)}"
                    self.blocks.append({
                        "object": "block", "type": bt,
                        bt: {"rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ]},
                    })
                elif tag == "li":
                    bt = "numbered_list_item" if self._list_type == "ol" else "bulleted_list_item"
                    self.blocks.append({
                        "object": "block", "type": bt,
                        bt: {"rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ]},
                    })
                elif tag in ("ul", "ol"):
                    self._list_type = None

            def handle_data(self, data: str) -> None:
                self._text += data

        parser = _Parser()
        try:
            parser.feed(html_content)
        except Exception:
            pass
        # Flush any remaining text as a paragraph.
        if parser._text.strip():
            parser.blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": parser._text.strip()[:2000]}}
                ]},
            })
        return parser.blocks

    def _sync_zotero_notes(self, paper_page_id: str, props: dict) -> None:
        """
        Fetch Zotero notes and annotations for the paper and append any new
        ones to the Notion paper page.

        Idempotent: reads existing blocks to collect already-synced Zotero
        keys (stored in callout block headers) and skips duplicates.
        """
        zotero_key = self._get_zotero_key(props)
        if not zotero_key:
            logger.warning(
                "PromotionEngine: no Zotero key found for paper page %s — "
                "skipping note sync.",
                paper_page_id,
            )
            return

        # Collect already-synced keys from existing blocks.
        try:
            existing_blocks = self.notion.get_block_children(paper_page_id)
        except Exception:
            existing_blocks = []

        synced_keys: set[str] = set()
        for block in existing_blocks:
            bt = block.get("type")
            if bt == "callout":
                rt = block.get("callout", {}).get("rich_text", [])
                text = "".join(seg.get("plain_text", "") for seg in rt)
                m = re.search(r'\[zotero:([A-Z0-9]{8})\]', text)
                if m:
                    synced_keys.add(m.group(1))

        notes = self._fetch_zotero_notes(zotero_key)
        annotations = self._fetch_zotero_annotations(zotero_key)

        new_blocks: list[dict] = []

        # Sync notes.
        for note in notes:
            note_key = note.get("key", "")
            if note_key in synced_keys:
                continue
            title = note.get("title") or "Zotero Note"
            content_html = note.get("content", "")
            # Callout header identifies the note for idempotency.
            header = f"[zotero:{note_key}] {title}"
            new_blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": header[:2000]}}],
                    "icon": {"type": "emoji", "emoji": "📝"},
                    "color": "gray_background",
                },
            })
            if content_html:
                new_blocks.extend(self._zotero_html_to_blocks(content_html))

        # Sync annotations.
        for ann in annotations:
            ann_key = ann.get("key", "")
            if ann_key in synced_keys:
                continue
            page_label = ann.get("pageLabel", "?")
            highlighted = ann.get("text", "").strip()
            comment = ann.get("comment", "").strip()

            if highlighted:
                quote_text = f"[p.{page_label}] {highlighted}"
                new_blocks.append({
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [{"type": "text", "text": {"content": quote_text[:2000]}}],
                        "color": "yellow_background",
                    },
                })
                # Tag for idempotency.
                new_blocks.append({
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"type": "text", "text": {
                            "content": f"[zotero:{ann_key}]"[:200]
                        }}],
                        "icon": {"type": "emoji", "emoji": "🔖"},
                        "color": "yellow_background",
                    },
                })
            if comment:
                new_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [
                        {"type": "text", "text": {"content": comment[:2000]}}
                    ]},
                })

        if not new_blocks:
            logger.info(
                "PromotionEngine: no new Zotero notes/annotations for paper %s.",
                paper_page_id,
            )
            return

        # Append in batches of 100.
        for i in range(0, len(new_blocks), 100):
            self.notion.append_block_children(paper_page_id, new_blocks[i:i + 100])

        logger.info(
            "PromotionEngine: synced %d Zotero block(s) to paper page %s.",
            len(new_blocks), paper_page_id,
        )

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
        cache: dict[str, str] = {}
        for page in pages:
            title = self._get_page_title(page)
            if not title:
                continue
            cache[title] = page["id"]
            cache[re.sub(r'^\[[^\]]+\]\s*', '', title).strip()] = page["id"]
        return cache

    def _create_sb_concept(self, ki_item: dict) -> str | None:
        """
        Create a new Second Brain Concept page from a Knowledge Inbox item.
        Returns the new page ID, or None on failure.
        """
        props        = ki_item["properties"]
        # REQ-8: Use Corrected Title if set, fall back to page title.
        corrected    = self._get_text(props, "Corrected Title").strip()
        title        = corrected if corrected else self._get_page_title(ki_item)
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
            sb_props["Sources"] = self.notion.relation_prop(source_ids)
        if concept_type:
            sb_props["Type"] = self.notion.select_prop(concept_type)

        # Assumptions and Statement LaTeX are always written — even when empty —
        # so that rebuild() can read them back from Notion and log debug warnings
        # for degraded embeddings rather than silently missing properties.
        sb_props["Assumptions"] = {
            "rich_text": self.notion.rich_text(self._get_text(props, "Assumptions"))
        }
        sb_props["Statement LaTeX"] = {
            "rich_text": self.notion.rich_text(self._get_text(props, "Statement LaTeX"))
        }

        for ki_key, sb_key in [
            ("Interpretation",  "Interpretation"),
            ("Proof Idea",      "Proof Idea"),
            ("Aliases",         "Aliases"),
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
            "Status":        self.notion.select_prop("pending"),
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
    

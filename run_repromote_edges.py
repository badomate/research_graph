#!/usr/bin/env python3
"""
run_repromote_edges.py
─────────────────────────────────────────────────────────────────────────────
Re-promote edges for already-promoted KI concepts.

Use this AFTER running `run_reprocess_paper.py --edges` (which refreshes the
Edge Suggestions JSON on KI pages).  The concepts are already in the Second
Brain — this script ONLY runs Pass 2 (edge creation) without touching concept
nodes.

For each verified + promoted KI concept linked to the given paper:
  1. Reads the current Edge Suggestions JSON from the KI page.
  2. Resolves the SB page ID from the KI page's Promotion Target relation.
  3. Optionally archives all existing Edges DB rows whose Source Papers
     contain this paper (--delete-existing).
  4. Creates new Edges DB rows from the current Edge Suggestions.
  5. Resolves any pending deferred edges whose targets are now in the SB.

Usage:
  # Preview — show what would be created / deleted without writing anything:
  python run_repromote_edges.py <paper-page-id> --dry-run

  # Create new edges on top of existing ones:
  python run_repromote_edges.py <paper-page-id>

  # Archive old edges for this paper first, then create new ones (recommended):
  python run_repromote_edges.py <paper-page-id> --delete-existing

  # Scope deletion more narrowly: only delete edges whose From Concept is one
  # of the SB pages produced from this paper's KI concepts (rather than every
  # edge whose Source Papers contains the paper):
  python run_repromote_edges.py <paper-page-id> --delete-existing --by-from-concept
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── path / env setup ──────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from modules.notion_client_wrapper import NotionClientWrapper  # noqa: E402
from modules.promotion import PromotionEngine  # noqa: E402
from modules.vector_index import VectorIndexEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Edge deletion helpers ─────────────────────────────────────────────────────

def _find_edges_by_source_paper(
    notion: NotionClientWrapper,
    edges_db: str,
    paper_page_id: str,
) -> list[dict]:
    """
    Return all Edges DB rows whose Source Papers relation contains paper_page_id.
    """
    return notion.query_database(
        edges_db,
        filter={
            "property": "Source Papers",
            "relation": {"contains": paper_page_id},
        },
    )


def _find_edges_by_from_concept(
    notion: NotionClientWrapper,
    edges_db: str,
    from_sb_id: str,
) -> list[dict]:
    """
    Return all Edges DB rows whose From Concept relation contains from_sb_id.
    """
    return notion.query_database(
        edges_db,
        filter={
            "property": "From Concept",
            "relation": {"contains": from_sb_id},
        },
    )


def _archive_edge(
    notion: NotionClientWrapper,
    edge_page_id: str,
    dry_run: bool = False,
) -> None:
    if dry_run:
        log.info("[DRY RUN]  Would archive edge %s", edge_page_id)
        return
    try:
        notion.update_page(page_id=edge_page_id, properties={}, archived=True)
        log.info("Archived edge %s", edge_page_id)
    except Exception as exc:
        log.warning("Could not archive edge %s: %s", edge_page_id, exc)


# ── Property helpers ──────────────────────────────────────────────────────────

def _get_page_title(page: dict) -> str:
    for v in page.get("properties", {}).values():
        if isinstance(v, dict) and v.get("type") == "title":
            try:
                return v["title"][0]["plain_text"]
            except (KeyError, IndexError):
                return ""
    return ""


def _get_select(props: dict, key: str) -> str:
    try:
        return props[key]["select"]["name"] or ""
    except (KeyError, TypeError):
        return ""


def _get_promotion_target(props: dict) -> str | None:
    try:
        targets = props["Promotion Target"]["relation"]
        return targets[0]["id"] if targets else None
    except (KeyError, TypeError):
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def repromote_edges(
    paper_page_id: str,
    delete_existing: bool = False,
    by_from_concept: bool = False,
    dry_run: bool = False,
) -> None:
    notion = NotionClientWrapper()
    edges_db = os.environ.get("NOTION_EDGES_DB_ID", "")
    ki_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]

    if not edges_db:
        log.error("NOTION_EDGES_DB_ID is not set — aborting.")
        sys.exit(1)

    # ── Fetch paper page ──────────────────────────────────────────────────────
    log.info("Fetching paper page %s ...", paper_page_id)
    paper_page = notion.get_page(paper_page_id)
    paper_title = _get_page_title(paper_page["properties"]) or paper_page_id
    log.info("Paper: %s", paper_title)

    # ── Initialise PromotionEngine ────────────────────────────────────────────
    vector_index: VectorIndexEngine | None = None
    if os.environ.get("VECTOR_INDEX_ENABLED"):
        try:
            vector_index = VectorIndexEngine()
            log.info("VectorIndexEngine: available=%s", vector_index.available)
        except Exception:
            log.warning("VectorIndexEngine init failed — continuing without Qdrant.")

    engine = PromotionEngine(vector_index=vector_index)

    # ── Seed Second Brain title cache ─────────────────────────────────────────
    log.info("Building Second Brain title cache ...")
    engine._sb_title_cache = engine._build_sb_title_cache()
    log.info("Title cache: %d concept(s).", len(engine._sb_title_cache))

    # ── Fetch KI pages for this paper ────────────────────────────────────────
    log.info("Querying Knowledge Inbox for KI concepts of this paper ...")
    all_ki = notion.query_database(
        ki_db,
        filter={"property": "Source Paper", "relation": {"contains": paper_page_id}},
    )
    log.info("Found %d KI page(s) total.", len(all_ki))

    # Filter to concepts that are already promoted (have a Promotion Target).
    promoted_ki = [
        p for p in all_ki
        if _get_promotion_target(p["properties"]) is not None
    ]
    log.info(
        "%d promoted KI page(s) will be processed (%d have no Promotion Target — skipped).",
        len(promoted_ki),
        len(all_ki) - len(promoted_ki),
    )

    if not promoted_ki:
        log.warning(
            "No promoted KI pages found for this paper.\n"
            "  • If the paper was never promoted, run the normal promotion flow instead.\n"
            "  • If concepts are promoted but Promotion Target is unset, re-run promotion."
        )
        return

    # ── Optional: delete existing edges ──────────────────────────────────────
    if delete_existing:
        if by_from_concept:
            # Narrow deletion: only edges whose From Concept is one of the SB
            # pages produced from this paper's KI concepts.
            log.info(
                "--delete-existing --by-from-concept: archiving edges per From Concept ..."
            )
            for ki_page in promoted_ki:
                ki_title = _get_page_title(ki_page["properties"])
                from_sb_id = _get_promotion_target(ki_page["properties"])
                if not from_sb_id:
                    continue
                stale = _find_edges_by_from_concept(notion, edges_db, from_sb_id)
                log.info(
                    "  '%s' → %d edge(s) to archive.", ki_title, len(stale)
                )
                for edge in stale:
                    _archive_edge(notion, edge["id"], dry_run=dry_run)
        else:
            # Broad deletion: all edges whose Source Papers contains this paper.
            log.info(
                "--delete-existing: archiving all Edges DB rows for this paper ..."
            )
            stale = _find_edges_by_source_paper(notion, edges_db, paper_page_id)
            log.info("%d existing edge(s) found.", len(stale))
            for edge in stale:
                _archive_edge(notion, edge["id"], dry_run=dry_run)

    # ── Create new edges ──────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Creating new edges for %d promoted KI concept(s) ...", len(promoted_ki))

    total_edges = 0

    for ki_page in promoted_ki:
        ki_id = ki_page["id"]
        ki_props = ki_page["properties"]
        ki_title = _get_page_title(ki_props)
        from_sb_id = _get_promotion_target(ki_props)

        if not from_sb_id:
            log.warning("  '%s' has no Promotion Target — skipping.", ki_title)
            continue

        verification = _get_select(ki_props, "verification_status")
        if verification == "rejected":
            log.info("  '%s' is rejected — skipping.", ki_title)
            continue

        # Parse edge suggestions.
        edges = engine._parse_edge_suggestions(ki_props, ki_id)
        log.info("  '%s' — %d edge proposal(s) in Edge Suggestions.", ki_title, len(edges))

        if not edges:
            log.info("  '%s' — no Edge Suggestions, nothing to do.", ki_title)
            continue

        if dry_run:
            for e in edges:
                target_sb_id = engine._sb_title_cache.get(e["target_title"])
                status = "→ SB resolves" if target_sb_id else "→ would defer"
                log.info(
                    "[DRY RUN]    %s -[%s]-> '%s' (conf=%.2f, needs_review=%s) %s",
                    ki_title, e["relation_type"], e["target_title"],
                    e["confidence"], e.get("needs_review", False), status,
                )
            continue

        # Use PromotionEngine's Pass 2 method directly.
        try:
            n = engine._promote_concept_edges(ki_page, from_sb_id)
            total_edges += n
        except Exception:
            log.exception("  '%s' — edge promotion failed.", ki_title)

    if dry_run:
        log.info("═" * 60)
        log.info("DRY RUN complete — no changes written.")
        return

    # ── Sweep deferred edges ──────────────────────────────────────────────────
    log.info("Sweeping deferred edges ...")
    try:
        engine._resolve_all_deferred()
    except Exception:
        log.warning("Deferred edge sweep failed — continuing.", exc_info=True)

    log.info("═" * 60)
    log.info(
        "Done.  %d new edge(s) created for paper: %s", total_edges, paper_title
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-promote edges for already-promoted KI concepts. "
            "Use after run_reprocess_paper.py --edges has refreshed Edge Suggestions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paper_page_id",
        help="Notion Paper Tracker page ID (32-char hex, with or without dashes).",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help=(
            "Archive existing Edges DB rows for this paper before creating new ones. "
            "By default archives all rows where Source Papers contains the paper. "
            "Use --by-from-concept to narrow the scope."
        ),
    )
    parser.add_argument(
        "--by-from-concept",
        action="store_true",
        help=(
            "When --delete-existing is set, only archive edges whose From Concept "
            "matches one of the SB pages promoted from this paper's KI concepts "
            "(rather than every edge tagged with this paper in Source Papers)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created / deleted without making any API calls.",
    )

    args = parser.parse_args()

    if args.by_from_concept and not args.delete_existing:
        parser.error("--by-from-concept requires --delete-existing")

    repromote_edges(
        paper_page_id=args.paper_page_id,
        delete_existing=args.delete_existing,
        by_from_concept=args.by_from_concept,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

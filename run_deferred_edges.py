"""
run_deferred_edges.py — resolve all pending rows in the Deferred Edges DB.

The stored Target Title values often have a hub suffix appended by the
linking stage, e.g.:
    "McKean-Vlasov SDE Wellposedness [Mean Field Games]"

This script strips that suffix before matching against the Second Brain
title cache, so those edges get resolved.

Usage:
    python run_deferred_edges.py [--dry-run]

    --dry-run   Print what would be done without writing anything to Notion.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from modules.notion_client_wrapper import NotionClientWrapper  # noqa: E402
from modules.promotion import PromotionEngine  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Print actions without writing to Notion.")
args = parser.parse_args()
DRY_RUN: bool = args.dry_run

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

if DRY_RUN:
    logger.info("DRY-RUN mode — no Notion writes will be performed.")


# ── Helpers ───────────────────────────────────────────────────────────────────

# Strip trailing "[Hub Name]" — e.g. "Foo Bar [Mean Field Games]" → "Foo Bar"
_HUB_SUFFIX_RE = re.compile(r'\s*\[[^\]]+\]\s*$')

def _strip_hub_suffix(title: str) -> str:
    return _HUB_SUFFIX_RE.sub("", title).strip()


def _strip_type_prefix(title: str) -> str:
    """Strip leading '[Type]' prefix — e.g. '[Theorem] Foo' → 'Foo'."""
    return re.sub(r'^\[[^\]]+\]\s*', '', title).strip()


def _candidate_titles(raw: str) -> list[str]:
    """Return all title variants to try when looking up the SB cache."""
    stripped_hub    = _strip_hub_suffix(raw)
    stripped_type   = _strip_type_prefix(raw)
    stripped_both   = _strip_type_prefix(stripped_hub)
    # Deduplicate while preserving order.
    seen, out = set(), []
    for t in (raw, stripped_hub, stripped_type, stripped_both):
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

engine = PromotionEngine(vector_index=None)
notion = engine.notion

# 1. Build the full SB title cache (title → page_id).
logger.info("Building Second Brain title cache ...")
sb_cache = engine._build_sb_title_cache()
logger.info("  %d SB concept(s) in cache.", len(sb_cache))

# 2. Fetch all pending deferred edges.
if not engine.deferred_edges_db:
    logger.error("NOTION_DEFERRED_EDGES_DB_ID is not set in .env — aborting.")
    sys.exit(1)
if not engine.edges_db:
    logger.error("NOTION_EDGES_DB_ID is not set in .env — aborting.")
    sys.exit(1)

logger.info("Fetching pending deferred edges ...")
pending = notion.query_database(
    engine.deferred_edges_db,
    filter={"property": "Status", "select": {"equals": "pending"}},
)
logger.info("  %d pending row(s) found.", len(pending))

if not pending:
    logger.info("Nothing to do.")
    sys.exit(0)

# 3. Resolve each row.
resolved = stale = skipped = 0

for row in pending:
    row_id     = row["id"]
    props      = row["properties"]

    raw_target = engine._get_text(props, "Target Title")
    rel_type   = engine._get_select(props, "Relation Type")
    rationale  = engine._get_text(props, "Rationale")
    confidence = props.get("AI Confidence", {}).get("number") or 0.0
    from_ids   = engine._get_relation(props, "From Concept")
    paper_ids  = engine._get_relation(props, "Source Papers")

    if not raw_target:
        logger.warning("Row %s has no Target Title — marking stale.", row_id)
        if not DRY_RUN:
            engine._mark_deferred_stale(row_id)
        stale += 1
        continue

    if not from_ids:
        logger.warning("Row %s has no From Concept — marking stale.", row_id)
        if not DRY_RUN:
            engine._mark_deferred_stale(row_id)
        stale += 1
        continue

    # Try all title variants (with/without hub suffix, with/without type prefix).
    to_sb_id: str | None = None
    matched_title: str | None = None
    for candidate in _candidate_titles(raw_target):
        to_sb_id = sb_cache.get(candidate)
        if to_sb_id:
            matched_title = candidate
            break

    if not to_sb_id:
        logger.info(
            "SKIP  '%s' — target not found in Second Brain (tried: %s).",
            raw_target, _candidate_titles(raw_target),
        )
        skipped += 1
        continue

    logger.info(
        "RESOLVE  '%s'  →  matched as '%s'  -[%s]->  SB:%s",
        raw_target, matched_title, rel_type, to_sb_id,
    )

    if DRY_RUN:
        resolved += 1
        continue

    try:
        engine._create_edge(
            from_sb_id=from_ids[0],
            to_sb_id=to_sb_id,
            relation_type=rel_type,
            rationale=rationale,
            confidence=confidence,
            source_paper_ids=paper_ids,
        )
        engine._mark_deferred_resolved(row_id, to_sb_id)
        resolved += 1
    except Exception:
        logger.exception("Failed to resolve row %s — leaving as pending.", row_id)

# 4. Summary.
logger.info(
    "Done.  resolved=%d  stale=%d  still-pending=%d",
    resolved, stale, skipped,
)

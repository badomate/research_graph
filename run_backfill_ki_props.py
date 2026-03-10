#!/usr/bin/env python3
"""
run_backfill_ki_props.py
─────────────────────────────────────────────────────────────────────────────
Read the body of every KI concept page and backfill the following DB
properties from the heading-labelled sections in the page body:

  Page body section   →  Notion DB property
  ─────────────────       ─────────────────────
  ## Assumptions      →  Assumptions          (rich_text)
  ## Statement        →  Statement LaTeX      (rich_text)
  ## Conclusion       →  Summary              (rich_text)
  ## Interpretation   →  Interpretation       (rich_text)
  ## Proof Idea       →  Proof Idea           (rich_text)

Properties that already contain text are skipped (idempotent).

Usage:
  python run_backfill_ki_props.py [--dry-run] [--page-id <id> [<id> ...]]
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Heading → DB property mapping ─────────────────────────────────────────────
# Keys must match the exact heading_2 text written by ingestion.py.
SECTION_TO_PROP: dict[str, str] = {
    "Assumptions":    "Assumptions",
    "Statement":      "Statement LaTeX",
    "Interpretation": "Interpretation",
    "Proof Idea":     "Proof Idea",
}


# ── Block text extraction ──────────────────────────────────────────────────────

def _rich_text_to_str(rich_text: list[dict]) -> str:
    """Reconstruct plain text from a Notion rich_text array.

    Inline equation segments become $expression$ so LaTeX is preserved
    in the DB property value.
    """
    parts: list[str] = []
    for seg in rich_text:
        if seg.get("type") == "equation":
            parts.append(f"${seg['equation']['expression']}$")
        else:
            parts.append(seg.get("plain_text", ""))
    return "".join(parts)


def extract_sections(blocks: list[dict]) -> dict[str, str]:
    """Parse a flat Notion block list and return {section_heading: text}.

    Collects paragraph and standalone equation blocks under each heading_2
    section.  Non-content blocks (to_do, callout, etc.) are ignored unless
    they fall under a recognised heading.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for block in blocks:
        btype = block.get("type", "")

        if btype == "heading_2":
            rt = block.get("heading_2", {}).get("rich_text", [])
            current = _rich_text_to_str(rt).strip()
            sections.setdefault(current, [])

        elif btype == "paragraph" and current is not None:
            rt = block.get("paragraph", {}).get("rich_text", [])
            text = _rich_text_to_str(rt)
            if text.strip():
                sections[current].append(text)

        elif btype == "equation" and current is not None:
            expr = block.get("equation", {}).get("expression", "")
            if expr.strip():
                sections[current].append(f"$${expr}$$")

        # to_do, callout, bulleted_list_item, etc. — skip

    return {k: "\n".join(v).strip() for k, v in sections.items()}


# ── Notion property helpers ────────────────────────────────────────────────────

def _get_rich_text(props: dict, key: str) -> str:
    try:
        return "".join(seg.get("plain_text", "") for seg in props[key]["rich_text"])
    except (KeyError, TypeError):
        return ""


def _get_title(props: dict) -> str:
    for v in props.values():
        if isinstance(v, dict) and v.get("type") == "title":
            return "".join(seg.get("plain_text", "") for seg in v.get("title", []))
    return "(no title)"


# ── Main backfill logic ────────────────────────────────────────────────────────

def backfill(dry_run: bool = False, page_ids: list[str] | None = None) -> None:
    notion = NotionClientWrapper()
    ki_db  = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]

    if page_ids:
        pages = [notion.get_page(pid) for pid in page_ids]
        log.info("Operating on %d specified page(s).", len(pages))
    else:
        pages = notion.query_database(ki_db)
        log.info("Fetched %d KI pages from database.", len(pages))

    total_updated = 0
    total_already_full = 0
    total_no_body = 0
    total_failed = 0

    for page in pages:
        pid   = page["id"]
        props = page["properties"]
        title = _get_title(props)

        # Fetch page body
        try:
            blocks = notion.get_block_children(pid)
        except Exception as exc:
            log.warning("[%s] could not fetch blocks: %s", title, exc)
            total_failed += 1
            continue

        sections = extract_sections(blocks)

        # Build update — only properties that are currently empty
        update: dict[str, dict] = {}
        for heading, prop_name in SECTION_TO_PROP.items():
            body_text = sections.get(heading, "").strip()
            if not body_text:
                continue
            if _get_rich_text(props, prop_name).strip():
                continue  # already populated — skip
            update[prop_name] = {
                "rich_text": [{"type": "text", "text": {"content": body_text[:2000]}}]
            }

        if not update:
            total_already_full += 1
            log.debug("[%s] all target properties already set or body empty.", title)
            continue

        prop_names = sorted(update.keys())
        if dry_run:
            log.info("[DRY RUN] [%s] would write: %s", title, prop_names)
            total_updated += 1
            continue

        try:
            notion.update_page(pid, update)
            log.info("[%s] wrote: %s", title, prop_names)
            total_updated += 1
        except Exception as exc:
            log.warning("[%s] update failed: %s", title, exc)
            total_failed += 1

    log.info(
        "Finished.  updated=%d  already-set=%d  no-content=%d  failed=%d",
        total_updated, total_already_full, total_no_body, total_failed,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Backfill KI DB properties from page body heading sections.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print what would be written without making any Notion API calls.",
)
parser.add_argument(
    "--page-id",
    nargs="+",
    metavar="PAGE_ID",
    help="Only process these specific KI page IDs (space-separated).",
)
args = parser.parse_args()
backfill(dry_run=args.dry_run, page_ids=args.page_id)

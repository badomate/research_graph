#!/usr/bin/env python3
"""
run_reprocess_paper.py
─────────────────────────────────────────────────────────────────────────────
Re-process a paper given its Notion Paper Tracker page ID.

Two independent operations, each toggled with flags:

  --concepts   Re-run Stage 1 (LLM extraction) from the cached Markdown.
               Reads the Markdown that was produced by a prior pipeline run
               and stored in the Koofr Markdown cache.  Existing KI pages for
               this paper are archived before new ones are created (use
               --keep-existing to skip archiving).

  --edges      For each KI concept currently linked to this paper, re-run
               Stage 2 (candidate retrieval) + Stage 3 (LLM linking) and
               refresh the Edge Suggestions property and cross-paper edge
               section in the page body.  Old edge blocks are removed before
               the new section is written.

Both flags can be combined.  If neither is given, --edges is assumed.

Usage:
  # Re-run edges only (default):
  python run_reprocess_paper.py <page-id>

  # Re-run concept extraction only (archives existing KI pages first):
  python run_reprocess_paper.py <page-id> --concepts

  # Re-run both — archive old KI pages and refresh everything:
  python run_reprocess_paper.py <page-id> --concepts --edges

  # Re-run both, keep existing KI pages alongside the new ones:
  python run_reprocess_paper.py <page-id> --concepts --edges --keep-existing

  # Dry-run: show what would happen without any API calls:
  python run_reprocess_paper.py <page-id> --concepts --edges --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── path / env setup ──────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from orchestrator.modules.extraction_schema import MathObject, check_completeness  # noqa: E402
from orchestrator.modules.ingestion import IngestionEngine, _count_tokens  # noqa: E402
from orchestrator.modules.notion_client_wrapper import NotionClientWrapper  # noqa: E402
from orchestrator.modules.vector_index import VectorIndexEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Notion property helpers ────────────────────────────────────────────────────

def _get_text_prop(props: dict, key: str) -> str:
    """Extract plain text from a Notion rich_text or url property."""
    prop = props.get(key, {})
    if prop.get("type") == "url":
        return prop.get("url") or ""
    try:
        return "".join(seg.get("plain_text", "") for seg in prop["rich_text"])
    except (KeyError, TypeError):
        return ""


def _get_multi(props: dict, key: str) -> list[str]:
    try:
        return [o["name"] for o in props[key]["multi_select"]]
    except (KeyError, TypeError):
        return []


def _get_select(props: dict, key: str) -> str:
    try:
        return props[key]["select"]["name"] or ""
    except (KeyError, TypeError):
        return ""


def _get_number(props: dict, key: str) -> float:
    try:
        val = props[key]["number"]
        return float(val) if val is not None else 0.0
    except (KeyError, TypeError):
        return 0.0


def _get_page_title(props: dict) -> str:
    for v in props.values():
        if isinstance(v, dict) and v.get("type") == "title":
            return "".join(seg.get("plain_text", "") for seg in v.get("title", []))
    return ""


# ── Page body helpers ──────────────────────────────────────────────────────────

def _extract_body_sections(blocks: list[dict]) -> dict[str, str]:
    """
    Parse a flat Notion block list and return {section_heading: text}.

    Collects paragraph and standalone equation blocks under each heading_2.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for block in blocks:
        btype = block.get("type", "")
        if btype == "heading_2":
            rt = block.get("heading_2", {}).get("rich_text", [])
            current = "".join(seg.get("plain_text", "") for seg in rt).strip()
            sections.setdefault(current, [])
        elif btype == "paragraph" and current is not None:
            rt = block.get("paragraph", {}).get("rich_text", [])
            text = "".join(seg.get("plain_text", "") for seg in rt)
            if text.strip():
                sections[current].append(text)
        elif btype == "equation" and current is not None:
            expr = block.get("equation", {}).get("expression", "")
            if expr.strip():
                sections[current].append(f"$${expr}$$")
    return {k: "\n".join(v).strip() for k, v in sections.items()}


# ── Reconstruct MathObject from a KI Notion page ─────────────────────────────

def ki_page_to_math_object(
    page: dict,
    body_sections: dict[str, str] | None = None,
) -> MathObject:
    """
    Reconstruct a :class:`MathObject` from a Knowledge Inbox Notion page.

    ``body_sections`` — output of :func:`_extract_body_sections` for this page.
    Used to recover ``conclusion``, which is stored only in the page body (under
    the ``## Conclusion`` heading), not as a DB property.
    """
    props = page["properties"]

    title = _get_page_title(props)
    # Strip legacy type prefix e.g. "[Theorem] " added by early pipeline versions.
    title = re.sub(r"^\[[^\]]+\]\s*", "", title).strip()

    # Conclusion lives in the body, not in a property.
    conclusion = ""
    if body_sections:
        conclusion = body_sections.get("Conclusion", "")
    if not conclusion:
        # Fallback: use Interpretation as a semantic surrogate.
        conclusion = _get_text_prop(props, "Interpretation") or "(see Statement LaTeX)"

    return MathObject(
        type=_get_select(props, "Type") or "Definition",
        title=title or "(untitled)",
        statement_latex=_get_text_prop(props, "Statement LaTeX") or "Not recorded.",
        assumptions=_get_text_prop(props, "Assumptions") or "None explicitly stated.",
        variables="",
        conclusion=conclusion,
        source_pages=[],
        source_quotes=None,
        confidence=_get_number(props, "AI Confidence") or 0.8,
        suggested_hub=_get_text_prop(props, "Suggested Hub") or "Uncategorized",
        interpretation=_get_text_prop(props, "Interpretation"),
        proof_idea=_get_text_prop(props, "Proof Idea"),
        source_anchors=_get_text_prop(props, "Source Anchors"),
        named_tools=_get_multi(props, "Named Tools"),
        setting=_get_multi(props, "Setting"),
        result_category=_get_select(props, "Result Category"),
        canonical_keywords=_get_multi(props, "Keywords"),
        prereq_keywords=_get_multi(props, "Prereq Keywords"),
        downstream_keywords=_get_multi(props, "Downstream Keywords"),
        aliases=_get_text_prop(props, "Aliases"),
    )


# ── Edge section cleanup ──────────────────────────────────────────────────────

def _delete_block(notion: NotionClientWrapper, block_id: str) -> None:
    """Delete a single Notion block via the underlying SDK client."""
    notion._call(notion._client.blocks.delete, block_id=block_id)


def clear_edge_section(
    notion: NotionClientWrapper,
    ki_page_id: str,
    dry_run: bool = False,
) -> int:
    """
    Find and delete the ``## Proposed Cross-Paper Edges`` section (and the
    divider immediately before it) from a KI page body.

    Returns the number of blocks removed.
    """
    blocks = notion.get_block_children(ki_page_id)
    start_idx: int | None = None

    for i, block in enumerate(blocks):
        btype = block.get("type", "")

        if btype == "heading_2":
            rt = block.get("heading_2", {}).get("rich_text", [])
            heading_text = "".join(seg.get("plain_text", "") for seg in rt).strip()
            if "Proposed Cross-Paper Edges" in heading_text:
                # Walk back one block: include a preceding divider if present.
                if i > 0 and blocks[i - 1].get("type") == "divider":
                    start_idx = i - 1
                else:
                    start_idx = i
                break

    if start_idx is None:
        return 0

    to_delete = blocks[start_idx:]
    log.info(
        "KI %s: edge section starts at block index %d — %d block(s) to remove.",
        ki_page_id, start_idx, len(to_delete),
    )

    if dry_run:
        return len(to_delete)

    deleted = 0
    for block in to_delete:
        bid = block["id"]
        try:
            _delete_block(notion, bid)
            deleted += 1
        except Exception as exc:
            log.warning("Could not delete block %s: %s", bid, exc)

    return deleted


# ── Main logic ────────────────────────────────────────────────────────────────

def reprocess_paper(
    paper_page_id: str,
    run_concepts: bool,
    run_edges: bool,
    keep_existing: bool = False,
    dry_run: bool = False,
) -> None:
    notion = NotionClientWrapper()

    # ── Fetch paper page ──────────────────────────────────────────────────────
    log.info("Fetching paper page %s ...", paper_page_id)
    paper_page = notion.get_page(paper_page_id)
    props = paper_page["properties"]
    paper_title = _get_page_title(props) or paper_page_id
    log.info("Paper: %s", paper_title)

    # ── Initialise IngestionEngine ────────────────────────────────────────────
    vector_index: VectorIndexEngine | None = None
    if os.environ.get("VECTOR_INDEX_ENABLED"):
        try:
            vector_index = VectorIndexEngine()
            log.info("VectorIndexEngine: available=%s", vector_index.available)
        except Exception:
            log.warning("Could not initialise VectorIndexEngine")
            return

    engine = IngestionEngine(vector_index=vector_index)
    run_id = uuid.uuid4().hex[:8]
    log.info("Run ID: %s", run_id)

    # ── Build hubs + Second Brain index ───────────────────────────────────────
    log.info("Building hubs and Second Brain index ...")
    hubs = engine._fetch_allowed_hubs()
    sb_index = engine._build_second_brain_index()
    log.info("Loaded %d hub(s), %d SB concept(s).", len(hubs), len(sb_index))

    # Concepts carried forward from the extraction stage into the linking stage.
    fresh_ki_pages: list[tuple[MathObject, str]] = []

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Re-run concept extraction
    # ═════════════════════════════════════════════════════════════════════════
    if run_concepts:
        log.info("═" * 60)
        log.info("CONCEPTS  Re-running Stage 1 extraction for: %s", paper_title)

        # Locate attachment key (written to paper page by the main pipeline).
        attachment_key = _get_text_prop(props, "Zotero Attachment Key").strip()
        if not attachment_key:
            log.error(
                "CONCEPTS  'Zotero Attachment Key' is empty on this paper page.\n"
                "          Run the main pipeline at least once so the attachment\n"
                "          key is resolved and the Markdown cache is populated."
            )
            sys.exit(1)

        koofr_markdown_dir = os.environ.get("KOOFR_MARKDOWN_PATH", "/zotero_markdown")
        md_remote = f"{koofr_markdown_dir}/{attachment_key}.md"
        log.info("CONCEPTS  Markdown cache path: %s", md_remote)

        # Download markdown (skip in dry-run).
        markdown_text: str = ""
        if not dry_run:
            try:
                raw_bytes = engine._koofr_download_bytes(md_remote)
                markdown_text = raw_bytes.decode("utf-8")
                token_count = _count_tokens(markdown_text)
                log.info(
                    "CONCEPTS  Markdown downloaded: %d chars, ~%d tokens.",
                    len(markdown_text), token_count,
                )
            except Exception as exc:
                log.error(
                    "CONCEPTS  Cannot download Markdown cache from Koofr: %s\n"
                    "          Make sure the paper has been run through the main\n"
                    "          pipeline so the Markdown cache exists on Koofr.",
                    exc,
                )
                sys.exit(1)
        else:
            token_count = 0

        # Archive existing KI pages (unless --keep-existing).
        existing_ki = notion.query_database(
            os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"],
            filter={"property": "Source Paper", "relation": {"contains": paper_page_id}},
        )
        if not keep_existing:
            log.info(
                "CONCEPTS  Archiving %d existing KI page(s) ...", len(existing_ki)
            )
            for ki in existing_ki:
                ki_id = ki["id"]
                ki_title = _get_page_title(ki["properties"])
                if dry_run:
                    log.info("[DRY RUN]  Would archive KI page: %s  (%s)", ki_title, ki_id)
                else:
                    try:
                        notion.update_page(ki_id, properties={}, archived=True)
                        log.info("CONCEPTS  Archived: %s", ki_title)
                    except Exception as exc:
                        log.warning("CONCEPTS  Could not archive %s: %s", ki_id, exc)
        else:
            log.info(
                "CONCEPTS  --keep-existing: preserving %d existing KI page(s).",
                len(existing_ki),
            )

        if dry_run:
            log.info(
                "[DRY RUN]  Would run extraction on %d tokens and create KI pages.",
                token_count,
            )
        else:
            # Run extraction.
            log.info(
                "CONCEPTS  Running LLM extraction (%d tokens) ...", token_count
            )
            extraction = engine._run_extraction(
                markdown_text, token_count, hubs, run_id
            )
            concepts = extraction.extracted_concepts
            log.info("CONCEPTS  %d concept(s) extracted.", len(concepts))

            if not concepts:
                log.warning("CONCEPTS  No concepts returned — nothing to create.")
            else:
                created = 0
                rejected = 0
                for concept in concepts:
                    verdict = check_completeness(concept)
                    if verdict.status == "reject":
                        log.info(
                            "CONCEPTS  '%s' rejected by completeness gate: %s",
                            concept.title, verdict.reasons,
                        )
                        rejected += 1
                        continue
                    flag_reasons = verdict.reasons if verdict.status == "flag" else None
                    try:
                        ki_page_id = engine._create_knowledge_item(
                            paper_page_id, concept, hubs, flag_reasons=flag_reasons
                        )
                        fresh_ki_pages.append((concept, ki_page_id))
                        created += 1
                        log.info("CONCEPTS  Created KI page: %s", concept.title)
                        if vector_index and vector_index.available:
                            try:
                                vector_index.index_concept(
                                    concept, ki_page_id, verified=False
                                )
                            except Exception:
                                log.warning(
                                    "CONCEPTS  VectorIndex failed to index '%s'.",
                                    concept.title,
                                )
                    except Exception:
                        log.exception(
                            "CONCEPTS  Failed to create KI page for '%s'.",
                            concept.title,
                        )

                log.info(
                    "CONCEPTS  Done: %d created, %d rejected.", created, rejected
                )

                # Inject new KI pages into sb_index so Stage 2/3 can cross-link.
                engine._inject_ki_pages_into_index(fresh_ki_pages, sb_index)

                # Patch paper page body with extracted concepts callout.
                engine._patch_paper_page(
                    paper_page_id, [kid for _, kid in fresh_ki_pages]
                )

                # Update paper Tracker metadata.
                engine._patch_notion_page(paper_page_id, extraction, run_id)
                notion.update_page(
                    page_id=paper_page_id,
                    properties={
                        "Status": notion.status_prop("s2-extracted"),
                        "Extraction Count": {"number": len(fresh_ki_pages)},
                        "Extraction Tokens": {"number": token_count},
                    },
                )
                log.info("CONCEPTS  Paper status → s2-extracted.")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 + 3 — Re-run edge linking
    # ═════════════════════════════════════════════════════════════════════════
    if run_edges:
        log.info("═" * 60)
        log.info("EDGES  Re-running Stage 2 + 3 for: %s", paper_title)

        # Use concepts from the extraction stage, or load existing KI pages.
        ki_pages_for_linking: list[tuple[MathObject, str]]

        if fresh_ki_pages:
            # Came straight from a --concepts run in this session.
            log.info(
                "EDGES  Using %d freshly extracted concept(s).",
                len(fresh_ki_pages),
            )
            ki_pages_for_linking = fresh_ki_pages
            # sb_index injection already done in the concepts stage.
        else:
            # Load existing KI pages from Notion.
            existing_ki = notion.query_database(
                os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"],
                filter={
                    "property": "Source Paper",
                    "relation": {"contains": paper_page_id},
                },
            )
            log.info("EDGES  Found %d existing KI page(s).", len(existing_ki))

            if not existing_ki:
                log.warning(
                    "EDGES  No KI pages found for this paper — run --concepts first."
                )
                return

            log.info("EDGES  Reconstructing MathObjects from KI properties ...")
            ki_pages_for_linking = []
            for ki_page in existing_ki:
                ki_id = ki_page["id"]
                # Fetch page body to recover the Conclusion section.
                try:
                    body_blocks = notion.get_block_children(ki_id)
                    body_sections = _extract_body_sections(body_blocks)
                except Exception:
                    log.warning(
                        "EDGES  Could not fetch body for KI page %s — using empty sections.",
                        ki_id,
                    )
                    body_sections = {}

                concept = ki_page_to_math_object(ki_page, body_sections)
                ki_pages_for_linking.append((concept, ki_id))
                log.info("EDGES  Loaded concept: %s", concept.title)

            # Inject existing KI concepts into sb_index so they can cross-link.
            engine._inject_ki_pages_into_index(ki_pages_for_linking, sb_index)

        if dry_run:
            log.info(
                "[DRY RUN]  Would run Stage 2 + 3 for %d concept(s).",
                len(ki_pages_for_linking),
            )
            return

        # -- Clear old edge data from each KI page ----------------------------
        log.info("EDGES  Clearing existing edge sections from KI pages ...")
        for _, ki_id in ki_pages_for_linking:
            n_deleted = clear_edge_section(notion, ki_id, dry_run=False)
            if n_deleted:
                log.info("EDGES  Removed %d edge block(s) from %s.", n_deleted, ki_id)
            # Reset edge-related properties.
            try:
                notion.update_page(
                    page_id=ki_id,
                    properties={
                        "Edge Suggestions": {
                            "rich_text": notion.rich_text("")
                        },
                        "Graph Link Status": engine._ki_prop(
                            "Graph Link Status", "unlinked"
                        ),
                    },
                )
            except Exception as exc:
                log.warning(
                    "EDGES  Could not reset edge props for KI %s: %s", ki_id, exc
                )

        # -- Stage 2: retrieve candidates -------------------------------------
        log.info("EDGES  Stage 2 — retrieving candidates ...")
        all_ki_ids = {kid for _, kid in ki_pages_for_linking}
        concept_candidates: list[tuple[MathObject, str, list[dict]]] = []

        for concept, ki_id in ki_pages_for_linking:
            same_paper_ids = all_ki_ids - {ki_id}
            candidates = engine._retrieve_candidates_for_concept(
                concept,
                sb_index,
                current_page_id=ki_id,
                same_paper_ids=same_paper_ids,
            )
            engine._update_knowledge_item_candidates(ki_id, candidates)
            concept_candidates.append((concept, ki_id, candidates))
            log.info(
                "EDGES  '%s' — %d candidate(s).", concept.title, len(candidates)
            )

        # -- Stage 3: LLM linking ---------------------------------------------
        log.info("EDGES  Stage 3 — LLM linking ...")
        linked = 0
        failed = 0
        for concept, ki_id, candidates in concept_candidates:
            try:
                link_result = engine._run_stage_link(concept, candidates, run_id)
                engine._update_knowledge_item_graph_data(ki_id, link_result)
                linked += 1
                log.info("EDGES  '%s' — linking done.", concept.title)
            except Exception:
                log.exception(
                    "EDGES  Link stage failed for concept '%s'.", concept.title
                )
                failed += 1

        log.info("EDGES  Done: %d linked, %d failed.", linked, failed)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Reprocess complete.  Paper: %s", paper_title)
    if dry_run:
        log.info("(DRY RUN — no changes were written.)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-process a Notion paper page: recreate concepts (Stage 1) "
            "and/or refresh edges (Stages 2 + 3)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "page_id",
        help="Notion Paper Tracker page ID (32-char hex, with or without dashes).",
    )
    parser.add_argument(
        "--concepts",
        action="store_true",
        help=(
            "Re-run Stage 1 LLM extraction from the Koofr Markdown cache. "
            "Archives existing KI pages for this paper unless --keep-existing is set."
        ),
    )
    parser.add_argument(
        "--edges",
        action="store_true",
        help=(
            "Re-run Stage 2 (candidate retrieval) + Stage 3 (LLM linking) "
            "for all KI concepts linked to this paper."
        ),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "When --concepts is set, preserve existing KI pages instead of "
            "archiving them before the new extraction."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any API calls or writes.",
    )

    args = parser.parse_args()

    # Default: if neither --concepts nor --edges is specified, run --edges.
    run_concepts = args.concepts
    run_edges = args.edges
    if not run_concepts and not run_edges:
        log.info("No operation flags given — defaulting to --edges.")
        run_edges = True

    reprocess_paper(
        paper_page_id=args.page_id,
        run_concepts=run_concepts,
        run_edges=run_edges,
        keep_existing=args.keep_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

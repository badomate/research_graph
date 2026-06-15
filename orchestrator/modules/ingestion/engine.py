"""
modules/ingestion/engine.py — IngestionEngine orchestrator.

Thin coordinator that owns the run loop and delegates to focused services:
PdfFetcherService, ExtractionService, CandidateRetriever, ConceptLinker,
KnowledgeInboxWriter.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

import anthropic
import instructor

from ..extraction_schema import (
    ExtractionResult,
    MathObject,
    check_completeness,
)
from ..exceptions import PipelineError
from ..job_ledger import JobLedger
from ..logging_utils import structured_log
from ..config import Config, get_config
from ..notion_client_wrapper import NotionClientWrapper
from ..tag_linter import TagLinter, lint_report_to_text
from ..vector_index import VectorIndexEngine
from .extractor import ExtractionService, _count_tokens
from .ki_writer import KnowledgeInboxWriter
from .linker import ConceptLinker
from .pdf_fetcher import PdfFetcherService, _ZOTERO_PARENT_RE
from .retriever import CandidateRetriever

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
EXTRACTION_VERSION: str = os.environ.get("EXTRACTION_VERSION", "v3")


class IngestionEngine:
    """Module 1: Core Ingestion Engine (Notion -> WebDAV -> Marker -> Claude -> Notion)."""

    def __init__(
        self,
        vector_index: Optional[VectorIndexEngine] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        package = sys.modules.get(__package__)
        notion_cls = getattr(package, "NotionClientWrapper", NotionClientWrapper)
        anthropic_module = getattr(package, "anthropic", anthropic)
        instructor_module = getattr(package, "instructor", instructor)
        self.notion = notion_cls(self.config)
        _anthropic = anthropic_module.Anthropic(api_key=self.config.anthropic_api_key)
        self.claude_client = instructor_module.from_anthropic(_anthropic)
        self.anthropic_raw = _anthropic
        self._ledger = JobLedger()
        self._tag_linter = TagLinter()
        self._vector_index: VectorIndexEngine | None = vector_index or None

        self.paper_tracker_db = self.config.notion_paper_tracker_db_id
        self.knowledge_inbox_db = self.config.notion_knowledge_inbox_db_id
        self.second_brain_db = self.config.notion_second_brain_db_id

        self._pdf_fetcher = PdfFetcherService(self.notion, self._ledger, self.config)
        self._extractor = ExtractionService(self.claude_client, self.anthropic_raw, self.config)
        self._retriever = CandidateRetriever(self.notion, self._vector_index, self.config)
        self._linker = ConceptLinker(
            self.claude_client, self.config, anthropic_raw=self.anthropic_raw
        )
        self._ki_writer = KnowledgeInboxWriter(self.notion, self.knowledge_inbox_db)

    @staticmethod
    def _build_webdav_client():
        """Compatibility shim for older tests; PDF fetching owns WebDAV now."""
        return PdfFetcherService._build_webdav_client()

    def hydrate_candidates(self, candidate_ids: list[str]):
        """Compatibility shim for the old monolithic ingestion module API."""
        return self._retriever.hydrate_candidates(candidate_ids)

    # -- Entry point -----------------------------------------------------------

    def run(self) -> None:
        logger.info("Ingestion: polling for s1-skim and s2-reextract papers ...")
        pages = self.notion.query_database(
            self.paper_tracker_db,
            filter={"property": "Status", "status": {"equals": "s1-skim"}},
        )
        pages_to_reextract = self.notion.query_database(
            self.paper_tracker_db,
            filter={"property": "Status", "status": {"equals": "s2-reextract"}},
        )
        logger.info(
            "Ingestion: found %d paper(s) to extract, %d paper(s) to re-extract.",
            len(pages), len(pages_to_reextract),
        )

        if not pages and not pages_to_reextract:
            return

        hubs: dict[str, str] = self._fetch_allowed_hubs()
        sb_index: list[dict] = self._build_second_brain_index()
        logger.info(
            "Ingestion: loaded %d hub(s), %d Second Brain concept(s).",
            len(hubs), len(sb_index),
        )

        for page in pages:
            try:
                self._process_paper(page, hubs, sb_index)
            except Exception as exc:
                logger.exception(
                    "Failed to process page %s; error_type=%s",
                    page["id"], type(exc).__name__,
                )

        for page in pages_to_reextract:
            try:
                self._reextract_missed_concepts(page, hubs, sb_index)
            except Exception as exc:
                logger.exception(
                    "Failed re-extraction for page %s; error_type=%s",
                    page["id"], type(exc).__name__,
                )

    # -- Hub / index helpers ---------------------------------------------------

    def _fetch_allowed_hubs(self) -> dict[str, str]:
        logger.debug("Ingestion: fetching Hub pages from Second Brain ...")
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={"property": "Note Level", "select": {"equals": "Hub"}},
        )
        hubs: dict[str, str] = {}
        for page in pages:
            name = self._get_page_title(page)
            if name:
                hubs[name] = page["id"]
        return hubs

    def _build_second_brain_index(self) -> list[dict]:
        concept_level = os.environ.get("SB_CONCEPT_LEVEL", "Concept")
        logger.debug(
            "Ingestion: building Second Brain index (Note Level='%s') ...", concept_level
        )
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={"property": "Note Level", "select": {"equals": concept_level}},
        )
        records: list[dict] = []
        for page in pages:
            title = self._get_page_title(page)
            if not title:
                continue
            props = page.get("properties", {})
            hub = self._get_text_prop(props, "Suggested Hub")
            summary = self._get_text_prop(props, "One Liner")
            tags = self._get_multi_select_prop(props, "Tags")
            keywords = self._get_multi_select_prop(props, "Keywords")

            def _tokenise(s: str) -> set:
                return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

            bag: set = set()
            bag |= _tokenise(title)
            for kw in keywords:
                bag |= _tokenise(kw)
            for tag in tags:
                bag |= _tokenise(tag)

            records.append({
                "id": page["id"],
                "title": title,
                "hub": hub,
                "summary": summary,
                "tags": tags,
                "keywords_bag": bag,
            })
        logger.debug("Ingestion: Second Brain index built — %d concept(s).", len(records))
        return records

    # -- Per-paper pipeline ----------------------------------------------------

    def _process_paper(
        self,
        page: dict,
        hubs: dict[str, str],
        sb_index: list[dict],
    ) -> None:
        page_id = page["id"]
        props = page["properties"]

        # REQ-1: Set s1-processing FIRST — race condition guard.
        self.notion.update_page(
            page_id=page_id,
            properties={"Status": self.notion.status_prop("s1-processing")},
        )

        run_id = uuid.uuid4().hex[:8]
        job_id: int | None = None
        cleaned_tokens: int = 0

        try:
            # Preflight gate 1: Parse Zotero parent key
            zotero_uri = self._get_text_prop(props, "Zotero URI")
            if not zotero_uri:
                zotero_uri = ""
            parent_match = _ZOTERO_PARENT_RE.search(zotero_uri)
            if not parent_match:
                logger.warning(
                    "[%s] Missing or invalid Zotero URI: '%s' — reverting to s1-skim.",
                    page_id, zotero_uri,
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1-skim"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            "Missing or invalid Zotero URI."
                        )},
                    },
                )
                return
            parent_key = parent_match.group(1)

            # Preflight gate 1b: Resolve attachment key
            resolved = self._pdf_fetcher.resolve_keys_and_update_notion(
                page_id, zotero_uri, parent_key, run_id
            )
            if resolved is None:
                logger.warning(
                    "[%s] Cannot resolve attachment key for parent '%s' "
                    "— setting s1b-waiting-attachment.",
                    run_id, parent_key,
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={"Status": self.notion.status_prop("s1b-waiting-attachment")},
                )
                return

            parent_key, attachment_key = resolved

            # Preflight gate 2: Check Koofr zip exists
            zip_remote = f"{self._pdf_fetcher.koofr_base}/{attachment_key}.zip"
            logger.info("[%s] Checking Koofr zip: %s", run_id, zip_remote)
            if not self._pdf_fetcher.koofr_exists(zip_remote):
                logger.warning("[%s] Zip not found — setting s1b-waiting-attachment.", run_id)
                self.notion.update_page(
                    page_id=page_id,
                    properties={"Status": self.notion.status_prop("s1b-waiting-attachment")},
                )
                return

            # Stage 1 / Step 1: Convert PDF to Markdown
            structured_log(logger, "info", "Stage 1: converting PDF to Markdown", run_id=run_id)
            markdown_text, job_id = self._pdf_fetcher.pdf_to_markdown(
                attachment_key=attachment_key,
                run_id=run_id,
                zip_remote=zip_remote,
                primary_pdf_filename=self._get_text_prop(props, "primary_pdf_filename") or None,
                page_id=page_id,
                props=props,
            )
            if markdown_text is None:
                return

            if job_id is not None:
                self._ledger.update_status(job_id, "marker_done")

            # Strip boilerplate and count tokens (already done in cache-miss path,
            # but on cache-hit the stored markdown is already stripped).
            cleaned_tokens = _count_tokens(markdown_text)

            # Stage 1 / Step 2: Extract via Claude
            structured_log(
                logger, "info", "Stage 1: extracting knowledge via Claude",
                run_id=run_id, tokens=cleaned_tokens,
            )
            extraction = self._extractor.run_extraction(markdown_text, cleaned_tokens, hubs, run_id)

            # Stage 1 / Step 3: Patch Paper Tracker metadata
            logger.info("[%s] Stage 1: patching Notion paper row ...", run_id)
            self._patch_notion_page(page_id, extraction, run_id)

            # Stage 1 / Step 4: Create Knowledge Inbox entries
            concepts = extraction.extracted_concepts

            # REQ-4: Zero concept guard
            if len(concepts) == 0:
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("blocked-extraction"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            "GPT returned 0 concepts. Check markdown quality or "
                            "add Re-extract Hints and set status back to s1-skim."
                        )},
                        "Extraction Count": {"number": 0},
                    },
                )
                logger.warning(
                    "Ingestion: zero concepts extracted for paper %s — set to blocked-extraction.",
                    page_id,
                )
                return

            logger.info(
                "[%s] Stage 1: creating %d Knowledge Inbox page(s) ...",
                run_id, len(concepts),
            )
            ki_pages: list[tuple[MathObject, str]] = []
            rejected_concepts: list[dict] = []
            for concept in concepts:
                verdict = check_completeness(concept)
                if verdict.status == "reject":
                    logger.info(
                        "[%s] Concept '%s' rejected by completeness gate: %s",
                        run_id, concept.title, verdict.reasons,
                    )
                    rejected_concepts.append({
                        "title": concept.title,
                        "type": concept.type,
                        "confidence": concept.confidence,
                        "reasons": verdict.reasons,
                    })
                    continue
                flag_reasons = verdict.reasons if verdict.status == "flag" else None
                try:
                    ki_page_id = self._ki_writer.create_knowledge_item(
                        page_id, concept, hubs, flag_reasons=flag_reasons
                    )
                    ki_pages.append((concept, ki_page_id))
                    if self._vector_index and self._vector_index.available:
                        try:
                            self._vector_index.index_concept(concept, ki_page_id, verified=False)
                        except Exception:
                            logger.warning(
                                "[%s] VectorIndex: failed to index '%s' — continuing.",
                                run_id, concept.title,
                            )
                except Exception:
                    logger.exception(
                        "[%s] Failed to create knowledge item '%s'", run_id, concept.title
                    )

            if rejected_concepts:
                try:
                    rejected_json = json.dumps(rejected_concepts, ensure_ascii=False)
                    self.notion.update_page(
                        page_id=page_id,
                        properties={
                            "Rejected Concepts": {
                                "rich_text": self.notion.rich_text(rejected_json[:2000])
                            }
                        },
                    )
                    logger.info(
                        "[%s] Wrote %d rejected concept(s) to Paper Tracker.",
                        run_id, len(rejected_concepts),
                    )
                except Exception:
                    logger.warning(
                        "[%s] Could not write Rejected Concepts to Paper Tracker.",
                        run_id, exc_info=True,
                    )

            structured_log(logger, "info", "Stage 1 complete: KI pages created", run_id=run_id, ki_pages=len(ki_pages))
            if job_id is not None:
                self._ledger.update_status(job_id, "extract_done")

            if len(ki_pages) == 0:
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("blocked-extraction"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            "All extracted concepts were rejected by the completeness "
                            "gate. Check markdown quality or add Re-extract Hints and "
                            "set status back to s1-skim."
                        )},
                        "Extraction Count": {"number": 0},
                    },
                )
                logger.warning(
                    "Ingestion: all concepts rejected by completeness gate for paper %s.",
                    page_id,
                )
                return

            self._inject_ki_pages_into_index(ki_pages, sb_index)

            # Stage 2: Retrieve candidates
            structured_log(logger, "info", "Stage 2: retrieving candidates", run_id=run_id)
            all_ki_ids = {ki_id for _, ki_id in ki_pages}
            concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
            for concept, ki_page_id in ki_pages:
                same_paper_ids = all_ki_ids - {ki_page_id}
                candidates = self._retriever.retrieve_candidates_for_concept(
                    concept, sb_index,
                    current_page_id=ki_page_id,
                    same_paper_ids=same_paper_ids,
                )
                self._retriever.update_knowledge_item_candidates(ki_page_id, candidates)
                concept_candidates.append((concept, ki_page_id, candidates))
                logger.info(
                    "[%s] '%s': %d candidate(s) retrieved.",
                    run_id, concept.title, len(candidates),
                )
            if job_id is not None:
                self._ledger.update_status(job_id, "retrieve_done")

            # Stage 3: LLM linking
            structured_log(logger, "info", "Stage 3: LLM linking", run_id=run_id)
            self._run_link_stage(concept_candidates, run_id)
            if job_id is not None:
                self._ledger.update_status(job_id, "link_done")

            # Finalise
            self._patch_notion_paper_post_linking(page_id, run_id)
            if job_id is not None:
                self._ledger.update_status(job_id, "notion_done")
                self._ledger.finish_job(job_id)

            self._ki_writer.patch_paper_page(page_id, [ki_id for _, ki_id in ki_pages])
            self.notion.update_page(
                page_id=page_id,
                properties={
                    "Status": self.notion.status_prop("s2-extracted"),
                    "Extraction Count": {"number": len(ki_pages)},
                    "Extraction Tokens": {"number": cleaned_tokens},
                },
            )
            structured_log(logger, "info", "Paper processing complete", run_id=run_id, ki_pages=len(ki_pages))

        except Exception as exc:
            logger.exception(
                "[%s] Pipeline failed; error_type=%s: %s",
                run_id, type(exc).__name__, exc,
            )
            if job_id is not None:
                self._ledger.update_status(job_id, "failed", error=str(exc))
            try:
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1-skim"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            traceback.format_exc()[:2000]
                        )},
                        "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
                    },
                )
            except Exception:
                logger.warning(
                    "[%s] Could not write error context to Notion Paper Tracker.", run_id
                )
            raise

    def _run_link_stage(
        self,
        concept_candidates: list[tuple[MathObject, str, list[dict]]],
        run_id: str,
    ) -> None:
        """Execute Stage 3 either as one batch (opt-in) or per-concept (default).

        On any hard failure of the batch path, fall through to the synchronous
        per-concept path so edges are still created.
        """
        if self.config.link_use_batch_api:
            try:
                results = self._linker.run_stage_link_batch(concept_candidates, run_id)
            except Exception:
                logger.exception(
                    "[%s] Stage 3: batch linking failed — falling back to per-concept.",
                    run_id,
                )
            else:
                for _concept, ki_page_id, _candidates in concept_candidates:
                    link_result = results.get(ki_page_id)
                    if link_result is None:
                        continue
                    try:
                        self._ki_writer.update_knowledge_item_graph_data(ki_page_id, link_result)
                    except Exception:
                        logger.exception(
                            "[%s] Stage 3: failed to write graph data for %s",
                            run_id, ki_page_id,
                        )
                return

        for concept, ki_page_id, candidates in concept_candidates:
            try:
                link_result = self._linker.run_stage_link(concept, candidates, run_id)
                self._ki_writer.update_knowledge_item_graph_data(ki_page_id, link_result)
            except Exception:
                logger.exception(
                    "[%s] Link stage failed for concept '%s'", run_id, concept.title
                )

    def _inject_ki_pages_into_index(
        self,
        ki_pages: list[tuple[MathObject, str]],
        sb_index: list[dict],
    ) -> None:
        def _toks(s: str) -> set:
            return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

        for concept, ki_page_id in ki_pages:
            bag: set = set()
            for kw_list in (concept.canonical_keywords, concept.prereq_keywords, concept.downstream_keywords):
                for kw in kw_list:
                    bag |= _toks(kw)
            bag |= _toks(concept.title)
            sb_index.append({
                "id": ki_page_id,
                "title": concept.title,
                "hub": concept.suggested_hub or "",
                "summary": concept.conclusion or "",
                "tags": concept.setting or [],
                "keywords_bag": bag,
            })
        logger.debug(
            "Injected %d new concept(s) into sb_index (total now: %d).",
            len(ki_pages), len(sb_index),
        )

    # -- Notion page patching --------------------------------------------------

    def _patch_notion_page(
        self, page_id: str, result: ExtractionResult, run_id: str,
        set_thesis_relevance: bool = False,
    ) -> None:
        properties: dict = {
            "AI Status": self.notion.select_prop("Unverified-AI"),
            "One Liner": {"rich_text": self.notion.rich_text(result.one_liner)},
            "Active Themes": self.notion.multi_select_prop(result.active_themes),
            "Extraction Version": {"rich_text": self.notion.rich_text(EXTRACTION_VERSION)},
            "Processed At": {"date": {"start": datetime.now(tz=timezone.utc).isoformat()}},
            "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
            "Last Error": {"rich_text": self.notion.rich_text("")},
        }
        self.notion.update_page(page_id=page_id, properties=properties)

    def _patch_notion_paper_post_linking(self, page_id: str, run_id: str) -> None:
        self.notion.update_page(
            page_id=page_id,
            properties={"Last Run ID": {"rich_text": self.notion.rich_text(run_id)}},
        )

    # -- Re-extraction flow (REQ-5) --------------------------------------------

    def _reextract_missed_concepts(
        self,
        page: dict,
        hubs: dict[str, str],
        sb_index: list[dict],
    ) -> None:
        from .prompts import LATEX_FORMATTING_RULES, REEXTRACT_SYSTEM_PROMPT

        page_id = page["id"]
        props = page["properties"]
        run_id = uuid.uuid4().hex[:8]

        hints = self._get_text_prop(props, "Re-extract Hints").strip()
        if not hints:
            logger.warning(
                "[%s] s2-reextract: 'Re-extract Hints' is empty for page %s — "
                "reverting to s2-extracted.",
                run_id, page_id,
            )
            self.notion.update_page(
                page_id=page_id,
                properties={"Status": self.notion.status_prop("s2-extracted")},
            )
            return

        existing_ki = self.notion.query_database(
            self.knowledge_inbox_db,
            filter={"property": "Source Paper", "relation": {"contains": page_id}},
        )
        existing_titles = [self._get_page_title(p) for p in existing_ki if self._get_page_title(p)]
        existing_titles_str = "\n".join(f"- {t}" for t in existing_titles) or "(none)"

        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = (
            REEXTRACT_SYSTEM_PROMPT
            .replace("{hints}", hints)
            .replace("{existing_titles}", existing_titles_str)
            .replace("{latex_formatting_rules}", LATEX_FORMATTING_RULES)
        )
        system_prompt += f"\n\nALLOWED_HUBS:\n[{hub_names_str}]"
        markdown_context = hints

        logger.info(
            "[%s] Re-extraction: %d hint(s), %d existing concept(s).",
            run_id, len(hints.split("\n")), len(existing_titles),
        )

        try:
            reextraction = self._extractor.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract ONLY the missing concepts described above.\n\n"
                        f"PAPER CONTEXT:\n{markdown_context[:20_000]}"
                    ),
                }],
                response_model=ExtractionResult,
            )
        except Exception:
            logger.exception("[%s] Re-extraction Claude call failed.", run_id)
            return

        new_concepts = [
            c for c in reextraction.extracted_concepts
            if self._extractor.normalize_concept_title(c.title)
            not in {self._extractor.normalize_concept_title(t) for t in existing_titles}
        ]

        logger.info("[%s] Re-extraction: %d new concept(s) after dedup.", run_id, len(new_concepts))

        if not new_concepts:
            logger.info("[%s] Re-extraction: nothing new — advancing to s2-extracted.", run_id)
            self.notion.update_page(
                page_id=page_id,
                properties={"Status": self.notion.status_prop("s2-extracted")},
            )
            return

        ki_pages: list[tuple[MathObject, str]] = []
        for concept in new_concepts:
            verdict = check_completeness(concept)
            if verdict.status == "reject":
                logger.info(
                    "[%s] Re-extraction: concept '%s' rejected: %s",
                    run_id, concept.title, verdict.reasons,
                )
                continue
            flag_reasons = verdict.reasons if verdict.status == "flag" else None
            try:
                ki_page_id = self._ki_writer.create_knowledge_item(
                    page_id, concept, hubs, flag_reasons=flag_reasons
                )
                ki_pages.append((concept, ki_page_id))
                if self._vector_index and self._vector_index.available:
                    try:
                        self._vector_index.index_concept(concept, ki_page_id, verified=False)
                    except Exception:
                        logger.warning("[%s] VectorIndex: failed to index '%s'.", run_id, concept.title)
            except Exception:
                logger.exception(
                    "[%s] Re-extraction: failed to create KI item '%s'", run_id, concept.title
                )

        self._inject_ki_pages_into_index(ki_pages, sb_index)

        concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
        for concept, ki_page_id in ki_pages:
            candidates = self._retriever.retrieve_candidates_for_concept(
                concept, sb_index, current_page_id=ki_page_id
            )
            self._retriever.update_knowledge_item_candidates(ki_page_id, candidates)
            concept_candidates.append((concept, ki_page_id, candidates))

        self._run_link_stage(concept_candidates, run_id)

        self.notion.update_page(
            page_id=page_id,
            properties={"Status": self.notion.status_prop("s2-extracted")},
        )
        logger.info(
            "[%s] Re-extraction complete: %d new concept(s) created.", run_id, len(ki_pages)
        )

    # -- Static property helpers -----------------------------------------------

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
    def _get_text_prop(props: dict, key: str) -> str:
        prop = props.get(key, {})
        if prop.get("type") == "url":
            return prop.get("url") or ""
        try:
            return prop["rich_text"][0]["plain_text"]
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _get_title_prop(props: dict) -> str:
        for value in props.values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return "unknown"
        return "unknown"

    @staticmethod
    def _get_multi_select_prop(props: dict, key: str) -> list[str]:
        try:
            return [opt["name"] for opt in props[key]["multi_select"]]
        except (KeyError, TypeError):
            return []

"""
modules/ingestion/engine.py — IngestionEngine orchestrator (SQLite Store backend).

Polls the papers table for s1-skim / s2-reextract rows and runs the 3-stage
pipeline (extract → retrieve → link), writing concepts and proposed edges to the
Store. Replaces the former Notion-backed implementation.
"""
from __future__ import annotations

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

from ..extraction_schema import ExtractionResult, MathObject, check_completeness
from ..job_ledger import JobLedger
from ..logging_utils import structured_log
from ..config import Config, get_config
from ..store import ConceptState, PaperStatus, Store, VerificationStatus, make_engine
from ..tag_linter import TagLinter
from ..vector_index import VectorIndexEngine
from .concept_writer import ConceptWriter
from .extractor import ExtractionService, _count_tokens
from .linker import ConceptLinker
from .pdf_fetcher import PdfFetcherService, _ZOTERO_PARENT_RE
from .retriever import CandidateRetriever

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
EXTRACTION_VERSION: str = os.environ.get("EXTRACTION_VERSION", "v3")


def _now():
    return datetime.now(tz=timezone.utc)


def _tokenise(s: str) -> set:
    return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s or "") if t.strip()}


class IngestionEngine:
    """Module 1: Core Ingestion Engine (Store -> PDF -> Marker -> Claude -> Store)."""

    def __init__(
        self,
        vector_index: Optional[VectorIndexEngine] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        package = sys.modules.get(__package__)
        anthropic_module = getattr(package, "anthropic", anthropic)
        instructor_module = getattr(package, "instructor", instructor)
        _anthropic = anthropic_module.Anthropic(api_key=self.config.anthropic_api_key)
        self.claude_client = instructor_module.from_anthropic(_anthropic)
        self.anthropic_raw = _anthropic

        self.store = Store(make_engine(self.config.database_url))
        self.store.create_all()
        self._ledger = JobLedger()
        self._tag_linter = TagLinter()
        self._vector_index: VectorIndexEngine | None = vector_index or None

        self._pdf_fetcher = PdfFetcherService(self.store, self._ledger, self.config)
        self._extractor = ExtractionService(self.claude_client, self.anthropic_raw, self.config)
        self._retriever = CandidateRetriever(self.store, self._vector_index, self.config)
        self._linker = ConceptLinker(
            self.claude_client, self.config, anthropic_raw=self.anthropic_raw
        )
        self._concept_writer = ConceptWriter(self.store)

    def hydrate_candidates(self, candidate_ids: list[str]):
        return self._retriever.hydrate_candidates(candidate_ids)

    # -- Entry point -----------------------------------------------------------

    def run(self) -> None:
        logger.info("Ingestion: polling for s1-skim and s2-reextract papers ...")
        pages = self.store.get_papers_by_status(PaperStatus.S1_SKIM.value)
        pages_to_reextract = self.store.get_papers_by_status(PaperStatus.S2_REEXTRACT.value)
        logger.info(
            "Ingestion: found %d paper(s) to extract, %d to re-extract.",
            len(pages), len(pages_to_reextract),
        )
        if not pages and not pages_to_reextract:
            return

        hubs = self.store.hubs()
        sb_index = self._build_second_brain_index()
        logger.info("Ingestion: %d hub(s), %d Second Brain concept(s).", len(hubs), len(sb_index))

        for paper in pages:
            try:
                self._process_paper(paper, hubs, sb_index)
            except Exception as exc:
                logger.exception(
                    "Failed to process paper %s; error_type=%s", paper.id, type(exc).__name__
                )
        for paper in pages_to_reextract:
            try:
                self._reextract_missed_concepts(paper, hubs, sb_index)
            except Exception as exc:
                logger.exception(
                    "Failed re-extraction for paper %s; error_type=%s", paper.id, type(exc).__name__
                )

    # -- Index helpers ---------------------------------------------------------

    def _build_second_brain_index(self) -> list[dict]:
        records: list[dict] = []
        for c in self.store.second_brain_index():
            bag: set = set()
            bag |= _tokenise(c.effective_title)
            for kw in (c.canonical_keywords or []):
                bag |= _tokenise(kw)
            for tag in (c.setting or []):
                bag |= _tokenise(tag)
            records.append({
                "id": c.id,
                "title": c.effective_title,
                "hub": c.suggested_hub or "",
                "summary": c.conclusion or "",
                "tags": list(c.setting or []),
                "keywords_bag": bag,
            })
        return records

    # -- Per-paper pipeline ----------------------------------------------------

    def _process_paper(self, paper, hubs: dict[str, str], sb_index: list[dict]) -> None:
        paper_id = paper.id
        # REQ-1: claim the paper first (race guard).
        self.store.set_paper_status(paper_id, PaperStatus.S1_PROCESSING.value)

        run_id = uuid.uuid4().hex[:8]
        job_id: int | None = None
        cleaned_tokens = 0

        try:
            markdown_text = self._acquire_markdown(paper, run_id)
            if markdown_text is None:
                return
            # _acquire_markdown returns (text, job_id) via attribute stash
            job_id = self._last_job_id

            cleaned_tokens = _count_tokens(markdown_text)
            structured_log(logger, "info", "Stage 1: extracting via Claude",
                           run_id=run_id, tokens=cleaned_tokens)
            extraction = self._extractor.run_extraction(markdown_text, cleaned_tokens, hubs, run_id)
            if job_id is not None:
                self._ledger.update_status(job_id, "extract_done")

            self._patch_paper_meta(paper_id, extraction, run_id)

            concepts = extraction.extracted_concepts
            if len(concepts) == 0:
                self.store.update_paper(
                    paper_id,
                    status=PaperStatus.BLOCKED_EXTRACTION.value,
                    extraction_error=("Claude returned 0 concepts. Add Re-extract Hints "
                                      "and set status back to s1-skim."),
                    extraction_count=0,
                )
                logger.warning("Ingestion: zero concepts for %s — blocked-extraction.", paper_id)
                return

            ki_pages = self._create_concepts(paper_id, concepts, run_id)
            if job_id is not None:
                self._ledger.update_status(job_id, "retrieve_done")  # set after retrieval below

            if not ki_pages:
                self.store.update_paper(
                    paper_id,
                    status=PaperStatus.BLOCKED_EXTRACTION.value,
                    extraction_error="All concepts rejected by the completeness gate.",
                    extraction_count=0,
                )
                logger.warning("Ingestion: all concepts rejected for %s.", paper_id)
                return

            self._inject_concepts_into_index(ki_pages, sb_index)

            # Stage 2: retrieve
            structured_log(logger, "info", "Stage 2: retrieving candidates", run_id=run_id)
            all_ids = {cid for _, cid in ki_pages}
            concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
            for concept, cid in ki_pages:
                candidates = self._retriever.retrieve_candidates_for_concept(
                    concept, sb_index, current_page_id=cid, same_paper_ids=all_ids - {cid},
                )
                concept_candidates.append((concept, cid, candidates))
            if job_id is not None:
                self._ledger.update_status(job_id, "retrieve_done")

            # Stage 3: link
            structured_log(logger, "info", "Stage 3: LLM linking", run_id=run_id)
            self._run_link_stage(concept_candidates, run_id)
            if job_id is not None:
                self._ledger.update_status(job_id, "link_done")
                self._ledger.update_status(job_id, "notion_done")
                self._ledger.finish_job(job_id)

            self.store.update_paper(
                paper_id,
                status=PaperStatus.S2_EXTRACTED.value,
                extraction_count=len(ki_pages),
                extraction_tokens=cleaned_tokens,
            )
            structured_log(logger, "info", "Paper complete", run_id=run_id, concepts=len(ki_pages))

        except Exception as exc:
            logger.exception("[%s] Pipeline failed; error_type=%s", run_id, type(exc).__name__)
            if job_id is not None:
                self._ledger.update_status(job_id, "failed", error=str(exc))
            try:
                self.store.update_paper(
                    paper_id,
                    status=PaperStatus.S1_SKIM.value,
                    extraction_error=traceback.format_exc()[:2000],
                    last_run_id=run_id,
                )
            except Exception:
                logger.warning("[%s] Could not write error context to store.", run_id)
            raise

    def _acquire_markdown(self, paper, run_id: str) -> str | None:
        """Resolve a paper's PDF to markdown (upload path or Zotero/Koofr path)."""
        self._last_job_id = None

        # Upload / local-PDF path bypasses Zotero + Koofr entirely.
        if paper.pdf_path:
            structured_log(logger, "info", "Stage 1: converting uploaded PDF", run_id=run_id)
            md, job_id = self._pdf_fetcher.markdown_from_local_pdf(paper.pdf_path, run_id, paper.id)
            self._last_job_id = job_id
            if job_id is not None:
                self._ledger.update_status(job_id, "marker_done")
            return md

        # Zotero/Koofr path.
        zotero_uri = paper.zotero_uri or ""
        parent_match = _ZOTERO_PARENT_RE.search(zotero_uri)
        if not parent_match:
            logger.warning("[%s] Missing/invalid Zotero URI — reverting to s1-skim.", run_id)
            self.store.update_paper(
                paper.id, status=PaperStatus.S1_SKIM.value,
                extraction_error="Missing or invalid Zotero URI (and no uploaded PDF).",
            )
            return None
        parent_key = parent_match.group(1)

        resolved = self._pdf_fetcher.resolve_keys_and_update(paper.id, zotero_uri, parent_key, run_id)
        if resolved is None:
            self.store.set_paper_status(paper.id, PaperStatus.S1B_WAITING_ATTACHMENT.value)
            return None
        parent_key, attachment_key = resolved

        zip_remote = f"{self._pdf_fetcher.koofr_base}/{attachment_key}.zip"
        if not self._pdf_fetcher.koofr_exists(zip_remote):
            logger.warning("[%s] Zip not found — s1b-waiting-attachment.", run_id)
            self.store.set_paper_status(paper.id, PaperStatus.S1B_WAITING_ATTACHMENT.value)
            return None

        structured_log(logger, "info", "Stage 1: converting PDF to Markdown", run_id=run_id)
        md, job_id = self._pdf_fetcher.pdf_to_markdown(
            attachment_key=attachment_key, run_id=run_id, zip_remote=zip_remote,
            primary_pdf_filename=paper.primary_pdf_filename or None, paper_id=paper.id,
        )
        self._last_job_id = job_id
        if md is not None and job_id is not None:
            self._ledger.update_status(job_id, "marker_done")
        return md

    def _create_concepts(self, paper_id, concepts, run_id) -> list[tuple[MathObject, str]]:
        """Apply the completeness gate and write surviving concepts as rows."""
        ki_pages: list[tuple[MathObject, str]] = []
        rejected: list[dict] = []
        for concept in concepts:
            verdict = check_completeness(concept)
            if verdict.status == "reject":
                rejected.append({"title": concept.title, "type": concept.type,
                                 "confidence": concept.confidence, "reasons": verdict.reasons})
                continue
            flag_reasons = verdict.reasons if verdict.status == "flag" else None
            try:
                cid = self._concept_writer.create_concept_row(paper_id, concept, flag_reasons)
                ki_pages.append((concept, cid))
                if self._vector_index and self._vector_index.available:
                    try:
                        self._vector_index.index_concept(concept, cid, verified=False)
                    except Exception:
                        logger.warning("[%s] VectorIndex: failed to index '%s'.", run_id, concept.title)
            except Exception:
                logger.exception("[%s] Failed to create concept '%s'.", run_id, concept.title)
        if rejected:
            self.store.update_paper(paper_id, rejected_concepts=rejected)
        return ki_pages

    def _run_link_stage(self, concept_candidates, run_id: str) -> None:
        """Stage 3: batch (opt-in) or per-concept; write edges via the Store."""
        if self.config.link_use_batch_api:
            try:
                results = self._linker.run_stage_link_batch(concept_candidates, run_id)
            except Exception:
                logger.exception("[%s] Batch linking failed — per-concept fallback.", run_id)
            else:
                for _concept, cid, _cands in concept_candidates:
                    link_result = results.get(cid)
                    if link_result is not None:
                        try:
                            self._concept_writer.write_edges(cid, link_result)
                        except Exception:
                            logger.exception("[%s] Edge write failed for %s.", run_id, cid)
                return
        for concept, cid, candidates in concept_candidates:
            try:
                link_result = self._linker.run_stage_link(concept, candidates, run_id)
                self._concept_writer.write_edges(cid, link_result)
            except Exception:
                logger.exception("[%s] Link stage failed for '%s'.", run_id, concept.title)

    def _inject_concepts_into_index(self, ki_pages, sb_index: list[dict]) -> None:
        for concept, cid in ki_pages:
            bag: set = set()
            for kw_list in (concept.canonical_keywords, concept.prereq_keywords, concept.downstream_keywords):
                for kw in kw_list:
                    bag |= _tokenise(kw)
            bag |= _tokenise(concept.title)
            sb_index.append({
                "id": cid, "title": concept.title, "hub": concept.suggested_hub or "",
                "summary": concept.conclusion or "", "tags": concept.setting or [],
                "keywords_bag": bag,
            })

    def _patch_paper_meta(self, paper_id, result: ExtractionResult, run_id: str) -> None:
        self.store.update_paper(
            paper_id,
            ai_status="Unverified-AI",
            one_liner=result.one_liner,
            active_themes=list(result.active_themes or []),
            extraction_version=EXTRACTION_VERSION,
            processed_at=_now(),
            last_run_id=run_id,
            extraction_error="",
        )

    # -- Re-extraction flow ----------------------------------------------------

    def _reextract_missed_concepts(self, paper, hubs, sb_index) -> None:
        from .prompts import LATEX_FORMATTING_RULES, REEXTRACT_SYSTEM_PROMPT

        paper_id = paper.id
        run_id = uuid.uuid4().hex[:8]
        hints = (paper.reextract_hints or "").strip()
        if not hints:
            logger.warning("[%s] s2-reextract with empty hints — back to s2-extracted.", run_id)
            self.store.set_paper_status(paper_id, PaperStatus.S2_EXTRACTED.value)
            return

        existing = self.store.concepts_for_paper(paper_id)
        existing_titles = [c.effective_title for c in existing if c.effective_title]
        existing_titles_str = "\n".join(f"- {t}" for t in existing_titles) or "(none)"
        hub_names_str = ", ".join(f'"{n}"' for n in hubs) if hubs else '"Uncategorized"'

        system_prompt = (
            REEXTRACT_SYSTEM_PROMPT
            .replace("{hints}", hints)
            .replace("{existing_titles}", existing_titles_str)
            .replace("{latex_formatting_rules}", LATEX_FORMATTING_RULES)
        ) + f"\n\nALLOWED_HUBS:\n[{hub_names_str}]"

        try:
            reextraction = self._extractor.claude_client.messages.create(
                model=CLAUDE_MODEL, max_tokens=8192, system=system_prompt,
                messages=[{"role": "user", "content": (
                    "Extract ONLY the missing concepts described above.\n\n"
                    f"PAPER CONTEXT:\n{hints[:20_000]}")}],
                response_model=ExtractionResult,
            )
        except Exception:
            logger.exception("[%s] Re-extraction Claude call failed.", run_id)
            return

        seen = {self._extractor.normalize_concept_title(t) for t in existing_titles}
        new_concepts = [
            c for c in reextraction.extracted_concepts
            if self._extractor.normalize_concept_title(c.title) not in seen
        ]
        if not new_concepts:
            self.store.set_paper_status(paper_id, PaperStatus.S2_EXTRACTED.value)
            return

        ki_pages = self._create_concepts(paper_id, new_concepts, run_id)
        self._inject_concepts_into_index(ki_pages, sb_index)

        concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
        for concept, cid in ki_pages:
            candidates = self._retriever.retrieve_candidates_for_concept(
                concept, sb_index, current_page_id=cid
            )
            concept_candidates.append((concept, cid, candidates))
        self._run_link_stage(concept_candidates, run_id)

        self.store.set_paper_status(paper_id, PaperStatus.S2_EXTRACTED.value)
        logger.info("[%s] Re-extraction complete: %d new concept(s).", run_id, len(ki_pages))

"""
modules/ingestion.py - Module 1: Core Ingestion Engine
-------------------------------------------------------
Pipeline for each paper in the 'Paper Tracker' Notion DB with Status == 's1-process-math':

  PREFLIGHT GATES
  ---------------
  1. Parse Zotero item key from "Zotero URI" rich_text property.
  2. Check Koofr zip exists ({key}.zip); if missing set status "s1b-waiting-attachment".
  3. Download the zip, extract the largest PDF (or "primary_pdf_filename" if set).
  4. Compute pdf_sha256; store in "PDF SHA256" property.
  5. Idempotency check via JobLedger; if already done set status "s2-extracted" and skip.

  TAG COMPLETENESS GATE
  ---------------------
  6. Read "Tags" multi-select; run TagLinter.
  7. If no valid tags: set status "blocked-tags", store lint report, return.

  EXTRACTION
  ----------
  8. Convert PDF to Markdown via marker-api (tenacity retry).
  9. Extract structured knowledge via GPT-4o (new schema with type, title,
     statement_latex, assumptions, variables, conclusion, source_pages,
     source_quotes, confidence).
  10. Validate with Pydantic; attempt one repair pass on failure.
  11. Run latex_sanity_check; downgrade confidence on failure.
  12. Patch Paper Tracker row to status "s2-extracted".
  13. Create Knowledge Inbox pages with verification_status and hub_suggestions.

Design constraints:
  - Notion text blocks are hard-capped at 1900 chars (safe margin below 2000).
  - Hub suggestions stored as text only - never set Parent Hub relation automatically.
  - JobLedger tracks milestones for idempotency and restart safety.
  - All Koofr / Marker / OpenAI calls wrapped with tenacity exponential backoff.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from webdav3.client import Client as WebDAVClient

from .extraction_schema import (
    EXTRACTION_VERSION,
    ExtractionResult,
    MathObject,
    latex_sanity_check,
    validate_extraction,
)
from .job_ledger import JobLedger
from .notion_client_wrapper import NotionClientWrapper
from .tag_linter import TagLinter, lint_report_to_text

logger = logging.getLogger(__name__)

# ── Scratch directory inside the Docker volume ─────────────────────────────────
TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))

# ── OpenAI model ───────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o"

# ── Notion hard limits ─────────────────────────────────────────────────────────
# Notion's API rejects any rich_text content exceeding 2000 characters.
# We use 1900 as our safe ceiling to avoid off-by-one edge cases.
NOTION_BLOCK_MAX_CHARS = 1900
# Notion allows at most 100 child blocks per append_block_children call.
NOTION_BLOCKS_PER_REQUEST = 100

# ── Zotero key regex ───────────────────────────────────────────────────────────
# Matches exactly 8 uppercase alphanumeric characters that are NOT surrounded
# by other uppercase-alphanumeric characters (i.e. not part of a longer token).
# Negative lookbehind/lookahead on [A-Z0-9] acts as a word-boundary for this
# character class since \b treats digits and uppercase as word chars.
# Maximum number of validation errors to include in the repair prompt.
MAX_REPAIR_ERRORS = 5

_ZOTERO_KEY_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{8})(?![A-Z0-9])")

# ── System prompt template ─────────────────────────────────────────────────────
# [INJECT_DYNAMIC_HUBS_HERE] is replaced at runtime with the live hub list.
EXTRACTION_SYSTEM_PROMPT = """\
You are a highly rigorous researcher in applied mathematics. Process the Markdown paper \
and extract strictly factual mathematical structures.

Rules:
1. Extract exact mathematical formulations. Do not paraphrase.
2. Format variables and equations in valid LaTeX ($ for inline, $$ for display).
3. Explicitly extract boundary conditions or assumptions. If none, write \
"None explicitly stated."
4. For hub suggestions: provide descriptive text only from ALLOWED_HUBS. \
Do not invent hubs. If none fit, use "Uncategorized".
5. For keywords: use lowercase, hyphen-separated terms (2–6 per field).
6. For setting: use values such as finite_state, continuous, graphon, ergodic, common_noise.
7. For result_category: use exactly one of: existence, uniqueness, convergence, \
stability, approximation — or omit if inapplicable.

ALLOWED_HUBS: [INJECT_DYNAMIC_HUBS_HERE]

You MUST respond in valid JSON matching this EXACT schema:
{
  "one_liner": "string - one sentence summary",
  "active_themes": ["string"],
  "extracted_concepts": [
    {
      "type": "Definition | Theorem | Lemma | Algorithm | Assumption | Proof",
      "title": "string - short label or theorem number",
      "statement_latex": "string - exact statement in valid LaTeX",
      "assumptions": "string - boundary conditions or None explicitly stated.",
      "variables": "string - comma-separated variable descriptions",
      "conclusion": "string - result in plain English",
      "source_pages": [1, 2],
      "source_quotes": "optional verbatim quote max 25 words or null",
      "confidence": 0.95,
      "suggested_hub": "string from ALLOWED_HUBS",
      "canonical_keywords": ["keyword1", "keyword2"],
      "prereq_keywords": ["keyword1"],
      "downstream_keywords": ["keyword1"],
      "interpretation": "optional plain-English meaning",
      "proof_idea": "optional proof sketch",
      "source_anchors": "optional section/equation refs e.g. Section 3.2; Eq. (12)",
      "named_tools": ["optional named mathematical tool"],
      "setting": ["optional setting tag"],
      "result_category": "optional: existence|uniqueness|convergence|stability|approximation",
      "aliases": "optional alternative names"
    }
  ]
}
"""


class IngestionEngine:
    """Module 1: Core Ingestion Engine (Notion -> WebDAV -> Marker -> OpenAI -> Notion)."""

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._webdav = self._build_webdav_client()
        self.marker_url = os.environ.get("MARKER_API_URL", "http://marker-api:8080")
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.koofr_base = os.environ.get("KOOFR_PDF_PATH", "/zotero")
        self._ledger = JobLedger()
        self._tag_linter = TagLinter()

    # ── WebDAV client factory ──────────────────────────────────────────────────

    @staticmethod
    def _build_webdav_client() -> WebDAVClient:
        options = {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": os.environ["KOOFR_USER"],
            "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
        }
        return WebDAVClient(options)

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Poll 'Paper Tracker' for s1-process-math papers and run the full
        ingestion pipeline on each one.

        Hubs and the Second Brain keyword index are fetched once per run()
        invocation so that every paper in the same batch uses a consistent
        snapshot.
        """
        logger.info("Ingestion: polling for s1-process-math papers ...")
        pages = self.notion.query_database(
            self.paper_tracker_db,
            filter={
                "property": "Status",
                "status": {"equals": "s1-process-math"},
            },
        )
        logger.info("Ingestion: found %d paper(s) to process.", len(pages))

        if not pages:
            return

        hubs: dict[str, str] = self._fetch_allowed_hubs()
        sb_index: dict[str, list[dict]] = self._build_second_brain_index()
        logger.info(
            "Ingestion: loaded %d hub(s), %d Second Brain keyword(s).",
            len(hubs),
            len(sb_index),
        )

        for page in pages:
            try:
                self._process_paper(page, hubs, sb_index)
            except Exception:
                logger.exception("Failed to process page %s", page["id"])

    # ── Hub fetching ───────────────────────────────────────────────────────────

    def _fetch_allowed_hubs(self) -> dict[str, str]:
        """
        Query the Second Brain DB for Hub pages.

        Returns
        -------
        dict mapping hub name -> Notion page ID.
        """
        logger.debug("Ingestion: fetching Hub pages from Second Brain ...")
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Note Level",
                "select": {"equals": "Hub"},
            },
        )
        hubs: dict[str, str] = {}
        for page in pages:
            name = self._get_page_title(page)
            if name:
                hubs[name] = page["id"]
        return hubs

    def _build_second_brain_index(self) -> dict[str, list[dict]]:
        """
        Query the Second Brain DB for Concept pages and build a keyword → page
        index used during the LINK stage to generate edge suggestions.

        The ``Note Level`` value to filter on is controlled by the
        ``SB_CONCEPT_LEVEL`` environment variable (default: ``"Concept"``).

        Returns
        -------
        dict mapping lowercase keyword string → list of
        ``{"page_id": str, "name": str}`` dicts.
        """
        concept_level = os.environ.get("SB_CONCEPT_LEVEL", "Concept")
        logger.debug(
            "Ingestion: building Second Brain index (Note Level='%s') ...",
            concept_level,
        )
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Note Level",
                "select": {"equals": concept_level},
            },
        )
        index: dict[str, list[dict]] = {}
        for page in pages:
            name = self._get_page_title(page)
            if not name:
                continue
            # "Keywords" multi-select (new property name per spec)
            keywords = self._get_multi_select_prop(page["properties"], "Keywords")
            if not keywords:
                # Fallback: use the page name itself as a keyword.
                # Log at debug so operators can identify pages missing Keywords.
                logger.debug(
                    "SB index: page '%s' has no Keywords property — using title as fallback.",
                    name,
                )
                keywords = [name]
            for kw in keywords:
                key = kw.lower().strip()
                if key:
                    index.setdefault(key, []).append(
                        {"page_id": page["id"], "name": name}
                    )
        logger.debug(
            "Ingestion: Second Brain index built — %d keyword(s) across %d page(s).",
            len(index),
            len(pages),
        )
        return index

    # ── Per-paper pipeline ─────────────────────────────────────────────────────

    def _process_paper(
        self,
        page: dict,
        hubs: dict[str, str],
        sb_index: dict[str, list[dict]],
    ) -> None:
        """
        Full ingestion pipeline for a single paper.

        Implements all preflight gates, tag completeness gate, extraction,
        validation, and Notion write-back.
        """
        page_id = page["id"]
        props = page["properties"]

        # ── Preflight gate 1: Parse Zotero key ────────────────────────────────
        zotero_uri = self._get_text_prop(props, "Zotero URI")
        match = _ZOTERO_KEY_RE.search(zotero_uri)
        if not match:
            logger.warning(
                "[%s] Missing or invalid Zotero URI: '%s' -- skipping.", page_id, zotero_uri
            )
            return
        zotero_key = match.group(1)

        # Short unique ID for log correlation and temp-file naming.
        run_id = uuid.uuid4().hex[:8]
        local_pdf: Path | None = None
        job_id: int | None = None

        try:
            # ── Preflight gate 2: Check zip exists ────────────────────────────
            zip_remote = f"{self.koofr_base}/{zotero_key}.zip"
            logger.info("[%s] Checking Koofr zip: %s", run_id, zip_remote)
            if not self._koofr_exists(zip_remote):
                logger.warning("[%s] Zip not found -- setting s1b-waiting-attachment.", run_id)
                self.notion.update_page(
                    page_id=page_id,
                    properties={"Status": self.notion.select_prop("s1b-waiting-attachment")},
                )
                return

            # ── Preflight gate 3: Download zip and extract PDF ────────────────
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            local_zip = TMP_DIR / f"{run_id}.zip"
            local_pdf = TMP_DIR / f"{run_id}.pdf"

            self._download_koofr(zip_remote, local_zip)

            primary_filename = self._get_text_prop(props, "primary_pdf_filename")
            self._extract_pdf_from_zip(
                local_zip, local_pdf, preferred=primary_filename or None
            )
            local_zip.unlink(missing_ok=True)

            # ── Preflight gate 4: Compute SHA256 ──────────────────────────────
            pdf_sha256 = self._sha256(local_pdf)
            logger.info("[%s] PDF SHA256: %s", run_id, pdf_sha256)
            self.notion.update_page(
                page_id=page_id,
                properties={"PDF SHA256": {"rich_text": self.notion.rich_text(pdf_sha256)}},
            )

            # ── Preflight gate 5: Idempotency check ───────────────────────────
            if self._ledger.is_already_done(zotero_key, pdf_sha256, EXTRACTION_VERSION):
                logger.info(
                    "[%s] Already processed (ledger hit) -- marking s2b-linked-ai.", run_id
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={"Status": self.notion.select_prop("s2b-linked-ai")},
                )
                return

            # ── Tag completeness gate ─────────────────────────────────────────
            tags = self._get_multi_select_prop(props, "Tags")
            lint_report = self._tag_linter.lint(tags)
            if not lint_report.valid_tags:
                report_text = lint_report_to_text(lint_report)
                logger.warning("[%s] Tag gate failed -- blocking.", run_id)
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.select_prop("blocked-tags"),
                        "tag_lint_report": {
                            "rich_text": self.notion.rich_text(report_text)
                        },
                    },
                )
                return

            if lint_report.errors:
                # Has valid tags but also has issues -- store report for awareness.
                report_text = lint_report_to_text(lint_report)
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "tag_lint_report": {
                            "rich_text": self.notion.rich_text(report_text)
                        }
                    },
                )

            # ── Start job ledger ──────────────────────────────────────────────
            job_id = self._ledger.start_job(zotero_key, pdf_sha256, EXTRACTION_VERSION)
            logger.info("[%s] JobLedger job_id=%d", run_id, job_id)

            # ── Stage 1 / Step 1: Convert PDF to Markdown ─────────────────────
            logger.info("[%s] Converting PDF to Markdown ...", run_id)
            markdown_text = self._pdf_to_markdown(local_pdf)
            self._ledger.update_status(job_id, "marker_done")

            # ── Stage 1 / Step 2: Extract via OpenAI ─────────────────────────
            logger.info("[%s] Extracting knowledge via OpenAI GPT-4o ...", run_id)
            extraction = self._extract_and_validate(markdown_text, hubs, run_id)
            self._ledger.update_status(job_id, "openai_done")

            # ── Stage 1 / Step 3: Patch Paper Tracker → s2-extracted ──────────
            logger.info("[%s] Patching Notion paper row (s2-extracted) ...", run_id)
            self._patch_notion_page(page_id, extraction, run_id)

            # ── Stage 1 / Step 4: Create Knowledge Inbox entries with edges ───
            concepts = extraction.extracted_concepts
            logger.info(
                "[%s] Creating %d Knowledge Inbox page(s) ...", run_id, len(concepts)
            )
            ki_created = 0
            for concept in concepts:
                try:
                    self._create_knowledge_item(page_id, concept, sb_index)
                    ki_created += 1
                except Exception:
                    logger.exception(
                        "[%s] Failed to create knowledge item '%s'",
                        run_id,
                        concept.title,
                    )
            logger.info("[%s] Created %d Knowledge Inbox page(s).", run_id, ki_created)

            # ── Stage 3 / Step 5: Patch Paper Tracker → s2b-linked-ai ─────────
            self._patch_notion_paper_post_linking(page_id, run_id)

            self._ledger.update_status(job_id, "notion_done")
            self._ledger.finish_job(job_id)
            logger.info("[%s] Done.", run_id)

        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", run_id, exc)
            if job_id is not None:
                self._ledger.update_status(job_id, "failed", error=str(exc))
            # Best-effort: write error context to Paper Tracker
            try:
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Last Error": {
                            "rich_text": self.notion.rich_text(str(exc)[:2000])
                        },
                        "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
                    },
                )
            except Exception:
                logger.warning(
                    "[%s] Could not write error context to Notion Paper Tracker.", run_id
                )
            raise

        finally:
            if local_pdf is not None and local_pdf.exists():
                local_pdf.unlink()

    # ── Koofr helpers ──────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _koofr_exists(self, remote_path: str) -> bool:
        """Return True if *remote_path* exists on Koofr."""
        try:
            return self._webdav.check(remote_path)
        except Exception:
            return False

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _download_koofr(self, remote_path: str, local_path: Path) -> None:
        """Download *remote_path* from Koofr to *local_path*."""
        self._webdav.download_sync(remote_path=remote_path, local_path=str(local_path))

    @staticmethod
    def _extract_pdf_from_zip(
        zip_path: Path, output_path: Path, preferred: str | None = None
    ) -> None:
        """
        Extract a PDF from *zip_path* to *output_path*.

        If *preferred* is set and found in the archive, use that file.
        Otherwise, extract the largest PDF in the archive.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            pdf_entries = [
                e for e in zf.infolist() if e.filename.lower().endswith(".pdf")
            ]
            if not pdf_entries:
                raise FileNotFoundError(f"No PDF found inside {zip_path}")

            if preferred:
                match = next(
                    (e for e in pdf_entries if Path(e.filename).name == preferred),
                    None,
                )
                if match:
                    data = zf.read(match.filename)
                    output_path.write_bytes(data)
                    return
                logger.warning(
                    "primary_pdf_filename '%s' not found in zip; using largest PDF.",
                    preferred,
                )

            # Select the largest PDF by uncompressed size.
            largest = max(pdf_entries, key=lambda e: e.file_size)
            data = zf.read(largest.filename)
            output_path.write_bytes(data)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute the hex SHA-256 digest of a file."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── Marker API ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _pdf_to_markdown(self, pdf_path: Path) -> str:
        """
        Ask the local marker-api container to convert the PDF and return
        the resulting Markdown string.

        Protocol: POST /marker  {"filepath": "<absolute path in container>"}
        Response shape: {"markdown": "...", "metadata": {...}, ...}
        """
        response = requests.post(
            f"{self.marker_url}/marker",
            json={"filepath": str(pdf_path)},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("markdown")
            or data.get("output")
            or data.get("text")
            or response.text
        )

    # ── OpenAI extraction ──────────────────────────────────────────────────────

    def _extract_and_validate(
        self, markdown: str, hubs: dict[str, str], run_id: str
    ) -> ExtractionResult:
        """
        Call OpenAI, validate the response, attempt a repair pass if needed,
        and run latex_sanity_check on each concept.
        """
        raw = self._call_openai(markdown, hubs)
        result, errors = validate_extraction(raw)

        if errors:
            logger.warning(
                "[%s] Pydantic validation failed (%d error(s)) -- attempting repair.",
                run_id,
                len(errors),
            )
            total_errors = len(errors)
            error_summary = "; ".join(errors[:MAX_REPAIR_ERRORS])
            if total_errors > MAX_REPAIR_ERRORS:
                error_summary += (
                    f" … (showing {MAX_REPAIR_ERRORS} of {total_errors} errors)"
                )
            raw2 = self._call_openai_repair(raw, error_summary)
            result, errors2 = validate_extraction(raw2)
            if errors2:
                logger.error(
                    "[%s] Repair also failed -- flagging concepts with confidence=0.",
                    run_id,
                )
                for concept in result.extracted_concepts:
                    concept.confidence = 0.0

        # LaTeX sanity check on each concept's statement.
        for concept in result.extracted_concepts:
            issues = latex_sanity_check(concept.statement_latex)
            if issues:
                logger.warning(
                    "[%s] LaTeX issues in concept '%s': %s",
                    run_id,
                    concept.title,
                    issues,
                )
                concept.confidence = min(concept.confidence, 0.5)

        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _call_openai(self, markdown: str, hubs: dict[str, str]) -> dict[str, Any]:
        """Send the paper Markdown to GPT-4o and return the parsed JSON dict."""
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = EXTRACTION_SYSTEM_PROMPT.replace(
            "[INJECT_DYNAMIC_HUBS_HERE]", hub_names_str
        )
        truncated_markdown = markdown[:100_000]

        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Extract structured knowledge from the following "
                        "academic paper (Markdown format).\n\n"
                        f"{truncated_markdown}"
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _call_openai_repair(
        self, invalid_output: dict[str, Any], error_summary: str
    ) -> dict[str, Any]:
        """Send the invalid output back to OpenAI with a repair instruction."""
        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON repair assistant. Fix the following JSON to match "
                        "the required schema. Return only valid JSON, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The following JSON failed validation with these errors:\n"
                        f"{error_summary}\n\n"
                        f"Invalid JSON:\n{json.dumps(invalid_output, indent=2)}\n\n"
                        "Return the corrected JSON."
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    # ── Patch Paper Tracker row ────────────────────────────────────────────────

    def _patch_notion_page(
        self, page_id: str, result: ExtractionResult, run_id: str
    ) -> None:
        """
        Update the Paper Tracker page after a successful extraction (s2-extracted):
          Status            -> s2-extracted
          AI Status         -> Unverified-AI
          One Liner         -> result.one_liner
          Active Themes     -> result.active_themes
          Extraction Version -> EXTRACTION_VERSION
          Processed At      -> current UTC timestamp
          Last Run ID       -> run_id
          Last Error        -> "" (cleared)
        """
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.select_prop("s2-extracted"),
                "AI Status": self.notion.select_prop("Unverified-AI"),
                "One Liner": {"rich_text": self.notion.rich_text(result.one_liner)},
                "Active Themes": self.notion.multi_select_prop(result.active_themes),
                "Extraction Version": {
                    "rich_text": self.notion.rich_text(EXTRACTION_VERSION)
                },
                "Processed At": {
                    "date": {
                        "start": datetime.now(tz=timezone.utc).isoformat()
                    }
                },
                "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
                "Last Error": {"rich_text": self.notion.rich_text("")},
            },
        )

    def _patch_notion_paper_post_linking(self, page_id: str, run_id: str) -> None:
        """
        Update the Paper Tracker page after the LINK stage completes:
          Status       -> s2b-linked-ai
          Last Run ID  -> run_id
        """
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.select_prop("s2b-linked-ai"),
                "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
            },
        )

    # ── Create Knowledge Inbox entry ───────────────────────────────────────────

    def _create_knowledge_item(
        self,
        paper_page_id: str,
        concept: MathObject,
        sb_index: dict[str, list[dict]],
    ) -> None:
        """
        Materialise a single MathObject as a Knowledge Inbox Notion page.

        Properties (metadata):
          - Name, Type, Status, verification_status
          - Source Paper (relation)
          - Source Pages (rich_text)
          - Suggested Hub (select — spec name, replaces "Hub Suggestions" JSON)
          - Confidence (number)
          - Keywords / Prereq Keywords / Downstream Keywords (multi-select)
          - Source Anchors, Interpretation, Proof Idea, Aliases (rich_text)
          - Named Tools, Setting (multi-select)
          - Result Category (select)
          - Source Quote (rich_text)
          - Edge Suggestions (rich_text JSON — generated from SB index)
          - graph_link_status (select = "needs-review" when edge suggestions present)

        Page body:
          - heading_2("Assumptions") + paragraphs
          - heading_2("Statement") + paragraphs with LaTeX
          - heading_2("Variables") + paragraphs (if present)
          - heading_2("Conclusion") + paragraphs (if present)
          - heading_2("Source Quote") + paragraphs (if present)
          - heading_2("Interpretation") + paragraphs (if present)
          - heading_2("Proof Idea") + paragraphs (if present)
          - heading_2("Edge Suggestions") + paragraphs (if any)
        """
        kind = concept.type
        title = concept.title

        # Source pages as comma-separated string.
        source_pages_str = (
            ", ".join(str(p) for p in concept.source_pages)
            if concept.source_pages
            else ""
        )

        # Generate edge suggestions from Second Brain index (LINK stage).
        edge_suggestions = self._generate_edge_suggestions(concept, sb_index)
        edge_suggestions_json = ""
        if edge_suggestions:
            # Trim the list until the serialized JSON fits within Notion's limit.
            # This preserves valid JSON rather than truncating mid-string.
            for cutoff in range(len(edge_suggestions), 0, -1):
                candidate = json.dumps(
                    edge_suggestions[:cutoff], ensure_ascii=False
                )
                if len(candidate) <= NOTION_BLOCK_MAX_CHARS:
                    edge_suggestions_json = candidate
                    break

        properties: dict = {
            "Name": self.notion.title_prop(f"[{kind}] {title}"),
            "Type": self.notion.select_prop(kind),
            "Status": self.notion.select_prop("Inbox"),
            "verification_status": self.notion.select_prop("unverified"),
            "Source Paper": self.notion.relation_prop([paper_page_id]),
        }

        if source_pages_str:
            properties["Source Pages"] = {
                "rich_text": self.notion.rich_text(source_pages_str)
            }

        # Suggested Hub — select (spec name replaces "Hub Suggestions" rich_text JSON).
        if concept.suggested_hub:
            properties["Suggested Hub"] = self.notion.select_prop(concept.suggested_hub)

        # Confidence — number.
        properties["Confidence"] = {"number": concept.confidence}

        # Keyword multi-selects.
        if concept.canonical_keywords:
            properties["Keywords"] = self.notion.multi_select_prop(
                concept.canonical_keywords
            )
        if concept.prereq_keywords:
            properties["Prereq Keywords"] = self.notion.multi_select_prop(
                concept.prereq_keywords
            )
        if concept.downstream_keywords:
            properties["Downstream Keywords"] = self.notion.multi_select_prop(
                concept.downstream_keywords
            )

        # Optional rich_text fields.
        if concept.source_anchors:
            properties["Source Anchors"] = {
                "rich_text": self.notion.rich_text(concept.source_anchors)
            }
        if concept.interpretation:
            properties["Interpretation"] = {
                "rich_text": self.notion.rich_text(concept.interpretation)
            }
        if concept.proof_idea:
            properties["Proof Idea"] = {
                "rich_text": self.notion.rich_text(concept.proof_idea)
            }
        if concept.aliases:
            properties["Aliases"] = {
                "rich_text": self.notion.rich_text(concept.aliases)
            }
        if concept.source_quotes:
            properties["Source Quote"] = {
                "rich_text": self.notion.rich_text(concept.source_quotes)
            }

        # Optional multi-select fields.
        if concept.named_tools:
            properties["Named Tools"] = self.notion.multi_select_prop(concept.named_tools)
        if concept.setting:
            properties["Setting"] = self.notion.multi_select_prop(concept.setting)

        # Optional select fields.
        if concept.result_category:
            properties["Result Category"] = self.notion.select_prop(
                concept.result_category
            )

        # Edge suggestions JSON + graph_link_status.
        if edge_suggestions_json:
            properties["Edge Suggestions"] = {
                "rich_text": self.notion.rich_text(edge_suggestions_json)
            }
            properties["graph_link_status"] = self.notion.select_prop("needs-review")

        new_page = self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties=properties,
        )
        new_page_id: str = new_page["id"]
        logger.debug(
            "Created Knowledge Inbox page %s for concept '%s'.", new_page_id, title
        )

        # Build page body blocks.
        body_blocks: list[dict] = []

        body_blocks.append(self._heading_block("Assumptions"))
        body_blocks.extend(self._paragraph_blocks(concept.assumptions))

        body_blocks.append(self._heading_block("Statement"))
        body_blocks.extend(self._paragraph_blocks(concept.statement_latex))

        if concept.variables:
            body_blocks.append(self._heading_block("Variables"))
            body_blocks.extend(self._paragraph_blocks(concept.variables))

        if concept.conclusion:
            body_blocks.append(self._heading_block("Conclusion"))
            body_blocks.extend(self._paragraph_blocks(concept.conclusion))

        if concept.source_quotes:
            body_blocks.append(self._heading_block("Source Quote"))
            body_blocks.extend(self._paragraph_blocks(concept.source_quotes))

        if concept.interpretation:
            body_blocks.append(self._heading_block("Interpretation"))
            body_blocks.extend(self._paragraph_blocks(concept.interpretation))

        if concept.proof_idea:
            body_blocks.append(self._heading_block("Proof Idea"))
            body_blocks.extend(self._paragraph_blocks(concept.proof_idea))

        if edge_suggestions:
            edge_lines = "\n".join(
                f"• {s['relation_type']}: {s['target_name']}"
                f" (conf {s['confidence']:.2f}) — {s['rationale']}"
                for s in edge_suggestions
            )
            body_blocks.append(self._heading_block("Edge Suggestions"))
            body_blocks.extend(self._paragraph_blocks(edge_lines))

        self._append_blocks_in_batches(new_page_id, body_blocks)

    # ── Edge suggestion generation (LINK stage) ────────────────────────────────

    def _generate_edge_suggestions(
        self,
        concept: MathObject,
        sb_index: dict[str, list[dict]],
    ) -> list[dict]:
        """
        Match concept keywords against the Second Brain index to produce a list
        of edge suggestion dicts.

        Each dict has keys: target_concept_id, target_name, relation_type,
        rationale, confidence.

        Returns at most 10 suggestions to keep the JSON within Notion limits.
        """
        suggestions: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()  # (target_page_id, relation_type)

        keyword_edge_map: list[tuple[list[str], str, float]] = [
            (concept.prereq_keywords, "depends_on", 0.7),
            (concept.downstream_keywords, "enables", 0.7),
            (concept.canonical_keywords, "related", 0.5),
        ]

        for keywords, relation_type, base_confidence in keyword_edge_map:
            for kw in keywords:
                for match in sb_index.get(kw.lower(), []):
                    pair = (match["page_id"], relation_type)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    suggestions.append(
                        {
                            "target_concept_id": match["page_id"],
                            "target_name": match["name"],
                            "relation_type": relation_type,
                            "rationale": f"Keyword match: '{kw}'",
                            "confidence": base_confidence,
                        }
                    )
                    if len(suggestions) >= 10:
                        return suggestions

        return suggestions

    # ── Text chunking ──────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_len: int = NOTION_BLOCK_MAX_CHARS) -> list[str]:
        """
        Split `text` into a list of strings each no longer than `max_len`
        characters, preferring newline split points to preserve LaTeX structure.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_len:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break

            split_pos = remaining.rfind("\n", 0, max_len)
            if split_pos <= 0:
                split_pos = max_len
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos:]
            else:
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos + 1:]

        return [c for c in chunks if c.strip()]

    # ── Notion block builders ──────────────────────────────────────────────────

    def _paragraph_blocks(self, text: str) -> list[dict]:
        """Convert a long string into a list of Notion paragraph block dicts."""
        return [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": chunk}}],
                },
            }
            for chunk in self._chunk_text(text)
        ]

    @staticmethod
    def _heading_block(text: str) -> dict:
        """Build a Notion heading_2 block for a section title."""
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
            },
        }

    def _append_blocks_in_batches(self, page_id: str, blocks: list[dict]) -> None:
        """Append blocks in batches of 100 to respect the Notion API limit."""
        for i in range(0, len(blocks), NOTION_BLOCKS_PER_REQUEST):
            batch = blocks[i : i + NOTION_BLOCKS_PER_REQUEST]
            self.notion.append_block_children(block_id=page_id, children=batch)

    # ── Property / page title helpers ──────────────────────────────────────────

    @staticmethod
    def _get_page_title(page: dict) -> str:
        """Extract the plain-text title from a raw Notion page object."""
        for value in page.get("properties", {}).values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return ""
        return ""

    @staticmethod
    def _get_text_prop(props: dict, key: str) -> str:
        """Extract plain text from a Notion rich_text property."""
        try:
            return props[key]["rich_text"][0]["plain_text"]
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _get_title_prop(props: dict) -> str:
        """Extract plain text from a Notion title property."""
        for value in props.values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return "unknown"
        return "unknown"

    @staticmethod
    def _get_multi_select_prop(props: dict, key: str) -> list[str]:
        """Extract option names from a Notion multi_select property."""
        try:
            return [opt["name"] for opt in props[key]["multi_select"]]
        except (KeyError, TypeError):
            return []

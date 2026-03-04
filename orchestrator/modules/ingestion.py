"""
modules/ingestion.py - Module 1: Core Ingestion Engine
-------------------------------------------------------
3-stage pipeline for each paper in the 'Paper Tracker' Notion DB with Status == 's1-process-math':

  PREFLIGHT GATES
  ---------------
  1. Parse Zotero parent key from "Zotero URI" rich_text / url property.
  1b. Resolve attachment key via Zotero API children endpoint.
  2. Check Koofr zip exists ({attachment_key}.zip); if missing set status "s1b-waiting-attachment".
  3. Download the zip, extract the largest PDF (or "primary_pdf_filename" if set).
  4. Compute pdf_sha256; store in "PDF SHA256" property.
  5. Idempotency check via JobLedger; if already done set status "s2b-linked-ai" and skip.

  TAG COMPLETENESS GATE
  ---------------------
  6. Read "Tags" multi-select; run TagLinter.
  7. If no valid tags: set status "blocked-tags", store lint report, return.

  STAGE 1 - EXTRACT
  -----------------
  8. Convert PDF to Markdown via marker-api (tenacity retry).
  9. Extract structured knowledge via GPT-4o (ExtractionResult schema).
  10. Validate with Pydantic; attempt one repair pass on failure.
  11. Run latex_sanity_check; downgrade confidence on failure.
  12. Patch Paper Tracker row to status "s2-extracted".
  13. Create Knowledge Inbox pages (graph_link_status = "unlinked").
  Ledger: extract_done

  STAGE 2 - RETRIEVE
  ------------------
  14. For each concept, score all Second Brain concepts by TF-IDF token overlap.
  15. Keep top-RETRIEVE_CANDIDATES_K candidates.
  Ledger: retrieve_done

  STAGE 3 - LINK
  --------------
  16. For each concept + candidates, call GPT to produce ConceptLinkResult edges.
  17. Write Edge Suggestions JSON + graph_link_status = "linked-ai" to KI page.
  18. Patch Paper Tracker row to status "s2b-linked-ai".
  Ledger: link_done -> notion_done

Design constraints:
  - Notion text blocks are hard-capped at 1900 chars (safe margin below 2000).
  - Hub suggestions stored as text only - never set Parent Hub relation automatically.
  - JobLedger tracks milestones for idempotency and restart safety.
  - All Koofr / Marker / OpenAI calls wrapped with tenacity exponential backoff.
  - Zotero parent key (from URI) != attachment key (PDF child item). Koofr stores {attachment_key}.zip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
    EDGE_CAPS,
    EXTRACTION_VERSION,
    ConceptLinkResult,
    ExtractionResult,
    MathObject,
    latex_sanity_check,
    validate_extraction,
    validate_link_result,
)
from .job_ledger import JobLedger
from .notion_client_wrapper import NotionClientWrapper
from .tag_linter import TagLinter, lint_report_to_text

logger = logging.getLogger(__name__)

# -- Scratch directory inside the Docker volume ---------------------------------
TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))

# -- OpenAI model ---------------------------------------------------------------
OPENAI_MODEL = "gpt-5.2"

# -- Notion hard limits ---------------------------------------------------------
NOTION_BLOCK_MAX_CHARS = 1900
NOTION_BLOCKS_PER_REQUEST = 100

# -- Stage 2 candidate retrieval limit -----------------------------------------
RETRIEVE_CANDIDATES_K: int = int(os.environ.get("RETRIEVE_CANDIDATES_K", "30"))

# -- Zotero --------------------------------------------------------------------
ZOTERO_API_BASE = "https://api.zotero.org"
# Matches the *parent* item key in a Zotero URI
_ZOTERO_PARENT_RE = re.compile(r"zotero\.org/[^/]+/items/([A-Z0-9]{8})(?:/|$)")
# Matches an explicit attachment key embedded in a Zotero URI
_ZOTERO_ATTACH_RE = re.compile(
    r"zotero\.org/[^/]+/items/[A-Z0-9]{8}/attachment/([A-Z0-9]{8})"
)

# -- Maximum validation errors shown in repair prompt --------------------------
MAX_REPAIR_ERRORS = 5

# -- Stage 3 linking system prompt ---------------------------------------------
LINKING_SYSTEM_PROMPT_V1 = """\
Given a mathematical concept and a list of candidate concepts from a knowledge base, \
identify which directed relationships hold between the given concept and the candidates.

For each relationship type, list the EXACT candidate titles that apply (use exact titles as given).
Return a JSON object with ONLY these keys:
  "depends_on"      -- concepts that this concept directly requires as prerequisites
  "enables"         -- concepts that follow naturally from or are unlocked by this one
  "generalizes"     -- concepts that this concept is a generalization of
  "special_case_of" -- concepts that this concept is a special case of
  "related"         -- loosely related concepts that share terminology or proof techniques

Rules:
- Values are arrays of exact candidate title strings.
- Omit a key entirely if no relationship of that type exists.
- Limit total references across ALL keys to 10.
- Prefer precision over recall; a false positive edge is worse than a missed one.
- Return ONLY valid JSON -- no explanation, no markdown fences.
"""

# -- System prompt template -----------------------------------------------------
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
5. For keywords: use lowercase, hyphen-separated terms (2-6 per field).
6. For setting: use values such as finite_state, continuous, graphon, ergodic, common_noise.
7. For result_category: use exactly one of: existence, uniqueness, convergence, \
stability, approximation -- or omit if inapplicable.

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
        self.zotero_user_id = os.environ["ZOTERO_USER_ID"]
        self.zotero_api_key = os.environ["ZOTERO_API_KEY"]

    # -- WebDAV client factory --------------------------------------------------

    @staticmethod
    def _build_webdav_client() -> WebDAVClient:
        options = {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": os.environ["KOOFR_USER"],
            "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
        }
        return WebDAVClient(options)

    # -- Entry point ------------------------------------------------------------

    def run(self) -> None:
        """
        Poll 'Paper Tracker' for s1-process-math papers and run the full
        ingestion pipeline on each one.

        Hubs and the Second Brain concept index are fetched once per run()
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
        sb_index: list[dict] = self._build_second_brain_index()
        logger.info(
            "Ingestion: loaded %d hub(s), %d Second Brain concept(s).",
            len(hubs),
            len(sb_index),
        )

        for page in pages:
            try:
                self._process_paper(page, hubs, sb_index)
            except Exception:
                logger.exception("Failed to process page %s", page["id"])

    # -- Hub fetching -----------------------------------------------------------

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

    # -- Second Brain index ----------------------------------------------------

    def _build_second_brain_index(self) -> list[dict]:
        """
        Query the Second Brain DB for Concept-level pages and return a flat
        list of concept records used by Stage 2 candidate retrieval.

        Each record has keys:
            id, title, hub, summary, tags, keywords_bag (set[str])

        The Note Level filter is controlled by SB_CONCEPT_LEVEL (default: "Concept").
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

            records.append(
                {
                    "id": page["id"],
                    "title": title,
                    "hub": hub,
                    "summary": summary,
                    "tags": tags,
                    "keywords_bag": bag,
                }
            )
        logger.debug(
            "Ingestion: Second Brain index built -- %d concept(s).", len(records)
        )
        return records

    # -- Per-paper pipeline ----------------------------------------------------

    def _process_paper(
        self,
        page: dict,
        hubs: dict[str, str],
        sb_index: list[dict],
    ) -> None:
        """
        Full ingestion pipeline for a single paper: preflight gates -> Stage 1
        (EXTRACT) -> Stage 2 (RETRIEVE) -> Stage 3 (LINK).
        """
        page_id = page["id"]
        props = page["properties"]

        # -- Preflight gate 1: Parse Zotero parent key --------------------------
        zotero_uri = self._get_text_prop(props, "Zotero URI")
        parent_match = _ZOTERO_PARENT_RE.search(zotero_uri)
        if not parent_match:
            logger.warning(
                "[%s] Missing or invalid Zotero URI: '%s' -- skipping.",
                page_id,
                zotero_uri,
            )
            return
        parent_key = parent_match.group(1)

        run_id = uuid.uuid4().hex[:8]
        local_pdf: Path | None = None
        job_id: int | None = None

        try:
            # -- Preflight gate 1b: Resolve attachment key ----------------------
            resolved = self._resolve_keys_and_update_notion(
                page_id, zotero_uri, parent_key, run_id
            )
            if resolved is None:
                logger.warning(
                    "[%s] Cannot resolve attachment key for parent '%s' "
                    "-- setting s1b-waiting-attachment.",
                    run_id,
                    parent_key,
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1b-waiting-attachment")
                    },
                )
                return
            parent_key, attachment_key = resolved

            # -- Preflight gate 2: Check Koofr zip exists ----------------------
            zip_remote = f"{self.koofr_base}/{attachment_key}.zip"
            logger.info("[%s] Checking Koofr zip: %s", run_id, zip_remote)
            if not self._koofr_exists(zip_remote):
                logger.warning(
                    "[%s] Zip not found -- setting s1b-waiting-attachment.", run_id
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1b-waiting-attachment")
                    },
                )
                return

            # -- Preflight gate 3: Download zip and extract PDF ----------------
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            local_zip = TMP_DIR / f"{run_id}.zip"
            local_pdf = TMP_DIR / f"{run_id}.pdf"

            self._download_koofr(zip_remote, local_zip)

            primary_filename = self._get_text_prop(props, "primary_pdf_filename")
            self._extract_pdf_from_zip(
                local_zip, local_pdf, preferred=primary_filename or None
            )
            local_zip.unlink(missing_ok=True)

            # -- Preflight gate 4: Compute SHA256 ------------------------------
            pdf_sha256 = self._sha256(local_pdf)
            logger.info("[%s] PDF SHA256: %s", run_id, pdf_sha256)
            self.notion.update_page(
                page_id=page_id,
                properties={"PDF SHA256": {"rich_text": self.notion.rich_text(pdf_sha256)}},
            )

            # -- Preflight gate 5: Idempotency check ---------------------------
            if self._ledger.is_already_done(attachment_key, pdf_sha256, EXTRACTION_VERSION):
                logger.info(
                    "[%s] Already processed (ledger hit) -- marking s2b-linked-ai.",
                    run_id,
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s2b-linked-ai")
                    },
                )
                return

            # -- Tag completeness gate -----------------------------------------
            tags = self._get_multi_select_prop(props, "Tags")
            lint_report = self._tag_linter.lint(tags)
            if not lint_report.valid_tags:
                report_text = lint_report_to_text(lint_report)
                logger.warning("[%s] Tag gate failed -- blocking.", run_id)
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("blocked-tags"),
                        "tag_lint_report": {
                            "rich_text": self.notion.rich_text(report_text)
                        },
                    },
                )
                return

            if lint_report.errors:
                report_text = lint_report_to_text(lint_report)
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "tag_lint_report": {
                            "rich_text": self.notion.rich_text(report_text)
                        }
                    },
                )

            # -- Start job ledger ----------------------------------------------
            job_id = self._ledger.start_job(attachment_key, pdf_sha256, EXTRACTION_VERSION)
            logger.info("[%s] JobLedger job_id=%d", run_id, job_id)

            # -- STAGE 1 / Step 1: Convert PDF to Markdown ---------------------
            logger.info("[%s] Stage 1: converting PDF to Markdown ...", run_id)
            markdown_text = self._pdf_to_markdown(local_pdf)
            self._ledger.update_status(job_id, "marker_done")

            # -- STAGE 1 / Step 2: Extract via OpenAI --------------------------
            logger.info("[%s] Stage 1: extracting knowledge via OpenAI ...", run_id)
            extraction = self._extract_and_validate(markdown_text, hubs, run_id)
            self._ledger.update_status(job_id, "openai_done")

            # -- STAGE 1 / Step 3: Patch Paper Tracker -> s2-extracted ---------
            logger.info("[%s] Stage 1: patching Notion paper row ...", run_id)
            self._patch_notion_page(page_id, extraction, run_id)

            # -- STAGE 1 / Step 4: Create Knowledge Inbox entries --------------
            concepts = extraction.extracted_concepts
            logger.info(
                "[%s] Stage 1: creating %d Knowledge Inbox page(s) ...",
                run_id,
                len(concepts),
            )
            ki_pages: list[tuple[MathObject, str]] = []
            for concept in concepts:
                try:
                    ki_page_id = self._create_knowledge_item(page_id, concept, hubs)
                    ki_pages.append((concept, ki_page_id))
                except Exception:
                    logger.exception(
                        "[%s] Failed to create knowledge item '%s'",
                        run_id,
                        concept.title,
                    )
            logger.info("[%s] Created %d Knowledge Inbox page(s).", run_id, len(ki_pages))
            self._ledger.update_status(job_id, "extract_done")

            # -- STAGE 2: Retrieve candidates ----------------------------------
            logger.info("[%s] Stage 2: retrieving candidates from Second Brain ...", run_id)
            concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
            for concept, ki_page_id in ki_pages:
                candidates = self._retrieve_candidates_for_concept(concept, sb_index)
                concept_candidates.append((concept, ki_page_id, candidates))
                logger.debug(
                    "[%s] '%s': %d candidate(s) retrieved.",
                    run_id,
                    concept.title,
                    len(candidates),
                )
            self._ledger.update_status(job_id, "retrieve_done")

            # -- STAGE 3: LLM linking ------------------------------------------
            logger.info("[%s] Stage 3: LLM linking ...", run_id)
            for concept, ki_page_id, candidates in concept_candidates:
                try:
                    link_result = self._run_stage_link(concept, candidates, run_id)
                    self._update_knowledge_item_graph_data(ki_page_id, link_result)
                except Exception:
                    logger.exception(
                        "[%s] Link stage failed for concept '%s'",
                        run_id,
                        concept.title,
                    )
            self._ledger.update_status(job_id, "link_done")

            # -- Finalise ------------------------------------------------------
            self._patch_notion_paper_post_linking(page_id, run_id)
            self._ledger.update_status(job_id, "notion_done")
            self._ledger.finish_job(job_id)
            logger.info("[%s] Done.", run_id)

        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", run_id, exc)
            if job_id is not None:
                self._ledger.update_status(job_id, "failed", error=str(exc))
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

    # -- Zotero helpers --------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _zotero_children(self, parent_key: str) -> list[dict]:
        """Fetch children of a Zotero item via the REST API."""
        url = (
            f"{ZOTERO_API_BASE}/users/{self.zotero_user_id}"
            f"/items/{parent_key}/children"
        )
        resp = requests.get(
            url,
            headers={"Zotero-API-Key": self.zotero_api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_attachment_key(
        self, zotero_uri: str, parent_key: str
    ) -> tuple[str, str] | None:
        """
        Resolve the attachment key (PDF child item) for a Zotero parent.

        Resolution order:
        1. If the URI itself contains an explicit attachment key, use it.
        2. Query Zotero API children; pick the first PDF attachment.

        Returns (parent_key, attachment_key) or None if not found.
        """
        attach_match = _ZOTERO_ATTACH_RE.search(zotero_uri)
        if attach_match:
            return parent_key, attach_match.group(1)

        try:
            children = self._zotero_children(parent_key)
        except Exception:
            logger.warning(
                "Could not fetch Zotero children for parent '%s'.", parent_key
            )
            return None

        for child in children:
            data = child.get("data", {})
            link_mode = data.get("linkMode", "")
            content_type = data.get("contentType", "")
            if link_mode in ("imported_file", "imported_url") and "pdf" in content_type:
                return parent_key, data["key"]

        logger.warning("No PDF attachment found for Zotero parent '%s'.", parent_key)
        return None

    def _resolve_keys_and_update_notion(
        self,
        page_id: str,
        zotero_uri: str,
        parent_key: str,
        run_id: str,
    ) -> tuple[str, str] | None:
        """
        Resolve (parent_key, attachment_key) and write the attachment key to
        the Paper Tracker Notion page for auditability.

        Returns (parent_key, attachment_key) or None on failure.
        """
        resolved = self._resolve_attachment_key(zotero_uri, parent_key)
        if resolved is None:
            return None
        _parent_key, attachment_key = resolved
        try:
            self.notion.update_page(
                page_id=page_id,
                properties={
                    "Zotero Attachment Key": {
                        "rich_text": self.notion.rich_text(attachment_key)
                    }
                },
            )
        except Exception:
            logger.warning(
                "[%s] Could not write Zotero Attachment Key to Notion.", run_id
            )
        return _parent_key, attachment_key

    # -- Koofr helpers ---------------------------------------------------------

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _koofr_exists(self, remote_path: str) -> bool:
        """Return True if remote_path exists on Koofr."""
        try:
            return self._webdav.check(remote_path)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "404" in exc_str or "not found" in exc_str or "no such" in exc_str:
                return False
            raise

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _download_koofr(self, remote_path: str, local_path: Path) -> None:
        """Download remote_path from Koofr to local_path."""
        self._webdav.download_sync(remote_path=remote_path, local_path=str(local_path))

    @staticmethod
    def _extract_pdf_from_zip(
        zip_path: Path, output_path: Path, preferred: str | None = None
    ) -> None:
        """
        Extract a PDF from zip_path to output_path.

        If preferred is set and found in the archive, use that file.
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

    # -- Marker API ------------------------------------------------------------

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _pdf_to_markdown(self, pdf_path: Path) -> str:
        """Ask the local marker-api container to convert the PDF and return Markdown."""
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

    # -- OpenAI extraction -----------------------------------------------------

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
                    f" ... (showing {MAX_REPAIR_ERRORS} of {total_errors} errors)"
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
        """Send the paper Markdown to GPT and return the parsed JSON dict."""
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = EXTRACTION_SYSTEM_PROMPT.replace(
            "[INJECT_DYNAMIC_HUBS_HERE]", hub_names_str
        )
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
                        f"{markdown[:100_000]}"
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

    # -- Patch Paper Tracker row -----------------------------------------------

    def _patch_notion_page(
        self, page_id: str, result: ExtractionResult, run_id: str
    ) -> None:
        """
        Update the Paper Tracker page after a successful extraction.

        Uses status_prop for the Status field (Notion status type).
        """
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.status_prop("s2-extracted"),
                "AI Status": self.notion.select_prop("Unverified-AI"),
                "One Liner": {"rich_text": self.notion.rich_text(result.one_liner)},
                "Active Themes": self.notion.multi_select_prop(result.active_themes),
                "Extraction Version": {
                    "rich_text": self.notion.rich_text(EXTRACTION_VERSION)
                },
                "Processed At": {
                    "date": {"start": datetime.now(tz=timezone.utc).isoformat()}
                },
                "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
                "Last Error": {"rich_text": self.notion.rich_text("")},
            },
        )

    def _patch_notion_paper_post_linking(self, page_id: str, run_id: str) -> None:
        """
        Update the Paper Tracker page after the LINK stage completes.

        Uses status_prop for the Status field (Notion status type).
        """
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.status_prop("s2b-linked-ai"),
                "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
            },
        )

    # -- Create Knowledge Inbox entry ------------------------------------------

    def _create_knowledge_item(
        self,
        paper_page_id: str,
        concept: MathObject,
        hubs: dict[str, str],
    ) -> str:
        """
        Materialise a single MathObject as a Knowledge Inbox Notion page.

        Sets graph_link_status = "unlinked" at creation.
        Stage 3 (_update_knowledge_item_graph_data) later writes edges and
        promotes to "linked-ai".

        Returns the Notion page ID of the created page.
        """
        kind = concept.type
        title = concept.title

        source_pages_str = (
            ", ".join(str(p) for p in concept.source_pages)
            if concept.source_pages
            else ""
        )

        properties: dict = {
            "Name": self.notion.title_prop(f"[{kind}] {title}"),
            "Type": self.notion.select_prop(kind),
            "Status": self.notion.select_prop("Inbox"),
            "verification_status": self.notion.select_prop("unverified"),
            "graph_link_status": self.notion.select_prop("unlinked"),
            "Source Paper": self.notion.relation_prop([paper_page_id]),
        }

        if source_pages_str:
            properties["Source Pages"] = {
                "rich_text": self.notion.rich_text(source_pages_str)
            }
        if concept.suggested_hub:
            properties["Suggested Hub"] = self.notion.select_prop(concept.suggested_hub)

        properties["Confidence"] = {"number": concept.confidence}

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
        if concept.named_tools:
            properties["Named Tools"] = self.notion.multi_select_prop(concept.named_tools)
        if concept.setting:
            properties["Setting"] = self.notion.multi_select_prop(concept.setting)
        if concept.result_category:
            properties["Result Category"] = self.notion.select_prop(
                concept.result_category
            )

        new_page = self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties=properties,
        )
        new_page_id: str = new_page["id"]
        logger.debug(
            "Created Knowledge Inbox page %s for concept '%s'.", new_page_id, title
        )

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

        self._append_blocks_in_batches(new_page_id, body_blocks)
        return new_page_id

    # -- Stage 3: write edge data to Knowledge Inbox page ----------------------

    def _update_knowledge_item_graph_data(
        self, ki_page_id: str, link_result: ConceptLinkResult
    ) -> None:
        """
        Write ConceptLinkResult edges to the KI page and set graph_link_status
        to "linked-ai". No-op if link_result is empty.
        """
        edge_dict = link_result.model_dump(exclude_none=True)
        edge_dict = {k: v for k, v in edge_dict.items() if v}
        if not edge_dict:
            logger.debug(
                "KI page %s: no edges produced -- remaining 'unlinked'.", ki_page_id
            )
            return
        edge_json = json.dumps(edge_dict, ensure_ascii=False)[:NOTION_BLOCK_MAX_CHARS]
        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "Edge Suggestions": {
                    "rich_text": self.notion.rich_text(edge_json)
                },
                "graph_link_status": self.notion.select_prop("linked-ai"),
            },
        )

    # -- Stage 2: candidate retrieval (TF-IDF token overlap) -------------------

    def _retrieve_candidates_for_concept(
        self,
        concept: MathObject,
        sb_index: list[dict],
        k: int = RETRIEVE_CANDIDATES_K,
    ) -> list[dict]:
        """
        Score every Second Brain concept record against this concept using
        TF-IDF-style token overlap with a hub-affinity bonus, return top-k.

        Score = |concept_tokens intersect r.keywords_bag| / log(1 + |bag|)
                + 0.2 if r.hub == concept.suggested_hub
        """
        def _toks(s: str) -> set:
            return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

        concept_tokens: set = set()
        for kw_list in (
            concept.canonical_keywords,
            concept.prereq_keywords,
            concept.downstream_keywords,
        ):
            for kw in kw_list:
                concept_tokens |= _toks(kw)
        concept_tokens |= _toks(concept.title)

        if not concept_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for record in sb_index:
            bag = record.get("keywords_bag", set())
            if not bag:
                continue
            overlap = len(concept_tokens & bag)
            if overlap == 0:
                continue
            score = overlap / math.log(1.0 + len(bag))
            if record.get("hub") and record["hub"] == concept.suggested_hub:
                score += 0.2
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    # -- Stage 3: LLM linking --------------------------------------------------

    def _run_stage_link(
        self,
        concept: MathObject,
        candidates: list[dict],
        run_id: str,
    ) -> ConceptLinkResult:
        """
        Call the OpenAI linking prompt and validate the result.
        Returns an empty ConceptLinkResult if there are no candidates or
        if the LLM response cannot be validated.
        """
        if not candidates:
            return ConceptLinkResult()
        try:
            raw = self._call_openai_link(concept, candidates)
        except Exception:
            logger.warning(
                "[%s] _call_openai_link failed for '%s'.", run_id, concept.title
            )
            return ConceptLinkResult()
        result = validate_link_result(raw)
        for rel, cap in EDGE_CAPS.items():
            current = getattr(result, rel, None)
            if current and len(current) > cap:
                setattr(result, rel, current[:cap])
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _call_openai_link(
        self, concept: MathObject, candidates: list[dict]
    ) -> dict[str, Any]:
        """Invoke the Stage 3 linking prompt via OpenAI."""
        candidate_lines = "\n".join(
            f"{i + 1}. {r['title']}"
            + (f" [{r['hub']}]" if r.get("hub") else "")
            + (f" -- {r['summary']}" if r.get("summary") else "")
            for i, r in enumerate(candidates)
        )
        concept_summary = (
            f"Title: {concept.title}\n"
            f"Type: {concept.type}\n"
            f"Conclusion: {concept.conclusion or '(none)'}\n"
            f"Keywords: {', '.join(concept.canonical_keywords)}\n"
            f"Prereq keywords: {', '.join(concept.prereq_keywords)}\n"
            f"Downstream keywords: {', '.join(concept.downstream_keywords)}"
        )
        user_message = (
            f"CONCEPT:\n{concept_summary}\n\n"
            f"CANDIDATES:\n{candidate_lines}\n\n"
            "Identify relationships. Return JSON only."
        )
        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LINKING_SYSTEM_PROMPT_V1},
                {"role": "user", "content": user_message},
            ],
        )
        return json.loads(response.choices[0].message.content)

    # -- Text chunking ---------------------------------------------------------

    def _chunk_text(self, text: str, max_len: int = NOTION_BLOCK_MAX_CHARS) -> list[str]:
        """Split text into chunks of at most max_len chars, preferring newlines."""
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

    # -- Notion block builders -------------------------------------------------

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

    # -- Property / page title helpers -----------------------------------------

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
        """
        Extract plain text from a Notion property.

        Handles both rich_text and url property types.
        """
        prop = props.get(key, {})
        if prop.get("type") == "url":
            return prop.get("url") or ""
        try:
            return prop["rich_text"][0]["plain_text"]
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

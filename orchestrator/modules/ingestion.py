"""
modules/ingestion.py - Module 1: Core Ingestion Engine (v3 — 3-stage pipeline)
-------------------------------------------------------------------------------
Pipeline for each paper in the 'Paper Tracker' Notion DB with Status == 's1-process-math':

  PREFLIGHT GATES
  ---------------
  1a. Parse parent item key from "Zotero URI" (Notero URL points to bibliographic item).
  1b. Resolve PDF attachment key via Zotero Web API (children endpoint).
        - If URL already encodes /attachment/<KEY> use that directly.
        - Otherwise GET /users/<uid>/items/<PARENT_KEY>/children, filter PDF attachments.
        - Store zotero_parent_key, zotero_attachment_key, attachment_resolution_status
          and attachment_resolution_log back to Notion.
  2. Check Koofr zip exists ({attachment_key}.zip); if missing set "s1b-waiting-attachment".
     LOCAL TEST MODE: looks for "{attachment_key}.zip" next to this module instead.
  3. Download the zip, extract the largest PDF (or "primary_pdf_filename" if set).
  4. Compute pdf_sha256; store in "PDF SHA256" property.
  5. Idempotency check via JobLedger; if already done set status "s2-extracted" and skip.

  TAG COMPLETENESS GATE
  ---------------------
  6. Read "Tags" multi-select; run TagLinter.
  7. If no valid tags: set status "blocked-tags", store lint report, return.

  STAGE 1 — EXTRACT (LLM)
  ------------------------
  8. Convert PDF to Markdown via marker-api (tenacity retry).
     LOCAL TEST MODE: reads pre-existing "{attachment_key}.md" next to this module.
  9. Extract structured knowledge via GPT (EXTRACTION_SYSTEM_PROMPT_V2):
     type, title, statement_latex, assumptions, variables, conclusion,
     source_pages, source_quotes, confidence, suggested_hub,
     canonical_keywords, prereq_keywords, downstream_keywords.
  10. Validate with Pydantic; attempt one repair pass on failure.
  11. Run latex_sanity_check; downgrade confidence on failure.
  12. Patch Paper Tracker row to status "s2-extracted".
  13. Create Knowledge Inbox pages with new graph fields.
  JobLedger checkpoint: extract_done

  STAGE 2 — RETRIEVE (deterministic, no LLM)
  -------------------------------------------
  14. Use pre-built index of Second Brain "Atomic Concept" pages (fetched once per run).
  15. Score each existing concept vs. the extracted concept using token overlap on
      titles + keywords + hub match bonus.  Top-K candidates returned (RETRIEVE_CANDIDATES_K).
  16. Serialize candidate list as JSON and store in "candidate_matches" on the
      Knowledge Inbox page.
  JobLedger checkpoint: retrieve_done

  STAGE 3 — LINK (LLM)
  ----------------------
  17. For each extracted concept + its top-K candidates, call GPT
      (LINKING_SYSTEM_PROMPT_V1) to produce edge_suggestions.
  18. Validate ConceptLinkResult with Pydantic; attempt one repair pass.
  19. Serialize edge_suggestions as JSON and store on the Knowledge Inbox page.
  20. Set graph_link_status = "linked-ai" on the Knowledge Inbox page.
  JobLedger checkpoint: link_done → notion_done

Design constraints:
  - Notion text blocks are hard-capped at 1900 chars (safe margin below 2000).
  - Hub suggestions stored as text only — never set Parent Hub relation automatically.
  - Edge suggestions stored as text only — never write live relations to Second Brain.
  - JobLedger tracks milestones for idempotency and restart safety.
  - All Koofr / Marker / OpenAI calls wrapped with tenacity exponential backoff.
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
from pathlib import Path
from typing import Any

import openai
import requests
from tenacity import (
    reraise,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from webdav3.client import Client as WebDAVClient

from .extraction_schema import (
    ConceptLinkResult,
    EDGE_CAPS,
    EXTRACTION_VERSION,
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

# ── Scratch directory inside the Docker volume ──────────────────────────────────────────────────────
TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))

# ── OpenAI model ──────────────────────────────────────────────────────────────────────
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ── Notion hard limits ────────────────────────────────────────────────────────────────
NOTION_BLOCK_MAX_CHARS = 1900
NOTION_BLOCKS_PER_REQUEST = 100

# ── Zotero URL regex ────────────────────────────────────────────────────────────────
_ZOTERO_PARENT_RE = re.compile(r"zotero\.org/[^/]+/items/([A-Z0-9]{8})")
_ZOTERO_ATTACH_RE = re.compile(
    r"zotero\.org/[^/]+/items/[A-Z0-9]{8}/attachment/([A-Z0-9]{8})"
)

# ── Zotero Web API ────────────────────────────────────────────────────────────────────────
ZOTERO_API_BASE = "https://api.zotero.org"

# Max Pydantic validation errors to include in the repair prompt.
MAX_REPAIR_ERRORS = 5

# Stage 2 retrieval: number of candidate concepts per extracted concept.
RETRIEVE_CANDIDATES_K: int = int(os.environ.get("RETRIEVE_CANDIDATES_K", "30"))


# ── Extraction system prompt (v2) ──────────────────────────────────────────────
# [INJECT_DYNAMIC_HUBS_HERE] is replaced at runtime with the live hub list.
EXTRACTION_SYSTEM_PROMPT_V2 = """\
You are a research assistant specialized in extracting precise mathematical knowledge
from scientific papers in applied mathematics, probability, optimization, and mean-field games.

Your task is to convert a Markdown representation of a research paper into a structured
set of reusable mathematical concepts suitable for a long-term research knowledge graph.

The goal is NOT summarization.  Extract the CORE REUSABLE MATHEMATICAL OBJECTS that
appear in the paper.

----------------------------------------------------------------------
CONCEPT NAMING RULES
----------------------------------------------------------------------
Each concept MUST have a descriptive canonical title.

DO NOT use numbering such as:
  ✗ "Theorem 1"
  ✗ "Lemma 3.2"

Instead use descriptive names:
  ✓ "Existence of Nash Equilibrium in Finite-State GMFG"
  ✓ "Lasry–Lions Monotonicity Condition"
  ✓ "Convergence of Policy Iteration for Finite-State MFG"

----------------------------------------------------------------------
WHAT TO EXTRACT (3–12 concepts per paper)
----------------------------------------------------------------------
GOOD:
  - Core definitions used throughout the paper
  - Main theoretical results (theorems, propositions)
  - Key lemmas that have independent reuse value
  - Algorithms or procedures
  - Important structural assumptions
  - Convergence/existence/uniqueness results

BAD (do NOT extract):
  - "Theorem 1" or "Equation 5" (non-descriptive names)
  - Intermediate proof steps used only locally
  - Discussion text or motivation paragraphs
  - Minor technical bounds without broader utility

Proofs: only extract if the PROOF TECHNIQUE is independently reusable.

----------------------------------------------------------------------
KEYWORD RULES
----------------------------------------------------------------------
Each concept must export three keyword lists (5–15 items each):

  canonical_keywords : What this concept IS (e.g. "Nash equilibrium",
                       "fixed-point", "mean-field limit", "monotonicity").

  prereq_keywords    : What this concept REQUIRES or builds on (e.g.
                       "Lipschitz continuity", "weak convergence",
                       "probability measure space").

  downstream_keywords: What this concept ENABLES or what uses it (e.g.
                       "epsilon-Nash", "MFG equilibrium characterization",
                       "numerical MFG solver").

Keywords should be short noun phrases (2–5 words).  Use the paper's own
notation where possible.

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Respond in VALID JSON ONLY.  No text outside the JSON block.

{
  "one_liner": "string",
  "active_themes": ["string"],
  "extracted_concepts": [
    {
      "type": "Definition | Theorem | Lemma | Algorithm | Assumption | Proof | ProofTechnique",
      "title": "string — descriptive canonical concept name",
      "statement_latex": "string — exact mathematical statement in LaTeX",
      "assumptions": "string or \"None explicitly stated.\"",
      "variables": "string — comma-separated variable descriptions",
      "conclusion": "string — result in plain English",
      "source_pages": [integer],
      "source_quotes": "string ≤25 words or null",
      "confidence": 0.0-1.0,
      "suggested_hub": "string from ALLOWED_HUBS",
      "canonical_keywords": ["string"],
      "prereq_keywords": ["string"],
      "downstream_keywords": ["string"]
    }
  ]
}

----------------------------------------------------------------------
QUALITY STANDARD
----------------------------------------------------------------------
Prefer FEWER, HIGHER-QUALITY concepts (3–12 per paper).
Each concept must represent a meaningful mathematical contribution.

----------------------------------------------------------------------
ALLOWED_HUBS
----------------------------------------------------------------------
[INJECT_DYNAMIC_HUBS_HERE]
"""

# ── Linking system prompt (v1) ─────────────────────────────────────────────────
LINKING_SYSTEM_PROMPT_V1 = """\
You are a knowledge-graph construction assistant.

Your task is to determine directed semantic edges between a NEWLY EXTRACTED CONCEPT
and a set of EXISTING CONCEPTS in a mathematical research knowledge graph.

----------------------------------------------------------------------
EDGE TYPES
----------------------------------------------------------------------
depends_on     : The new concept REQUIRES the target as a direct mathematical
                 prerequisite (e.g. uses its notation, builds on its result).
enables        : The new concept ENABLES or supports the target.
generalizes    : The new concept is a GENERALIZATION of the target (more general).
special_case_of: The new concept is a SPECIALIZATION of the target.
related        : The new concept is SEMANTICALLY RELATED but does not fit above.

----------------------------------------------------------------------
HARD CONSTRAINTS
----------------------------------------------------------------------
1. You MUST ONLY link to concepts from the CANDIDATE LIST provided below.
   Never invent concept IDs.  If no candidate fits, output empty lists.

2. Do NOT connect concepts merely because they appeared in the same paper.
   Connections must reflect genuine mathematical dependency or generalization.

3. Hard caps (max edges per type):
   depends_on    ≤ 3
   enables       ≤ 3
   generalizes   ≤ 2
   special_case_of ≤ 2
   related       ≤ 5

4. Each edge MUST include a one-sentence rationale and a confidence in [0,1].

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Respond in VALID JSON ONLY.  No text outside the JSON block.

{
  "depends_on": [
    {"target_concept_id": "...", "target_title": "...", "rationale": "...", "confidence": 0.9}
  ],
  "enables": [...],
  "generalizes": [...],
  "special_case_of": [...],
  "related": [...]
}

If a list is empty, output it as [].
"""


class IngestionEngine:
    """Module 1: Core Ingestion Engine (Notion → WebDAV → Marker → OpenAI → Notion)."""

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._webdav = self._build_webdav_client()
        self.marker_url = os.environ.get("MARKER_API_URL", "http://marker-api:8080")
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        # Koofr base path: should be an absolute WebDAV path like "/zotero"
        self.koofr_base = os.environ.get("KOOFR_PDF_PATH", "/zotero")
        self._ledger = JobLedger()
        self._tag_linter = TagLinter()
        self.zotero_user_id = os.environ["ZOTERO_USER_ID"]
        self.zotero_api_key = os.environ["ZOTERO_API_KEY"]

    # ── Entry point ────────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Poll 'Paper Tracker' for s1-process-math papers and run the full
        3-stage (EXTRACT → RETRIEVE → LINK) pipeline on each.

        Hubs and the Second Brain concept index are fetched once per run()
        invocation so that all papers in the same batch use a consistent snapshot.
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
        logger.info(
            "Ingestion: loaded %d hub(s) from Second Brain: %s",
            len(hubs),
            list(hubs.keys()),
        )

        sb_index: list[dict] = self._build_second_brain_index()
        logger.info(
            "Ingestion: Second Brain index built (%d atomic concepts).", len(sb_index)
        )

        for page in pages:
            try:
                self._process_paper(page, hubs, sb_index)
            except Exception:
                logger.exception("Failed to process page %s", page["id"])

    # ── Hub fetching ───────────────────────────────────────────────────────────────────────────

    def _fetch_allowed_hubs(self) -> dict[str, str]:
        """Query Second Brain for Hub pages.  Returns hub_name -> page_id."""
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

    # ── Zotero attachment resolution ──────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _zotero_children(self, parent_key: str) -> list[dict]:
        """Fetch children of a Zotero parent item via the Web API."""
        url = f"{ZOTERO_API_BASE}/users/{self.zotero_user_id}/items/{parent_key}/children"
        resp = requests.get(
            url,
            headers={"Zotero-API-Key": self.zotero_api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_attachment_key(
        self, zotero_uri: str, preferred_filename: str | None = None
    ) -> tuple[str, str]:
        """
        Resolve (parent_key, attachment_key) from a Notero Zotero URI.

        Raises ValueError if the URL cannot be parsed, RuntimeError if no PDF
        attachment is found among children.
        """
        m_attach = _ZOTERO_ATTACH_RE.search(zotero_uri)
        if m_attach:
            attach_key = m_attach.group(1)
            m_parent = _ZOTERO_PARENT_RE.search(zotero_uri)
            parent_key = m_parent.group(1) if m_parent else attach_key
            return parent_key, attach_key

        m_parent = _ZOTERO_PARENT_RE.search(zotero_uri)
        if not m_parent:
            raise ValueError(f"Cannot parse Zotero key from URI: {zotero_uri!r}")
        parent_key = m_parent.group(1)

        children = self._zotero_children(parent_key)

        pdfs: list[dict] = []
        for item in children:
            data = item.get("data", {})
            if data.get("itemType") != "attachment":
                continue
            fn = (data.get("filename") or "").lower()
            ct = (data.get("contentType") or "").lower()
            if ct == "application/pdf" or fn.endswith(".pdf"):
                pdfs.append({
                    "key": item.get("key"),
                    "filename": data.get("filename"),
                    "size": data.get("size") or 0,
                })

        if not pdfs:
            raise RuntimeError("No PDF attachment found among Zotero children.")

        if preferred_filename:
            for a in pdfs:
                if a["filename"] == preferred_filename:
                    return parent_key, a["key"]

        if len(pdfs) == 1:
            return parent_key, pdfs[0]["key"]

        pdfs.sort(key=lambda x: x["size"], reverse=True)
        logger.warning(
            "Multiple PDF attachments for parent %s; picking largest (%s, %d bytes).",
            parent_key, pdfs[0]["filename"], pdfs[0]["size"],
        )
        return parent_key, pdfs[0]["key"]

    # ── Preflight helpers ────────────────────────────────────────────────────────────────────

    def _resolve_keys_and_update_notion(
        self,
        page_id: str,
        zotero_uri: str,
        preferred_filename: str | None,
        run_id: str,
    ) -> tuple[str, str] | None:
        """
        Resolve (parent_key, attachment_key) and write result back to Notion.

        Returns the (parent_key, attachment_key) tuple on success, or None
        if resolution failed (Notion has already been set to s1b-waiting-attachment).
        """
        logger.info("[%s] Resolving Zotero attachment key for URI: %s", run_id, zotero_uri)
        try:
            parent_key, attachment_key = self._resolve_attachment_key(
                zotero_uri, preferred_filename=preferred_filename
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("[%s] Attachment resolution failed: %s", run_id, exc)
            self.notion.update_page(
                page_id=page_id,
                properties={
                    "Status": self.notion.status_prop("s1b-waiting-attachment"),
                    "attachment_resolution_status": self.notion.select_prop("error"),
                    "attachment_resolution_log": {
                        "rich_text": self.notion.rich_text(str(exc))
                    },
                },
            )
            return None

        logger.info(
            "[%s] Resolved parent=%s attachment=%s", run_id, parent_key, attachment_key
        )
        self.notion.update_page(
            page_id=page_id,
            properties={
                "zotero_parent_key": {"rich_text": self.notion.rich_text(parent_key)},
                "zotero_attachment_key": {
                    "rich_text": self.notion.rich_text(attachment_key)
                },
                "attachment_resolution_status": self.notion.select_prop("ok"),
                "attachment_resolution_log": {"rich_text": self.notion.rich_text("")},
            },
        )
        return parent_key, attachment_key

    def _load_markdown_for_paper(
        self,
        attachment_key: str,
        local_pdf: Path,
        run_id: str,
    ) -> str:
        """
        Convert the PDF to Markdown.

        LOCAL TEST MODE: if "{attachment_key}.md" exists next to this module,
        return its contents directly (skips marker-api and PDF).

        Production mode: post the PDF to marker-api.
        """
        script_dir = Path(__file__).resolve().parent
        md_path = script_dir / f"{attachment_key}.md"
        if md_path.exists():
            logger.info("[%s] LOCAL MD MODE: loading markdown from %s", run_id, md_path)
            return md_path.read_text(encoding="utf-8")

        logger.info("[%s] Converting PDF to Markdown via marker-api ...", run_id)
        return self._pdf_to_markdown(local_pdf)

    # ── WebDAV client factory ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_webdav_client() -> WebDAVClient:
        options = {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": os.environ["KOOFR_USER"],
            "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
        }
        return WebDAVClient(options)

    # ── Per-paper pipeline ─────────────────────────────────────────────────────────────────────────

    def _process_paper(
        self, page: dict, hubs: dict[str, str], sb_index: list[dict]
    ) -> None:
        """
        Full 3-stage ingestion pipeline for a single paper.

          preflight → tag gate → Stage 1 (extract) →
          Stage 2 (retrieve) → Stage 3 (link) → finalize
        """
        page_id = page["id"]
        props = page["properties"]

        # ── Preflight gate 1a: Zotero URI ──────────────────────────────────────────────
        zotero_uri = self._get_text_prop(props, "Zotero URI")
        if not _ZOTERO_PARENT_RE.search(zotero_uri):
            logger.warning(
                "[%s] Missing/unparseable Zotero URI: %r -- skipping.", page_id, zotero_uri
            )
            return

        run_id = uuid.uuid4().hex[:8]
        local_pdf: Path | None = None
        local_md: Path | None = None
        job_id: int | None = None

        try:
            # ── Preflight gate 1b: resolve attachment key ─────────────────────────────────
            preferred_filename = (
                self._get_text_prop(props, "primary_pdf_filename") or None
            )
            result = self._resolve_keys_and_update_notion(
                page_id, zotero_uri, preferred_filename, run_id
            )
            if result is None:
                return  # already set to s1b-waiting-attachment
            parent_key, attachment_key = result

            # ── Preflight gates 2–4: zip / PDF / SHA256 ─────────────────────────────────────
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            local_pdf = TMP_DIR / f"{run_id}.pdf"
            local_md = TMP_DIR / f"{run_id}.md"

            # Check LOCAL TEST MODE zip first; fall back to Koofr.
            script_dir = Path(__file__).resolve().parent
            local_zip = script_dir / f"{attachment_key}.zip"

            if local_zip.exists():
                logger.info(
                    "[%s] LOCAL ZIP MODE: %s -> %s", run_id, local_zip, local_pdf
                )
                self._extract_pdf_from_zip(
                    local_zip, local_pdf, preferred=preferred_filename
                )
            else:
                remote_path = f"{self.koofr_base}/{attachment_key}.zip"
                if not self._koofr_exists(remote_path):
                    logger.warning(
                        "[%s] Zip not found on Koofr: %s", run_id, remote_path
                    )
                    self.notion.update_page(
                        page_id=page_id,
                        properties={
                            "Status": self.notion.status_prop("s1b-waiting-attachment")
                        },
                    )
                    return
                logger.info(
                    "[%s] Downloading from Koofr: %s", run_id, remote_path
                )
                local_zip_tmp = TMP_DIR / f"{run_id}.zip"
                self._download_koofr(remote_path, local_zip_tmp)
                self._extract_pdf_from_zip(
                    local_zip_tmp, local_pdf, preferred=preferred_filename
                )
                local_zip_tmp.unlink(missing_ok=True)

            pdf_sha256 = self._sha256(local_pdf)
            logger.info("[%s] PDF SHA256: %s", run_id, pdf_sha256)
            self.notion.update_page(
                page_id=page_id,
                properties={
                    "PDF SHA256": {"rich_text": self.notion.rich_text(pdf_sha256)}
                },
            )

            # ── Preflight gate 5: idempotency ────────────────────────────────────────────────
            if self._ledger.is_already_done(
                attachment_key, pdf_sha256, EXTRACTION_VERSION
            ):
                logger.info(
                    "[%s] Ledger hit — already processed, marking s2-extracted.", run_id
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={"Status": self.notion.status_prop("s2-extracted")},
                )
                return

            # ── Tag completeness gate ──────────────────────────────────────────────────────────
            tags = self._get_multi_select_prop(props, "Tags")
            lint_report = self._tag_linter.lint(tags)
            if not lint_report.valid_tags:
                report_text = lint_report_to_text(lint_report)
                logger.warning("[%s] Tag gate failed — blocking.", run_id)
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
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "tag_lint_report": {
                            "rich_text": self.notion.rich_text(
                                lint_report_to_text(lint_report)
                            )
                        }
                    },
                )

            job_id = self._ledger.start_job(
                attachment_key, pdf_sha256, EXTRACTION_VERSION
            )
            logger.info("[%s] JobLedger job_id=%d", run_id, job_id)

            # ────────────────────────────────────────────────────────────────────────────
            # STAGE 1 — EXTRACT
            # ────────────────────────────────────────────────────────────────────────────
            markdown_text = self._load_markdown_for_paper(
                attachment_key, local_pdf, run_id
            )
            local_md.write_text(markdown_text, encoding="utf-8")
            self._ledger.update_status(job_id, "marker_done")

            logger.info("[%s] Stage 1: extracting knowledge via OpenAI ...", run_id)
            extraction = self._run_stage_extract(markdown_text, hubs, run_id)
            self._ledger.update_status(job_id, "extract_done")

            # Patch Paper Tracker
            self._patch_notion_page(page_id, extraction)

            # Create Knowledge Inbox pages; collect {concept_index: ki_page_id}.
            concepts = extraction.extracted_concepts
            logger.info(
                "[%s] Creating %d Knowledge Inbox page(s) ...", run_id, len(concepts)
            )
            ki_page_ids: list[str] = []
            for concept in concepts:
                try:
                    ki_page_id = self._create_knowledge_item(page_id, concept, hubs)
                    ki_page_ids.append(ki_page_id)
                except Exception:
                    logger.exception(
                        "[%s] Failed to create knowledge item %r", run_id, concept.title
                    )
                    ki_page_ids.append("")  # placeholder to keep index alignment

            # ────────────────────────────────────────────────────────────────────────────
            # STAGE 2 — RETRIEVE
            # ────────────────────────────────────────────────────────────────────────────
            logger.info(
                "[%s] Stage 2: retrieving candidates from Second Brain (K=%d) ...",
                run_id,
                RETRIEVE_CANDIDATES_K,
            )
            all_candidates: list[list[dict]] = []
            for i, concept in enumerate(concepts):
                candidates = self._retrieve_candidates_for_concept(
                    concept, sb_index, k=RETRIEVE_CANDIDATES_K
                )
                all_candidates.append(candidates)

                ki_pid = ki_page_ids[i]
                if not ki_pid:
                    continue
                try:
                    candidates_json = json.dumps(
                        [
                            {
                                "id": c["id"],
                                "title": c["title"],
                                "hub": c.get("hub", ""),
                                "summary": c.get("summary", ""),
                            }
                            for c in candidates
                        ],
                        ensure_ascii=False,
                    )
                    self.notion.update_page(
                        page_id=ki_pid,
                        properties={
                            "candidate_matches": {
                                "rich_text": self.notion.rich_text(
                                    candidates_json[:NOTION_BLOCK_MAX_CHARS]
                                )
                            }
                        },
                    )
                except Exception:
                    logger.exception(
                        "[%s] Failed to persist candidates for page %s", run_id, ki_pid
                    )

            self._ledger.update_status(job_id, "retrieve_done")

            # ────────────────────────────────────────────────────────────────────────────
            # STAGE 3 — LINK
            # ────────────────────────────────────────────────────────────────────────────
            logger.info("[%s] Stage 3: generating graph edges via OpenAI ...", run_id)
            for i, concept in enumerate(concepts):
                ki_pid = ki_page_ids[i]
                if not ki_pid:
                    continue
                candidates = all_candidates[i]
                if not candidates:
                    # Nothing to link; just set status.
                    try:
                        self.notion.update_page(
                            page_id=ki_pid,
                            properties={
                                "edge_suggestions": {
                                    "rich_text": self.notion.rich_text("{}")
                                },
                                "graph_link_status": self.notion.select_prop(
                                    "linked-ai"
                                ),
                            },
                        )
                    except Exception:
                        logger.exception(
                            "[%s] Failed to set empty edges for %s", run_id, ki_pid
                        )
                    continue

                try:
                    link_result = self._run_stage_link(concept, candidates, run_id)
                    self._update_knowledge_item_graph_data(ki_pid, link_result)
                except Exception:
                    logger.exception(
                        "[%s] Linking failed for concept %r", run_id, concept.title
                    )

            self._ledger.update_status(job_id, "link_done")

            self._ledger.update_status(job_id, "notion_done")
            self._ledger.finish_job(job_id)
            logger.info("[%s] Done.", run_id)

        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", run_id, exc)
            if job_id is not None:
                self._ledger.update_status(job_id, "failed", error=str(exc))
            raise

        finally:
            for tmp_file in filter(None, [local_pdf, local_md]):
                if tmp_file is not None and tmp_file.exists():
                    tmp_file.unlink()

    # ── Koofr helpers ───────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _koofr_exists(self, remote_path: str) -> bool:
        try:
            return self._webdav.check(remote_path)
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _download_koofr(self, remote_path: str, local_path: Path) -> None:
        self._webdav.download_sync(remote_path=remote_path, local_path=str(local_path))

    @staticmethod
    def _extract_pdf_from_zip(
        zip_path: Path, output_path: Path, preferred: str | None = None
    ) -> None:
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
                    output_path.write_bytes(zf.read(match.filename))
                    return
                logger.warning(
                    "primary_pdf_filename %r not found in zip; using largest PDF.",
                    preferred,
                )

            largest = max(pdf_entries, key=lambda e: e.file_size)
            output_path.write_bytes(zf.read(largest.filename))

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── Marker API ───────────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _pdf_to_markdown(self, pdf_path: Path) -> str:
        """POST the PDF to marker-api; return the Markdown string."""
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

    # ── Stage 1: Extract ───────────────────────────────────────────────────────────────────────────

    def _run_stage_extract(
        self, markdown: str, hubs: dict[str, str], run_id: str
    ) -> ExtractionResult:
        """
        Stage 1: call OpenAI extraction, validate with Pydantic, attempt repair,
        run LaTeX sanity check.
        """
        raw = self._call_openai_extract(markdown, hubs)
        result, errors = validate_extraction(raw)

        if errors:
            logger.warning(
                "[%s] Stage1 validation failed (%d error(s)) — attempting repair.",
                run_id, len(errors),
            )
            error_summary = "; ".join(errors[:MAX_REPAIR_ERRORS])
            if len(errors) > MAX_REPAIR_ERRORS:
                error_summary += (
                    f" … (showing {MAX_REPAIR_ERRORS} of {len(errors)} errors)"
                )
            raw2 = self._call_openai_repair(raw, error_summary)
            result, errors2 = validate_extraction(raw2)
            if errors2:
                logger.error(
                    "[%s] Stage1 repair also failed — flagging concepts with confidence=0.",
                    run_id,
                )
                for concept in result.extracted_concepts:
                    concept.confidence = 0.0

        for concept in result.extracted_concepts:
            issues = latex_sanity_check(concept.statement_latex)
            if issues:
                logger.warning(
                    "[%s] LaTeX issues in %r: %s", run_id, concept.title, issues
                )
                concept.confidence = min(concept.confidence, 0.5)

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _call_openai_extract(
        self, markdown: str, hubs: dict[str, str]
    ) -> dict[str, Any]:
        """Send the paper Markdown to GPT and return the parsed JSON dict."""
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = EXTRACTION_SYSTEM_PROMPT_V2.replace(
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
                        + markdown[:100_000]
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    def _call_openai_repair(
        self, invalid_output: dict[str, Any], error_summary: str
    ) -> dict[str, Any]:
        """Send invalid output back for repair."""
        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON repair assistant. Fix the JSON to match "
                        "the required schema. Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Errors:\n{error_summary}\n\n"
                        f"Invalid JSON:\n{json.dumps(invalid_output, indent=2)}\n\n"
                        "Return the corrected JSON."
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    # ── Stage 2: Retrieve ─────────────────────────────────────────────────────────────────────────

    def _build_second_brain_index(self) -> list[dict]:
        """
        Fetch all "Atomic Concept" pages from the Second Brain DB and build an
        in-memory search index (list of lightweight records).

        Each record: {id, title, hub, summary, tags, keywords_bag}
        keywords_bag is the lowercased token set used for scoring.

        Called once per run() invocation.
        """
        logger.debug("Building Second Brain concept index ...")
        try:
            pages = self.notion.query_database(
                self.second_brain_db,
                filter={
                    "property": "Note Level",
                    "select": {"equals": "Atomic Concept"},
                },
            )
        except Exception:
            logger.exception("Failed to query Second Brain for Atomic Concepts; index empty.")
            return []

        index: list[dict] = []
        for page in pages:
            page_id = page["id"]
            props = page.get("properties", {})
            title = self._get_page_title(page) or ""

            # Try to read a brief summary or one-liner property if it exists.
            summary = (
                self._get_text_prop(props, "One Liner")
                or self._get_text_prop(props, "Summary")
                or ""
            )
            # Hub
            hub = ""
            hub_prop = props.get("Hub") or props.get("Parent Hub") or {}
            if hub_prop.get("type") == "select":
                hub = (hub_prop.get("select") or {}).get("name", "")
            elif hub_prop.get("type") == "relation":
                # Hub stored as relation: just leave empty (we can't easily dereference here)
                hub = ""

            # Tags
            tags: list[str] = self._get_multi_select_prop(props, "Tags")

            # Build keyword bag: title tokens + summary tokens + tags
            bag = set(
                t.lower()
                for token in (title + " " + summary + " " + " ".join(tags)).split()
                for t in [re.sub(r"[^a-z0-9]", "", token.lower())]
                if t
            )

            index.append(
                {
                    "id": page_id,
                    "title": title,
                    "hub": hub,
                    "summary": summary,
                    "tags": tags,
                    "keywords_bag": bag,
                }
            )

        logger.debug("Second Brain index: %d atomic concepts loaded.", len(index))
        return index

    def _retrieve_candidates_for_concept(
        self, concept: MathObject, sb_index: list[dict], k: int = RETRIEVE_CANDIDATES_K
    ) -> list[dict]:
        """
        Deterministic retrieval of top-K candidate concepts from ``sb_index``
        using token-overlap scoring.

        Scoring dimensions
        ------------------
        1. Token overlap between concept keywords + title and the candidate's
           keywords_bag.
        2. Hub match bonus (+0.5 if hubs are identical).
        3. Simple IDF-style weighting: rare tokens score higher.

        Does NOT call any external service.
        """
        if not sb_index:
            return []

        # Build query bag from canonical + prereq + downstream keywords + title.
        query_tokens: list[str] = []
        for phrase in (
            concept.canonical_keywords
            + concept.prereq_keywords
            + concept.downstream_keywords
            + [concept.title]
        ):
            for tok in phrase.lower().split():
                clean = re.sub(r"[^a-z0-9]", "", tok)
                if clean:
                    query_tokens.append(clean)

        query_bag = set(query_tokens)
        if not query_bag:
            return []

        # IDF: count how many index docs contain each token.
        doc_freq: dict[str, int] = {}
        for rec in sb_index:
            for tok in rec["keywords_bag"]:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        N = len(sb_index)

        def score(rec: dict) -> float:
            bag = rec["keywords_bag"]
            common = query_bag & bag
            if not common:
                return 0.0
            # TF-IDF-style: sum of log(N/df) for common tokens
            tfidf = sum(
                math.log1p(N / doc_freq.get(t, 1)) for t in common
            )
            # Normalise by query bag size
            overlap_score = tfidf / (len(query_bag) + 1e-9)
            # Hub bonus
            hub_bonus = 0.5 if (rec["hub"] and rec["hub"] == concept.suggested_hub) else 0.0
            return overlap_score + hub_bonus

        scored = [(rec, score(rec)) for rec in sb_index]
        scored.sort(key=lambda x: x[1], reverse=True)
        # Keep only candidates with positive score
        top_k = [
            rec for rec, s in scored[:k] if s > 0.0
        ]
        return top_k

    # ── Stage 3: Link ────────────────────────────────────────────────────────────────────────────

    def _run_stage_link(
        self,
        concept: MathObject,
        candidates: list[dict],
        run_id: str,
    ) -> ConceptLinkResult:
        """
        Stage 3: call OpenAI linking prompt, validate with Pydantic,
        attempt one repair pass on failure.
        """
        raw = self._call_openai_link(concept, candidates)
        link_result, errors = validate_link_result(raw)

        if errors:
            logger.warning(
                "[%s] Stage3 validation failed for %r (%d errors) — repairing.",
                run_id, concept.title, len(errors),
            )
            error_summary = "; ".join(errors[:MAX_REPAIR_ERRORS])
            raw2 = self._call_openai_repair(raw, error_summary)
            link_result, errors2 = validate_link_result(raw2)
            if errors2:
                logger.error(
                    "[%s] Stage3 repair failed for %r — using empty edges.",
                    run_id, concept.title,
                )

        return link_result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _call_openai_link(
        self, concept: MathObject, candidates: list[dict]
    ) -> dict[str, Any]:
        """Call the linking prompt and return the parsed JSON dict."""
        # Concept summary for the LLM.
        concept_summary = (
            f"Title: {concept.title}\n"
            f"Type: {concept.type}\n"
            f"Hub: {concept.suggested_hub}\n"
            f"Statement: {concept.statement_latex[:500]}\n"
            f"Canonical keywords: {', '.join(concept.canonical_keywords[:10])}\n"
            f"Prereq keywords: {', '.join(concept.prereq_keywords[:10])}\n"
            f"Downstream keywords: {', '.join(concept.downstream_keywords[:10])}"
        )

        # Candidate list (compact).
        candidates_text = json.dumps(
            [
                {
                    "target_concept_id": c["id"],
                    "target_title": c["title"],
                    "hub": c.get("hub", ""),
                    "summary": c.get("summary", "")[:200],
                }
                for c in candidates
            ],
            ensure_ascii=False,
            indent=2,
        )

        caps_text = "\n".join(
            f"  {k} ≤ {v}" for k, v in EDGE_CAPS.items()
        )

        user_content = (
            "NEWLY EXTRACTED CONCEPT:\n"
            f"{concept_summary}\n\n"
            "CANDIDATE CONCEPTS (link ONLY to concepts in this list):\n"
            f"{candidates_text}\n\n"
            f"HARD CAPS PER EDGE TYPE:\n{caps_text}\n\n"
            "Produce the edge_suggestions JSON."
        )

        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LINKING_SYSTEM_PROMPT_V1},
                {"role": "user", "content": user_content},
            ],
        )
        return json.loads(response.choices[0].message.content)

    # ── Notion write helpers ──────────────────────────────────────────────────────────────────

    def _patch_notion_page(self, page_id: str, result: ExtractionResult) -> None:
        """Update Paper Tracker after a successful Stage 1 extraction."""
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.status_prop("s2-extracted"),
                "AI Status": self.notion.select_prop("Unverified-AI"),
                "One Liner": {"rich_text": self.notion.rich_text(result.one_liner)},
                "Active Themes": self.notion.multi_select_prop(result.active_themes),
            },
        )

    def _create_knowledge_item(
        self,
        paper_page_id: str,
        concept: MathObject,
        hubs: dict[str, str],
    ) -> str:
        """
        Materialise a single MathObject as a Knowledge Inbox Notion page.

        Returns the new page's Notion ID.
        """
        kind = concept.type
        title = concept.title

        suggested_hub = getattr(concept, "suggested_hub", "")
        hub_suggestion_text = (
            json.dumps({"suggested_hub": suggested_hub}, ensure_ascii=False)
            if suggested_hub
            else ""
        )

        source_pages_str = (
            ", ".join(str(p) for p in concept.source_pages)
            if concept.source_pages
            else ""
        )

        # Keywords stored as comma-separated rich_text.
        canonical_kw_str = ", ".join(concept.canonical_keywords)
        prereq_kw_str = ", ".join(concept.prereq_keywords)
        downstream_kw_str = ", ".join(concept.downstream_keywords)

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
        if hub_suggestion_text:
            properties["Hub Suggestions"] = {
                "rich_text": self.notion.rich_text(hub_suggestion_text)
            }
        if canonical_kw_str:
            properties["canonical_keywords"] = {
                "rich_text": self.notion.rich_text(canonical_kw_str[:NOTION_BLOCK_MAX_CHARS])
            }
        if prereq_kw_str:
            properties["prereq_keywords"] = {
                "rich_text": self.notion.rich_text(prereq_kw_str[:NOTION_BLOCK_MAX_CHARS])
            }
        if downstream_kw_str:
            properties["downstream_keywords"] = {
                "rich_text": self.notion.rich_text(
                    downstream_kw_str[:NOTION_BLOCK_MAX_CHARS]
                )
            }

        new_page = self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties=properties,
        )
        new_page_id: str = new_page["id"]
        logger.debug(
            "Created Knowledge Inbox page %s for concept %r.", new_page_id, title
        )

        # Page body.
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

        self._append_blocks_in_batches(new_page_id, body_blocks)
        return new_page_id

    def _update_knowledge_item_graph_data(
        self,
        ki_page_id: str,
        link_result: ConceptLinkResult,
    ) -> None:
        """Write edge_suggestions JSON and set graph_link_status on a KI page."""
        edge_dict: dict = {
            "depends_on": [e.model_dump() for e in link_result.depends_on],
            "enables": [e.model_dump() for e in link_result.enables],
            "generalizes": [e.model_dump() for e in link_result.generalizes],
            "special_case_of": [e.model_dump() for e in link_result.special_case_of],
            "related": [e.model_dump() for e in link_result.related],
        }
        edge_json = json.dumps(edge_dict, ensure_ascii=False)
        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "edge_suggestions": {
                    "rich_text": self.notion.rich_text(
                        edge_json[:NOTION_BLOCK_MAX_CHARS]
                    )
                },
                "graph_link_status": self.notion.select_prop("linked-ai"),
            },
        )

    # ── Text chunking ───────────────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_len: int = NOTION_BLOCK_MAX_CHARS) -> list[str]:
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
            remaining = remaining[split_pos + 1 :]

        return [c for c in chunks if c.strip()]

    # ── Notion block builders ───────────────────────────────────────────────────────────────

    def _paragraph_blocks(self, text: str) -> list[dict]:
        return [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": chunk}}]
                },
            }
            for chunk in self._chunk_text(text)
        ]

    @staticmethod
    def _heading_block(text: str) -> dict:
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    def _append_blocks_in_batches(self, page_id: str, blocks: list[dict]) -> None:
        for i in range(0, len(blocks), NOTION_BLOCKS_PER_REQUEST):
            batch = blocks[i : i + NOTION_BLOCKS_PER_REQUEST]
            self.notion.append_block_children(block_id=page_id, children=batch)

    # ── Property / page title helpers ──────────────────────────────────────────────────────

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
        prop_type = prop.get("type")
        if prop_type == "url":
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

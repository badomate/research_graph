"""
modules/ingestion.py — Module 1: Core Ingestion Engine
───────────────────────────────────────────────────────
Pipeline for each paper in the 'Paper Tracker' Notion DB with Status == 's0-inbox':

  1. Download the PDF from Koofr via WebDAV.
  2. Send the PDF to the local marker-api container (PDF → Markdown + LaTeX).
  3. Dynamically fetch "Hub" pages from the Second Brain DB.
  4. Send the Markdown + Hub list to GPT-4o (strict JSON mode) to extract
     structured metadata and mathematical concepts.
  5. Patch the Paper Tracker Notion row to status 's1-skim'.
  6. For each extracted concept, create a page in the "Knowledge Inbox" DB
     with metadata-only properties, then append the heavy LaTeX content and
     assumptions as page-body blocks (respecting Notion's 2000-char limit).

Design constraints honoured:
  - Notion text blocks are hard-capped at 1900 chars (safe margin below 2000).
  - 'content' and 'assumptions' are NEVER stored as Notion page properties.
  - Hub suggestions are validated against live Second Brain data every run.
  - OpenAI is called with response_format={"type": "json_object"} for
    guaranteed valid JSON — no fence-stripping hacks required.
  - Notion's 100-block-per-append limit is respected via batched writes.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import openai
import requests
from webdav3.client import Client as WebDAVClient

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

# ── Scratch directory inside the Docker volume ────────────────────────────────
TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))

# ── OpenAI model ──────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o"

# ── Notion hard limits ────────────────────────────────────────────────────────
# Notion's API rejects any rich_text content exceeding 2000 characters.
# We use 1900 as our safe ceiling to avoid off-by-one edge cases.
NOTION_BLOCK_MAX_CHARS = 1900
# Notion allows at most 100 child blocks per append_block_children call.
NOTION_BLOCKS_PER_REQUEST = 100

# ── System prompt template ────────────────────────────────────────────────────
# [INJECT_DYNAMIC_HUBS_HERE] is replaced at runtime with the live hub list
# fetched from the Second Brain DB before every OpenAI call.
EXTRACTION_SYSTEM_PROMPT = """\
You are a highly rigorous researcher in applied mathematics. Process the Markdown paper and extract strictly factual mathematical structures.
1. Extract exact mathematical formulations. Do not paraphrase.
2. Format variables and equations in valid LaTeX ($ for inline, $$ for display).
3. Explicitly extract boundary conditions or assumptions. If none, write "None explicitly stated."
4. Assign a `suggested_hub` from the ALLOWED_HUBS list. Do not invent hubs. If none fit, use "Uncategorized".

ALLOWED_HUBS: [INJECT_DYNAMIC_HUBS_HERE]

You must respond in valid JSON matching this exact schema:
{
  "one_liner": "string",
  "active_themes": ["string"],
  "extracted_concepts": [
    {
      "type": "Definition | Theorem | Assumption | Lemma | Algorithm | Proof",
      "name": "string",
      "content": "string (valid LaTeX)",
      "assumptions": "string",
      "suggested_hub": "string (must match ALLOWED_HUBS)"
    }
  ]
}
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
        self.koofr_base = os.environ.get("KOOFR_PDF_PATH", "/zotero")

    # ── WebDAV client factory ─────────────────────────────────────────────────

    @staticmethod
    def _build_webdav_client() -> WebDAVClient:
        options = {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": os.environ["KOOFR_USER"],
            "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
        }
        return WebDAVClient(options)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Poll 'Paper Tracker' for s0-inbox papers and run the full ingestion
        pipeline on each one.

        Hubs are fetched once per run() invocation so that:
          (a) Every paper in the same batch uses a consistent hub snapshot.
          (b) We avoid a redundant Notion API call for every single concept.
        """
        logger.info("Ingestion: polling for s0-inbox papers …")
        pages = self.notion.query_database(
            self.paper_tracker_db,
            filter={
                "property": "Status",
                "status": {"equals": "s0-inbox"},
            },
        )
        logger.info("Ingestion: found %d paper(s) to process.", len(pages))

        if not pages:
            return

        # Fetch the live hub registry once for all papers in this batch.
        hubs: dict[str, str] = self._fetch_allowed_hubs()
        logger.info(
            "Ingestion: loaded %d hub(s) from Second Brain: %s",
            len(hubs),
            list(hubs.keys()),
        )

        for page in pages:
            try:
                self._process_paper(page, hubs)
            except Exception:
                logger.exception("Failed to process page %s", page["id"])

    # ── Hub fetching ──────────────────────────────────────────────────────────

    def _fetch_allowed_hubs(self) -> dict[str, str]:
        """
        Query the Second Brain DB for all pages where the 'Note Level' Select
        property equals 'Hub'.

        Returns:
            dict mapping hub name (str) → Notion page ID (str).

        This dict serves two purposes:
          1. Injecting hub names into the OpenAI system prompt so the model
             can pick from real, user-defined categories.
          2. Resolving a concept's suggested_hub string back to a page ID
             for the 'Parent Hub' Relation property.
        """
        logger.debug("Ingestion: fetching Hub pages from Second Brain …")
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

    # ── Per-paper pipeline ────────────────────────────────────────────────────

    def _process_paper(self, page: dict, hubs: dict[str, str]) -> None:
        """
        Full ingestion pipeline for a single paper page:
          1. Download PDF from Koofr via WebDAV.
          2. Convert PDF → Markdown via marker-api.
          3. Extract structured concepts via OpenAI GPT-4o (strict JSON mode).
          4. Patch the Paper Tracker Notion row with summary metadata.
          5. Create a Knowledge Inbox page for every extracted concept.
        """
        page_id = page["id"]
        props = page["properties"]

        # Prefer an explicit "File Name" property; fall back to Title + ".pdf".
        pdf_filename = self._get_text_prop(props, "File Name") or (
            self._get_title_prop(props) + ".pdf"
        )
        remote_path = f"{self.koofr_base}/{pdf_filename}"

        # Short unique ID used for log correlation and temp-file naming.
        job_id = uuid.uuid4().hex[:8]
        local_pdf = TMP_DIR / f"{job_id}.pdf"
        local_md = TMP_DIR / f"{job_id}.md"

        try:
            # ── Step 1: Download PDF from Koofr ───────────────────────────────
            logger.info("[%s] Downloading %s …", job_id, remote_path)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            self._webdav.download_sync(
                remote_path=remote_path, local_path=str(local_pdf)
            )

            # ── Step 2: Convert PDF → Markdown via marker-api ─────────────────
            logger.info("[%s] Converting PDF to Markdown …", job_id)
            markdown_text = self._pdf_to_markdown(local_pdf)
            local_md.write_text(markdown_text, encoding="utf-8")

            # ── Step 3: Extract structured knowledge via OpenAI ───────────────
            logger.info("[%s] Extracting knowledge via OpenAI GPT-4o …", job_id)
            extracted = self._extract_knowledge(markdown_text, hubs)

            # ── Step 4: Patch the Paper Tracker Notion row ────────────────────
            logger.info("[%s] Patching Notion paper row …", job_id)
            self._patch_notion_page(page_id, extracted)

            # ── Step 5: Create Knowledge Inbox entries ────────────────────────
            concepts = extracted.get("extracted_concepts", [])
            logger.info(
                "[%s] Creating %d Knowledge Inbox page(s) …",
                job_id,
                len(concepts),
            )
            for concept in concepts:
                try:
                    self._create_knowledge_item(page_id, concept, hubs)
                except Exception:
                    logger.exception(
                        "[%s] Failed to create knowledge item '%s'",
                        job_id,
                        concept.get("name", "?"),
                    )

            logger.info("[%s] Done.", job_id)

        finally:
            # Always clean up temp files, even on failure.
            for tmp_file in (local_pdf, local_md):
                if tmp_file.exists():
                    tmp_file.unlink()

    # ── Step 2: Marker API ────────────────────────────────────────────────────

    def _pdf_to_markdown(self, pdf_path: Path) -> str:
        """
        Ask the local marker-api container to convert the PDF and return
        the resulting Markdown string.

        Protocol (marker_server from marker-pdf PyPI package):
          POST /marker
          Content-Type: application/json
          Body: {"filepath": "<absolute path accessible inside the container>"}

        Both the orchestrator and marker-api mount tmp_storage at
        /tmp/pipeline, so the same path is valid in both containers.
        Response shape: {"markdown": "...", "metadata": {...}, ...}
        """
        response = requests.post(
            f"{self.marker_url}/marker",
            json={"filepath": str(pdf_path)},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        # marker_server wraps the rendered output; for markdown format the
        # key is "markdown". Fall back to "output" / "text" defensively.
        return data.get("markdown") or data.get("output") or data.get("text") or response.text

    # ── Step 3: OpenAI extraction ─────────────────────────────────────────────

    def _extract_knowledge(
        self, markdown: str, hubs: dict[str, str]
    ) -> dict[str, Any]:
        """
        Send the paper Markdown to GPT-4o and return the parsed JSON dict.

        Key design decisions:
          - response_format={"type": "json_object"} enforces valid JSON output
            at the API level — no fence-stripping or error-prone post-processing.
          - Hub names are injected into the system prompt at call time so the
            model chooses from real, user-defined categories.
          - Input is truncated to ~100k chars (~25k tokens) to stay comfortably
            within GPT-4o's context window while leaving space for the response.
        """
        # Build a JSON-array-style string of hub names for the prompt.
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs)
            if hubs
            else '"Uncategorized"'
        )
        # Inject live hub names into the prompt template.
        system_prompt = EXTRACTION_SYSTEM_PROMPT.replace(
            "[INJECT_DYNAMIC_HUBS_HERE]", hub_names_str
        )

        truncated_markdown = markdown[:100_000]

        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=4096,
            response_format={"type": "json_object"},  # guaranteed valid JSON
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

        # With json_object mode the response is always valid JSON — parse directly.
        return json.loads(response.choices[0].message.content)

    # ── Step 4: Patch Paper Tracker row ──────────────────────────────────────

    def _patch_notion_page(self, page_id: str, extracted: dict) -> None:
        """
        Update the Paper Tracker page after a successful extraction run:
          - Status:       s0-inbox  → s1-skim
          - AI Status:    Unverified-AI
          - One Liner:    single-sentence summary from the model
          - Active Themes: multi-select keyword tags
        """
        one_liner = extracted.get("one_liner", "")
        themes = extracted.get("active_themes", [])

        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.select_prop("s1-skim"),
                "AI Status": self.notion.select_prop("Unverified-AI"),
                "One Liner": {"rich_text": self.notion.rich_text(one_liner)},
                "Active Themes": self.notion.multi_select_prop(themes),
            },
        )

    # ── Step 5: Create Knowledge Inbox entry ─────────────────────────────────

    def _create_knowledge_item(
        self,
        paper_page_id: str,
        concept: dict,
        hubs: dict[str, str],
    ) -> None:
        """
        Materialise a single extracted concept as a Knowledge Inbox Notion page.

        Architecture — why properties vs. body blocks:
          Notion database properties have a hard 2000-character limit and are
          not suitable for LaTeX proofs or lengthy assumption paragraphs.
          We therefore separate concerns strictly:
            • PROPERTIES  — lightweight metadata only (name, type, status,
                            relations). Fast to query, safe to index.
            • PAGE BODY   — heavy LaTeX content and assumptions, written as
                            block children after page creation. No size issues.

        Steps:
          1. Build metadata properties dict, including the optional Parent Hub
             relation resolved from the live hub registry.
          2. Create the Notion page (metadata only — no content yet).
          3. Append heading_2 + chunked paragraph blocks for "Assumptions"
             and "Mathematical Formulation" to the new page's body.
        """
        name: str = concept.get("name", "Untitled")
        kind: str = concept.get("type", "Theorem")
        content: str = concept.get("content", "")
        assumptions: str = concept.get("assumptions", "None explicitly stated.")
        suggested_hub: str = concept.get("suggested_hub", "")

        # ── Step 1: Build metadata-only properties ────────────────────────────
        properties: dict = {
            "Name": self.notion.title_prop(f"[{kind}] {name}"),
            "Type": self.notion.select_prop(kind),
            "Status": self.notion.select_prop("Inbox"),
            "Source Paper": self.notion.relation_prop([paper_page_id]),
        }

        # Attempt to resolve the suggested_hub to a live Notion page ID.
        # If the model hallucinated a hub name not in the registry, skip
        # gracefully — never create a broken relation or raise an error.
        hub_page_id = hubs.get(suggested_hub)
        if hub_page_id:
            properties["Parent Hub"] = self.notion.relation_prop([hub_page_id])
        else:
            logger.debug(
                "Hub '%s' not found in registry; skipping Parent Hub relation.",
                suggested_hub,
            )

        # ── Step 2: Create the page (metadata only) ───────────────────────────
        new_page = self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties=properties,
        )
        new_page_id: str = new_page["id"]
        logger.debug(
            "Created Knowledge Inbox page %s for concept '%s'.", new_page_id, name
        )

        # ── Steps 3 & 4: Append body blocks ──────────────────────────────────
        # Build an ordered list of blocks:
        #   heading_2("Assumptions") → paragraph(s) for assumptions text
        #   heading_2("Mathematical Formulation") → paragraph(s) for LaTeX content
        body_blocks: list[dict] = []

        body_blocks.append(self._heading_block("Assumptions"))
        body_blocks.extend(self._paragraph_blocks(assumptions))

        body_blocks.append(self._heading_block("Mathematical Formulation"))
        body_blocks.extend(self._paragraph_blocks(content))

        # Write blocks in batches of 100 to respect the Notion API limit.
        self._append_blocks_in_batches(new_page_id, body_blocks)

    # ── Text chunking ─────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_len: int = NOTION_BLOCK_MAX_CHARS) -> list[str]:
        """
        Split `text` into a list of strings each no longer than `max_len`
        characters, preferring newline split points to preserve LaTeX structure.

        Algorithm:
          1. If the text fits in one chunk, return it as-is.
          2. Otherwise, find the last newline within the allowed window and
             split there, keeping LaTeX display blocks intact.
          3. If no newline exists within the window, fall back to a hard
             character split (rare for well-formatted LaTeX).
          4. Repeat on the remainder until exhausted.

        Args:
            text:    The input string to chunk (will be stripped first).
            max_len: Maximum characters per chunk (default: NOTION_BLOCK_MAX_CHARS).

        Returns:
            A list of non-empty, stripped string chunks.
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

            # Look for the last newline within the allowed window.
            split_pos = remaining.rfind("\n", 0, max_len)

            if split_pos <= 0:
                # No suitable newline — hard-split at the character limit.
                split_pos = max_len
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos:]
            else:
                chunks.append(remaining[:split_pos])
                # Advance past the newline character itself.
                remaining = remaining[split_pos + 1:]

        return [c for c in chunks if c.strip()]

    # ── Notion block builders ─────────────────────────────────────────────────

    def _paragraph_blocks(self, text: str) -> list[dict]:
        """
        Convert a (potentially long) string into a list of Notion paragraph
        block dicts, each within the NOTION_BLOCK_MAX_CHARS safe limit.
        Uses _chunk_text internally.
        """
        return [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": chunk}}
                    ],
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

    def _append_blocks_in_batches(
        self, page_id: str, blocks: list[dict]
    ) -> None:
        """
        Append blocks to a Notion page in batches of NOTION_BLOCKS_PER_REQUEST
        (100) to respect the Notion API's per-call block limit.
        Each batch is a separate API call managed by NotionClientWrapper
        (which handles rate limiting and retries automatically).
        """
        for i in range(0, len(blocks), NOTION_BLOCKS_PER_REQUEST):
            batch = blocks[i : i + NOTION_BLOCKS_PER_REQUEST]
            self.notion.append_block_children(block_id=page_id, children=batch)

    # ── Property / page title helpers ─────────────────────────────────────────

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

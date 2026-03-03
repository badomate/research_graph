"""
modules/ingestion.py — Module 1: Core Ingestion Engine
───────────────────────────────────────────────────────
Poll "Paper Tracker" for rows where Status == 's0-inbox', then:

  1. Download the PDF from Koofr via WebDAV.
  2. Send the PDF to the local marker-api container (PDF → Markdown + LaTeX).
  3. Send the Markdown to Claude 3.5 Sonnet to extract structured metadata.
  4. Patch the Notion row to status 's1-skim'.
  5. Create rows in the "Knowledge Inbox" for every extracted theorem/definition.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import anthropic
import requests
from webdav3.client import Client as WebDAVClient

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

# ── Scratch directory inside the Docker volume ────────────────────────────────
TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))

# ── Claude model ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"


# ── Prompt template ───────────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a precise mathematical knowledge extractor for a PhD researcher in applied mathematics (Mean Field Games, PDEs, stochastic control).

Return ONLY valid JSON. No explanatory text, no markdown fences, no conversational filler.

Output schema:
{
  "one_liner": "<one concise sentence summarising the paper's main contribution>",
  "active_themes": ["<theme1>", "<theme2>", ...],
  "extracted_knowledge": [
    {
      "type": "theorem|definition|lemma|proposition|corollary|remark",
      "label": "<e.g. Theorem 3.2>",
      "content": "<full statement in LaTeX, using $ for inline math and $$ for display math>"
    }
  ]
}

Rules:
- Use $...$ for inline mathematics.
- Use $$...$$ for display mathematics (on its own line).
- active_themes: 3–8 short tags (e.g. "Mean Field Games", "viscosity solutions").
- extracted_knowledge: include every formal mathematical statement you can find.
- Strings must be JSON-safe (escape backslashes: \\\\ not \\).
"""


class IngestionEngine:
    """Module 1: Core Ingestion Engine (Notion → WebDAV → Marker → Claude → Notion)."""

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.claude = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
        self._webdav = self._build_webdav_client()
        self.marker_url = os.environ.get("MARKER_API_URL", "http://marker-api:8080")
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]

    # ── WebDAV ────────────────────────────────────────────────────────────────

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
        """Poll for inbox papers and process each one."""
        logger.info("Ingestion: polling for s0-inbox papers …")
        pages = self.notion.query_database(
            self.paper_tracker_db,
            filter={
                "property": "Status",
                "select": {"equals": "s0-inbox"},
            },
        )
        logger.info("Ingestion: found %d paper(s) to process.", len(pages))
        for page in pages:
            try:
                self._process_paper(page)
            except Exception:
                logger.exception("Failed to process page %s", page["id"])

    # ── Per-paper pipeline ────────────────────────────────────────────────────

    def _process_paper(self, page: dict) -> None:
        page_id = page["id"]
        props = page["properties"]

        # Resolve the PDF filename from a "File Name" or "Title" property
        pdf_filename = self._get_text_prop(props, "File Name") or (
            self._get_title_prop(props) + ".pdf"
        )
        koofr_base = os.environ.get("KOOFR_PDF_PATH", "/Papers")
        remote_path = f"{koofr_base}/{pdf_filename}"

        job_id = uuid.uuid4().hex[:8]
        local_pdf = TMP_DIR / f"{job_id}.pdf"
        local_md = TMP_DIR / f"{job_id}.md"

        try:
            # 1. Download PDF from Koofr
            logger.info("[%s] Downloading %s …", job_id, remote_path)
            self._webdav.download_sync(remote_path=remote_path, local_path=str(local_pdf))

            # 2. Convert PDF → Markdown via marker-api
            logger.info("[%s] Converting PDF to Markdown …", job_id)
            markdown_text = self._pdf_to_markdown(local_pdf)
            local_md.write_text(markdown_text, encoding="utf-8")

            # 3. Extract knowledge with Claude
            logger.info("[%s] Extracting knowledge with Claude …", job_id)
            extracted = self._extract_knowledge(markdown_text)

            # 4. Patch Notion row
            logger.info("[%s] Updating Notion page …", job_id)
            self._patch_notion_page(page_id, extracted)

            # 5. Create Knowledge Inbox rows
            logger.info("[%s] Creating %d knowledge item(s) …", job_id, len(extracted["extracted_knowledge"]))
            for item in extracted["extracted_knowledge"]:
                self._create_knowledge_item(page_id, item)

            logger.info("[%s] Done.", job_id)

        finally:
            # Purge temp files regardless of success/failure
            for f in (local_pdf, local_md):
                if f.exists():
                    f.unlink()

    # ── Step 2: Marker API ────────────────────────────────────────────────────

    def _pdf_to_markdown(self, pdf_path: Path) -> str:
        """POST the PDF to the marker-api and return the Markdown string."""
        with pdf_path.open("rb") as fh:
            response = requests.post(
                f"{self.marker_url}/convert",
                files={"file": (pdf_path.name, fh, "application/pdf")},
                timeout=300,
            )
        response.raise_for_status()
        data = response.json()
        # marker-api returns {"markdown": "...", ...}
        return data.get("markdown") or data.get("text") or response.text

    # ── Step 3: Claude extraction ─────────────────────────────────────────────

    def _extract_knowledge(self, markdown: str) -> dict[str, Any]:
        """Send the Markdown to Claude and return the parsed JSON."""
        # Truncate to avoid token limits (~100k chars ≈ 25k tokens)
        truncated = markdown[:100_000]

        message = self.claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract structured knowledge from the following paper Markdown.\n\n"
                        f"{truncated}"
                    ),
                }
            ],
        )
        raw = message.content[0].text.strip()

        # Strip accidental markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw)

    # ── Step 4: Patch Notion ──────────────────────────────────────────────────

    def _patch_notion_page(self, page_id: str, extracted: dict) -> None:
        one_liner = extracted.get("one_liner", "")
        themes = extracted.get("active_themes", [])

        self.notion.update_page(
            page_id=page_id,
            properties={
                "Status": self.notion.select_prop("s1-skim"),
                "AI Status": self.notion.select_prop("Unverified-AI"),
                "One Liner": {
                    "rich_text": self.notion.rich_text(one_liner)
                },
                "Active Themes": self.notion.multi_select_prop(themes),
            },
        )

    # ── Step 5: Knowledge Inbox ───────────────────────────────────────────────

    def _create_knowledge_item(self, paper_page_id: str, item: dict) -> None:
        label = item.get("label", "Untitled")
        content = item.get("content", "")
        kind = item.get("type", "theorem").capitalize()

        self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties={
                "Name": self.notion.title_prop(f"[{kind}] {label}"),
                "Type": self.notion.select_prop(kind),
                "Content": {"rich_text": self.notion.rich_text(content)},
                "Source Paper": self.notion.relation_prop([paper_page_id]),
                "Status": self.notion.select_prop("Inbox"),
            },
        )

    # ── Property helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_text_prop(props: dict, key: str) -> str:
        try:
            return props[key]["rich_text"][0]["plain_text"]
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

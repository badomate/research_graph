"""
modules/arxiv_sniper.py — Module 2: ArXiv Sniper
─────────────────────────────────────────────────
Daily job (06:00) that:

  1. Queries export.arxiv.org for configurable mathematical keywords.
  2. Sends each abstract to ChatGPT (gpt-4o) for a relevance score (1–10).
  3. Creates a Notion "Paper Tracker" row (Status = s0-inbox, Tag = Automated-Radar)
     for every paper scoring >= threshold (default 8).

ArXiv API constraint: **hard 3-second sleep between requests**.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree

import anthropic

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

CLAUDE_FAST_MODEL = os.environ.get("CLAUDE_FAST_MODEL", "claude-sonnet-4-6")

# ── Relevance scoring prompt ──────────────────────────────────────────────────
RELEVANCE_SYSTEM_PROMPT = """You are a research relevance scorer for a PhD student specialising in Mean Field Games, McKean-Vlasov stochastic control, Hamilton-Jacobi-Bellman PDEs, and related applied mathematics.

Given a paper title and abstract, return ONLY valid JSON. No explanatory text, no markdown fences.

Output schema:
{
  "score": <integer 1-10>,
  "justification": "<one or two sentences explaining the score, using LaTeX where appropriate>"
}

Scoring guide:
10 — Core topic (MFG, McKean-Vlasov, master equation, HJB PDE in MFG context).
8-9 — Highly related (related stochastic control, viscosity solutions for MFGs, FBSDE with mean-field).
6-7 — Peripherally related (general stochastic control, generic PDEs).
1-5 — Unrelated.
"""


class ArXivSniper:
    """Module 2: Daily ArXiv search and relevance scoring."""

    ARXIV_API_URL = "https://export.arxiv.org/api/query"
    ARXIV_DELAY_SECONDS = 3  # hard constraint per ArXiv ToS

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]

        raw_keywords = os.environ.get(
            "ARXIV_KEYWORDS",
            "Mean Field Games,Master Equation",
        )
        self.keywords: list[str] = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        self.threshold: int = int(os.environ.get("ARXIV_RELEVANCE_THRESHOLD", "8"))

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Search ArXiv for each keyword and process new papers."""
        logger.info("ArXiv Sniper: starting daily run …")
        seen_ids: set[str] = set()  # deduplicate across keyword queries

        for keyword in self.keywords:
            logger.info("ArXiv Sniper: querying '%s' …", keyword)
            entries = self._fetch_arxiv(keyword)
            logger.info("ArXiv Sniper: got %d entries for '%s'.", len(entries), keyword)

            for entry in entries:
                arxiv_id = entry["id"]
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                try:
                    self._evaluate_and_ingest(entry)
                except Exception:
                    logger.exception("Failed processing ArXiv entry %s", arxiv_id)

                # Hard throttle between requests (ArXiv requirement)
                time.sleep(self.ARXIV_DELAY_SECONDS)

        logger.info("ArXiv Sniper: done — evaluated %d unique entries.", len(seen_ids))

    # ── ArXiv fetch ───────────────────────────────────────────────────────────

    def _fetch_arxiv(self, keyword: str, max_results: int = 20) -> list[dict]:
        """Query the ArXiv Atom feed and return a list of entry dicts."""
        params = urllib.parse.urlencode(
            {
                "search_query": f'all:"{keyword}"',
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"{self.ARXIV_API_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read()
        except Exception:
            logger.exception("ArXiv request failed for keyword '%s'", keyword)
            return []

        return self._parse_atom(content)

    @staticmethod
    def _parse_atom(content: bytes) -> list[dict]:
        """Parse ArXiv Atom XML and return simplified entry dicts."""
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ElementTree.fromstring(content)
        entries = []

        for entry in root.findall("atom:entry", ns):
            arxiv_id_el = entry.find("atom:id", ns)
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)

            if arxiv_id_el is None or title_el is None or summary_el is None:
                continue

            # Normalise whitespace
            title = " ".join((title_el.text or "").split())
            abstract = " ".join((summary_el.text or "").split())
            arxiv_url = (arxiv_id_el.text or "").strip()

            entries.append(
                {
                    "id": arxiv_url,
                    "title": title,
                    "abstract": abstract,
                    "url": arxiv_url,
                }
            )

        return entries

    # ── Evaluate + ingest ─────────────────────────────────────────────────────

    def _evaluate_and_ingest(self, entry: dict) -> None:
        score, justification = self._score_relevance(entry["title"], entry["abstract"])
        logger.info(
            "ArXiv: '%s' → score %d/10", entry["title"][:60], score
        )

        if score < self.threshold:
            return

        logger.info("ArXiv: score >= %d — creating Notion row …", self.threshold)
        self._create_notion_row(entry, score, justification)

    # ── OpenAI scoring ────────────────────────────────────────────────────────

    def _score_relevance(self, title: str, abstract: str) -> tuple[int, str]:
        """Ask Claude to score relevance; return (score, justification)."""
        response = self.claude.messages.create(
            model=CLAUDE_FAST_MODEL,
            max_tokens=256,
            system=RELEVANCE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Title: {title}\n\nAbstract: {abstract}",
                },
            ],
        )
        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        return int(data["score"]), str(data.get("justification", ""))

    # ── Notion row creation ───────────────────────────────────────────────────

    def _create_notion_row(
        self, entry: dict, score: int, justification: str
    ) -> None:
        title = entry["title"]
        url = entry["url"]
        note = f"ArXiv score: {score}/10 — {justification}\n\nURL: {url}"

        self.notion.create_page(
            parent={"database_id": self.paper_tracker_db},
            properties={
                "Name": self.notion.title_prop(title),
                "Status": self.notion.select_prop("s0-inbox"),
                "Tags": self.notion.multi_select_prop(["Automated-Radar"]),
                "ArXiv URL": {"url": url},
                "AI Notes": {
                    "rich_text": self.notion.rich_text(note)
                },
            },
        )

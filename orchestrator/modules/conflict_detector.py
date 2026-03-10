"""
modules/conflict_detector.py — Module 3: Assumption Conflict Detector
───────────────────────────────────────────────────────────────────────
Polls the "Knowledge Inbox" for theorems that have been manually approved
to the "Second Brain" DB.  For each newly approved theorem:

  1. Fetch all existing theorems in the Second Brain linked to the same topic.
  2. Send the new theorem + existing theorems to ChatGPT to detect mathematical
     contradictions or relaxations of assumptions.
  3. If a conflict is found, append a warning callout block to the Notion page.
"""

from __future__ import annotations

import json
import logging
import os

import anthropic

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

CLAUDE_FAST_MODEL = os.environ.get("CLAUDE_FAST_MODEL", "claude-haiku-3-5")

# ── Conflict detection prompt ─────────────────────────────────────────────────
CONFLICT_SYSTEM_PROMPT = """You are a rigorous mathematical consistency checker for a PhD researcher in applied mathematics.

Given a NEW theorem/definition and a list of EXISTING theorems/definitions, detect:
  - Direct contradictions (incompatible conclusions under the same assumptions).
  - Differences in assumptions that change the validity of results (e.g. $N \\to \\infty$ limit, boundary conditions, Lipschitz vs. monotone assumptions).
  - Generalisations or relaxations of assumptions.

Return ONLY valid JSON. No explanatory text, no markdown fences.

Output schema:
{
  "conflict_found": true | false,
  "conflict_type": "contradiction | relaxation | generalisation | none",
  "conflicting_item_label": "<label of the conflicting existing item, or null>",
  "explanation": "<detailed mathematical explanation using LaTeX where needed. $ for inline, $$ for display.>"
}
"""


class ConflictDetector:
    """Module 3: Detects assumption conflicts between approved theorems."""

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db = os.environ["NOTION_SECOND_BRAIN_DB_ID"]

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Poll for newly approved theorems and check for conflicts."""
        logger.info("Conflict Detector: polling for newly approved items …")

        # Query Second Brain for items that have not yet been conflict-checked
        items = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Conflict Checked",
                "checkbox": {"equals": False},
            },
        )
        logger.info("Conflict Detector: found %d unchecked item(s).", len(items))

        for item in items:
            try:
                self._check_item(item)
            except Exception:
                logger.exception("Conflict check failed for %s", item["id"])

    # ── Per-item check ────────────────────────────────────────────────────────

    def _check_item(self, item: dict) -> None:
        page_id = item["id"]
        props = item["properties"]

        new_label = self._get_title(props)
        new_content = self._get_text(props, "Content")
        topic = self._get_select(props, "Topic")

        logger.info("Conflict Detector: checking '%s' (topic: %s) …", new_label, topic)

        # Fetch existing theorems on the same topic (excluding the current item)
        existing = self._fetch_existing(topic, exclude_id=page_id)

        if not existing:
            logger.info("Conflict Detector: no existing items for topic '%s'.", topic)
            self._mark_checked(page_id)
            return

        result = self._detect_conflict(new_label, new_content, existing)

        if result["conflict_found"]:
            logger.warning(
                "Conflict Detector: CONFLICT found for '%s': %s",
                new_label,
                result["conflict_type"],
            )
            self._append_warning(page_id, result)
        else:
            logger.info("Conflict Detector: no conflict for '%s'.", new_label)

        self._mark_checked(page_id)

    # ── Fetch existing theorems ───────────────────────────────────────────────

    def _fetch_existing(self, topic: str, exclude_id: str) -> list[dict]:
        """Return all Second Brain items with the same topic, excluding exclude_id."""
        if not topic:
            return []

        pages = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Topic",
                "select": {"equals": topic},
            },
        )
        result = []
        for p in pages:
            if p["id"] == exclude_id:
                continue
            label = self._get_title(p["properties"])
            content = self._get_text(p["properties"], "Content")
            result.append({"label": label, "content": content})
        return result

    # ── OpenAI conflict detection ─────────────────────────────────────────────

    def _detect_conflict(
        self,
        new_label: str,
        new_content: str,
        existing: list[dict],
    ) -> dict:
        existing_text = "\n\n".join(
            f"[{e['label']}]\n{e['content']}" for e in existing
        )
        user_message = (
            f"NEW ITEM:\n[{new_label}]\n{new_content}\n\n"
            f"EXISTING ITEMS:\n{existing_text}"
        )
        response = self.claude.messages.create(
            model=CLAUDE_FAST_MODEL,
            max_tokens=1024,
            system=CONFLICT_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    # ── Append warning callout ────────────────────────────────────────────────

    def _append_warning(self, page_id: str, result: dict) -> None:
        conflict_type = result.get("conflict_type", "conflict")
        conflicting = result.get("conflicting_item_label") or "unknown"
        explanation = result.get("explanation", "")

        warning_text = (
            f"⚠️ CONFLICT DETECTED ({conflict_type.upper()})\n"
            f"Conflicting item: {conflicting}\n\n"
            f"{explanation}"
        )

        self.notion.append_block_children(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": warning_text[:2000]},
                            }
                        ],
                        "icon": {"type": "emoji", "emoji": "⚠️"},
                        "color": "red_background",
                    },
                }
            ],
        )

    # ── Mark checked ──────────────────────────────────────────────────────────

    def _mark_checked(self, page_id: str) -> None:
        self.notion.update_page(
            page_id=page_id,
            properties={"Conflict Checked": self.notion.checkbox_prop(True)},
        )

    # ── Property helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_title(props: dict) -> str:
        for value in props.values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return "Untitled"
        return "Untitled"

    @staticmethod
    def _get_text(props: dict, key: str) -> str:
        try:
            return props[key]["rich_text"][0]["plain_text"]
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _get_select(props: dict, key: str) -> str:
        try:
            return props[key]["select"]["name"]
        except (KeyError, TypeError):
            return ""

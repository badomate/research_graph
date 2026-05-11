"""
modules/promotion/zotero_sync.py — Zotero reading note and annotation sync.

Fetches notes and annotations from the Zotero API and appends them to the
Notion paper page (idempotent via callout key markers).
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

import requests

from ..notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)


class ZoteroSync:
    """Syncs Zotero notes and annotations to a Notion paper page."""

    def __init__(
        self,
        notion: NotionClientWrapper,
        zotero_user_id: str,
        zotero_api_key: str,
    ) -> None:
        self.notion = notion
        self.zotero_user_id = zotero_user_id
        self.zotero_api_key = zotero_api_key

    def sync_zotero_notes(self, paper_page_id: str, props: dict) -> None:
        """
        Fetch Zotero notes and annotations and append any new ones to the paper page.

        Idempotent: reads existing blocks to collect already-synced Zotero keys
        (stored in callout block headers) and skips duplicates.
        """
        zotero_key = self._get_zotero_key(props)
        if not zotero_key:
            logger.warning(
                "ZoteroSync: no Zotero key found for paper page %s — skipping.",
                paper_page_id,
            )
            return

        try:
            existing_blocks = self.notion.get_block_children(paper_page_id)
        except Exception:
            existing_blocks = []

        synced_keys: set[str] = set()
        for block in existing_blocks:
            if block.get("type") == "callout":
                rt = block.get("callout", {}).get("rich_text", [])
                text = "".join(seg.get("plain_text", "") for seg in rt)
                m = re.search(r'\[zotero:([A-Z0-9]{8})\]', text)
                if m:
                    synced_keys.add(m.group(1))

        notes = self._fetch_zotero_notes(zotero_key)
        annotations = self._fetch_zotero_annotations(zotero_key)

        new_blocks: list[dict] = []

        for note in notes:
            note_key = note.get("key", "")
            if note_key in synced_keys:
                continue
            title = note.get("title") or "Zotero Note"
            header = f"[zotero:{note_key}] {title}"
            new_blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": header[:2000]}}],
                    "icon": {"type": "emoji", "emoji": "📝"},
                    "color": "gray_background",
                },
            })
            if note.get("content"):
                new_blocks.extend(self._zotero_html_to_blocks(note["content"]))

        for ann in annotations:
            ann_key = ann.get("key", "")
            if ann_key in synced_keys:
                continue
            page_label = ann.get("pageLabel", "?")
            highlighted = ann.get("text", "").strip()
            comment = ann.get("comment", "").strip()

            if highlighted:
                quote_text = f"[p.{page_label}] {highlighted}"
                new_blocks.append({
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [{"type": "text", "text": {"content": quote_text[:2000]}}],
                        "color": "yellow_background",
                    },
                })
                new_blocks.append({
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"type": "text", "text": {
                            "content": f"[zotero:{ann_key}]"[:200]
                        }}],
                        "icon": {"type": "emoji", "emoji": "🔖"},
                        "color": "yellow_background",
                    },
                })
            if comment:
                new_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [
                        {"type": "text", "text": {"content": comment[:2000]}}
                    ]},
                })

        if not new_blocks:
            logger.info(
                "ZoteroSync: no new notes/annotations for paper page %s.", paper_page_id
            )
            return

        for i in range(0, len(new_blocks), 100):
            self.notion.append_block_children(paper_page_id, new_blocks[i:i + 100])

        logger.info(
            "ZoteroSync: synced %d block(s) to paper page %s.",
            len(new_blocks), paper_page_id,
        )

    def _get_zotero_key(self, props: dict) -> str | None:
        for prop_name in ("Item Key", "Key"):
            val = _get_text(props, prop_name).strip()
            if val and re.match(r'^[A-Z0-9]{8}$', val, re.IGNORECASE):
                return val.upper()
        uri = props.get("Zotero URI", {}).get("url") or _get_text(props, "Zotero URI")
        if uri:
            match = re.search(r'/items/([A-Z0-9]{8})(?:/|$)', uri, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _fetch_zotero_notes(self, zotero_item_key: str) -> list[dict]:
        if not self.zotero_user_id or not self.zotero_api_key:
            return []
        url = (
            f"https://api.zotero.org/users/{self.zotero_user_id}"
            f"/items/{zotero_item_key}/children"
        )
        try:
            resp = requests.get(
                url,
                headers={"Zotero-API-Key": self.zotero_api_key},
                params={"itemType": "note"},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception:
            logger.warning("ZoteroSync: failed to fetch notes for key %s", zotero_item_key)
            return []

        return [
            {
                "key": item.get("key", ""),
                "title": item["data"].get("title", ""),
                "content": item["data"].get("note", ""),
                "date_modified": item["data"].get("dateModified", ""),
            }
            for item in items
            if item.get("data", {}).get("itemType") == "note"
        ]

    def _fetch_zotero_annotations(self, zotero_item_key: str) -> list[dict]:
        if not self.zotero_user_id or not self.zotero_api_key:
            return []
        url = (
            f"https://api.zotero.org/users/{self.zotero_user_id}"
            f"/items/{zotero_item_key}/children"
        )
        try:
            resp = requests.get(
                url,
                headers={"Zotero-API-Key": self.zotero_api_key},
                params={"itemType": "annotation"},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception:
            logger.warning(
                "ZoteroSync: failed to fetch annotations for key %s", zotero_item_key
            )
            return []

        return [
            {
                "key": item.get("key", ""),
                "type": item["data"].get("annotationType", ""),
                "comment": item["data"].get("comment", ""),
                "text": item["data"].get("annotationText", ""),
                "pageLabel": item["data"].get("pageLabel", ""),
                "date_modified": item["data"].get("dateModified", ""),
            }
            for item in items
            if item.get("data", {}).get("itemType") == "annotation"
        ]

    def _zotero_html_to_blocks(self, html_content: str) -> list[dict]:
        class _Parser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.blocks: list[dict] = []
                self._text = ""
                self._list_type: str | None = None
                self._in_tag: str | None = None

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag in ("p", "h1", "h2", "h3", "li"):
                    self._text = ""
                    self._in_tag = tag
                elif tag == "br":
                    self._text += "\n"
                elif tag == "ul":
                    self._list_type = "ul"
                elif tag == "ol":
                    self._list_type = "ol"

            def handle_endtag(self, tag: str) -> None:
                text = self._text.strip()
                if not text:
                    return
                if tag == "p":
                    self.blocks.append({
                        "object": "block", "type": "paragraph",
                        "paragraph": {"rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ]},
                    })
                elif tag in ("h1", "h2", "h3"):
                    level = int(tag[1])
                    bt = f"heading_{min(level, 3)}"
                    self.blocks.append({
                        "object": "block", "type": bt,
                        bt: {"rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ]},
                    })
                elif tag == "li":
                    bt = "numbered_list_item" if self._list_type == "ol" else "bulleted_list_item"
                    self.blocks.append({
                        "object": "block", "type": bt,
                        bt: {"rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ]},
                    })
                elif tag in ("ul", "ol"):
                    self._list_type = None

            def handle_data(self, data: str) -> None:
                self._text += data

        parser = _Parser()
        try:
            parser.feed(html_content)
        except Exception:
            pass
        if parser._text.strip():
            parser.blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": parser._text.strip()[:2000]}}
                ]},
            })
        return parser.blocks


def _get_text(props: dict, key: str) -> str:
    try:
        segments = props[key]["rich_text"]
        return "".join(seg.get("plain_text", "") for seg in segments)
    except (KeyError, TypeError):
        return ""

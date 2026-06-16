"""
modules/promotion/zotero_sync.py — Zotero note/annotation sync into the Store.

Fetches notes + annotations from the Zotero API and stores them as plain text on
``Paper.ai_notes``. Best-effort and idempotent (skips if notes already present).
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

import requests

from ..store import Store

logger = logging.getLogger(__name__)


class ZoteroSync:
    """Syncs Zotero notes/annotations onto a paper row."""

    def __init__(self, store: Store, zotero_user_id: str, zotero_api_key: str) -> None:
        self.store = store
        self.zotero_user_id = zotero_user_id
        self.zotero_api_key = zotero_api_key

    def sync_zotero_notes(self, paper) -> None:
        if paper.ai_notes:  # already synced
            return
        key = self._zotero_key(paper)
        if not key:
            return

        parts: list[str] = []
        for note in self._fetch_children(key, "note"):
            title = note["data"].get("title") or "Note"
            body = _html_to_text(note["data"].get("note", ""))
            parts.append(f"### {title}\n{body}".strip())
        for ann in self._fetch_children(key, "annotation"):
            data = ann["data"]
            page = data.get("pageLabel", "?")
            text = (data.get("annotationText") or "").strip()
            comment = (data.get("comment") or "").strip()
            if text:
                parts.append(f"> [p.{page}] {text}")
            if comment:
                parts.append(comment)

        if parts:
            self.store.update_paper(paper.id, ai_notes="\n\n".join(parts)[:20_000])
            logger.info("ZoteroSync: stored %d note block(s) for %s.", len(parts), paper.id)

    def _zotero_key(self, paper) -> str | None:
        if paper.zotero_key:
            return paper.zotero_key
        m = re.search(r"/items/([A-Z0-9]{8})(?:/|$)", paper.zotero_uri or "", re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _fetch_children(self, item_key: str, item_type: str) -> list[dict]:
        if not (self.zotero_user_id and self.zotero_api_key):
            return []
        url = f"https://api.zotero.org/users/{self.zotero_user_id}/items/{item_key}/children"
        try:
            resp = requests.get(
                url, headers={"Zotero-API-Key": self.zotero_api_key},
                params={"itemType": item_type}, timeout=30,
            )
            resp.raise_for_status()
            return [it for it in resp.json() if it.get("data", {}).get("itemType") == item_type]
        except Exception:
            logger.warning("ZoteroSync: fetch %s failed for %s.", item_type, item_key)
            return []


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []

    def handle_data(self, data: str) -> None:
        self.out.append(data)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html or "")
    except Exception:
        return html or ""
    return " ".join("".join(p.out).split())

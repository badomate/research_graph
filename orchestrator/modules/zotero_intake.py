"""
modules/zotero_intake.py — poll the Zotero library for new items (replaces Notero).

Fetches top-level Zotero items and inserts a `papers` row (status s0-inbox) for
each one not already in the database, de-duplicated by Zotero item key.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import requests

from .store import PaperStatus, Store, make_engine

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

_API = "https://api.zotero.org"
_SKIP_TYPES = {"attachment", "note", "annotation", "linkAttachment"}


def _creators_to_authors(creators: list[dict]) -> str:
    names = []
    for c in creators or []:
        name = c.get("name") or " ".join(
            p for p in (c.get("firstName", ""), c.get("lastName", "")) if p
        ).strip()
        if name:
            names.append(name)
    return ", ".join(names)


def parse_zotero_items(items: list[dict], user_id: str) -> list[dict]:
    """Pure: Zotero API items → candidate paper dicts (offline-testable)."""
    papers: list[dict] = []
    for item in items:
        data = item.get("data", {})
        if data.get("itemType") in _SKIP_TYPES:
            continue
        key = item.get("key") or data.get("key")
        title = data.get("title")
        if not key or not title:
            continue
        papers.append({
            "title": title,
            "authors": _creators_to_authors(data.get("creators", [])),
            "zotero_key": key,
            "zotero_uri": f"http://zotero.org/users/{user_id}/items/{key}",
            "doi": data.get("DOI", ""),
            "source": "zotero",
            "status": PaperStatus.S0_INBOX.value,
        })
    return papers


class ZoteroIntake:
    """Polls Zotero and creates new paper rows."""

    def __init__(self, config: "Optional[Config]" = None) -> None:
        from .config import get_config
        config = config or get_config()
        self.store = Store(make_engine(config.database_url))
        self.store.create_all()
        self.user_id = config.zotero_user_id
        self.api_key = config.zotero_api_key

    def run(self, limit: int = 50) -> int:
        """Import new top-level Zotero items. Returns the number created."""
        if not (self.user_id and self.api_key):
            logger.info("ZoteroIntake: Zotero credentials not set — skipping.")
            return 0

        items = self._fetch_top(limit)
        created = 0
        for cand in parse_zotero_items(items, self.user_id):
            if self.store.find_paper_by_external(
                zotero_key=cand["zotero_key"], doi=cand.get("doi", "")
            ):
                continue
            self.store.create_paper(**cand)
            created += 1
        logger.info("ZoteroIntake: imported %d new paper(s).", created)
        return created

    def _fetch_top(self, limit: int) -> list[dict]:
        url = f"{_API}/users/{self.user_id}/items/top"
        try:
            resp = requests.get(
                url,
                headers={"Zotero-API-Key": self.api_key},
                params={"limit": limit, "sort": "dateAdded", "direction": "desc"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("ZoteroIntake: failed to fetch Zotero items.", exc_info=True)
            return []

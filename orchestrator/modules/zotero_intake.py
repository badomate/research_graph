"""
modules/zotero_intake.py — poll the Zotero library for new items (replaces Notero).

Fetches top-level Zotero items and inserts a `papers` row (status s0-inbox) for
each one not already in the database, de-duplicated by Zotero item key.

Uses only the stdlib (urllib) and accepts credentials directly, so it can run
inside the light web app (Sync button) as well as the orchestrator scheduler.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request

from .store import PaperStatus, Store, make_engine

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

    def __init__(
        self,
        config=None,
        store: Store | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if store is not None:
            self.store = store
        elif config is not None:
            self.store = Store(make_engine(config.database_url))
        else:
            self.store = Store()  # uses DATABASE_URL env
        self.store.create_all()
        self.user_id = user_id or (config.zotero_user_id if config else os.environ.get("ZOTERO_USER_ID", ""))
        self.api_key = api_key or (config.zotero_api_key if config else os.environ.get("ZOTERO_API_KEY", ""))

    def run(self, limit: int = 50) -> int:
        """Import new top-level Zotero items. Returns the number created."""
        if not (self.user_id and self.api_key):
            logger.info("ZoteroIntake: Zotero credentials not set — skipping.")
            return 0

        created = 0
        for cand in parse_zotero_items(self._fetch_top(limit), self.user_id):
            if self.store.find_paper_by_external(
                zotero_key=cand["zotero_key"], doi=cand.get("doi", "")
            ):
                continue
            self.store.create_paper(**cand)
            created += 1
        logger.info("ZoteroIntake: imported %d new paper(s).", created)
        return created

    def _fetch_top(self, limit: int) -> list[dict]:
        params = urllib.parse.urlencode({"limit": limit, "sort": "dateAdded", "direction": "desc"})
        url = f"{_API}/users/{self.user_id}/items/top?{params}"
        req = urllib.request.Request(url, headers={"Zotero-API-Key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception:
            logger.warning("ZoteroIntake: failed to fetch Zotero items.", exc_info=True)
            return []

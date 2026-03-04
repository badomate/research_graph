"""
modules/notion_client_wrapper.py
─────────────────────────────────
A thin wrapper around the official `notion-client` SDK that adds:

  * A thread-safe rate-limiter capped at 3 requests/second (Notion's limit).
  * Automatic retry with *exponential back-off* on HTTP 429 / 5xx responses
    (implemented via `tenacity`).
  * Convenience helpers used by every other module.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from notion_client import Client
from notion_client.errors import APIResponseError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ── Rate-limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Token-bucket limiter: allows at most `rate` calls per `per` seconds."""

    def __init__(self, rate: float = 3.0, per: float = 1.0) -> None:
        self._rate = rate        # max calls per window
        self._per = per          # window size in seconds
        self._allowance: float = rate
        self._last_check: float = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_check
            self._last_check = now
            self._allowance += elapsed * (self._rate / self._per)
            if self._allowance > self._rate:
                self._allowance = self._rate  # cap
            if self._allowance < 1.0:
                sleep_time = (1.0 - self._allowance) * (self._per / self._rate)
                time.sleep(sleep_time)
                self._allowance = 0.0
            else:
                self._allowance -= 1.0


_rate_limiter = _RateLimiter(rate=3.0, per=1.0)


# ── Retry predicate ───────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 429 (rate limit) and 5xx server errors."""
    if isinstance(exc, APIResponseError):
        return exc.status in (429, 500, 502, 503, 504)
    return False


# ── Public wrapper ────────────────────────────────────────────────────────────

class NotionClientWrapper:
    """
    Wraps every Notion SDK call with:
      1. Rate-limiting  (≤ 3 req/s).
      2. Exponential back-off retry on 429 / 5xx.
    """

    def __init__(self) -> None:
        token = os.environ["NOTION_TOKEN"]
        self._client = Client(auth=token)

    # ── Internal call helper ──────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call(self, fn, *args: Any, **kwargs: Any) -> Any:
        _rate_limiter.acquire()
        return fn(*args, **kwargs)

    # ── Databases ─────────────────────────────────────────────────────────

    def query_database(self, database_id: str, **kwargs: Any) -> list[dict]:
        """Return all pages matching the query (handles pagination)."""
        results: list[dict] = []
        has_more = True
        start_cursor = None

        while has_more:
            params: dict[str, Any] = {"database_id": database_id, **kwargs}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = self._call(self._client.databases.query, **params)
            results.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        return results

    # ── Pages ─────────────────────────────────────────────────────────────

    def get_page(self, page_id: str) -> dict:
        return self._call(self._client.pages.retrieve, page_id=page_id)

    def create_page(self, parent: dict, properties: dict, **kwargs: Any) -> dict:
        logger.info("CREATE_PAGE payload properties keys: %s", list(properties.keys()))
        logger.info("CREATE_PAGE full properties: %s", properties)
    
        return self._call(
            self._client.pages.create,
            parent=parent,
            properties=properties,
            **kwargs,
        )

    def update_page(self, page_id: str, properties: dict, **kwargs: Any) -> dict:
        return self._call(
            self._client.pages.update,
            page_id=page_id,
            properties=properties,
            **kwargs,
        )

    # ── Blocks / comments ─────────────────────────────────────────────────

    def append_block_children(self, block_id: str, children: list[dict]) -> dict:
        return self._call(
            self._client.blocks.children.append,
            block_id=block_id,
            children=children,
        )

    def get_block_children(self, block_id: str) -> list[dict]:
        results: list[dict] = []
        has_more = True
        start_cursor = None

        while has_more:
            params: dict[str, Any] = {"block_id": block_id}
            if start_cursor:
                params["start_cursor"] = start_cursor
            response = self._call(
                self._client.blocks.children.list, **params
            )
            results.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        return results
    
    def get_database(self, database_id: str) -> dict:
        return self._call(self._client.databases.retrieve, database_id=database_id)

    def get_title_property_name(self, database_id: str) -> str:
        db = self.get_database(database_id)
        props = db.get("properties", {})
        for prop_name, prop in props.items():
            if prop.get("type") == "title":
                return prop_name
        raise RuntimeError(f"No title property found in database {database_id}")
    # ── Property helpers ──────────────────────────────────────────────────

    @staticmethod
    def rich_text(content: str) -> list[dict]:
        """Build a Notion rich_text array from a plain string."""
        return [{"type": "text", "text": {"content": content[:2000]}}]

    @staticmethod
    def select_prop(name: str) -> dict:
        return {"select": {"name": name}}

    @staticmethod
    def status_prop(name: str) -> dict:
        """Build a Notion status property value (distinct from select)."""
        return {"status": {"name": name}}

    @staticmethod
    def multi_select_prop(names: list[str]) -> dict:
        return {"multi_select": [{"name": n} for n in names]}

    @staticmethod
    def title_prop(content: str) -> dict:
        return {"title": [{"type": "text", "text": {"content": content[:2000]}}]}

    @staticmethod
    def relation_prop(page_ids: list[str]) -> dict:
        return {"relation": [{"id": pid} for pid in page_ids]}

    @staticmethod
    def checkbox_prop(value: bool) -> dict:
        return {"checkbox": value}

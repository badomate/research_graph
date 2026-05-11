"""
modules/notion/client.py — Config-aware Notion client wrapper
──────────────────────────────────────────────────────────────
Re-exports NotionClientWrapper from the top-level module_client_wrapper.py.
When all callers have been migrated to use this path, the original file
can be removed.
"""

from modules.notion_client_wrapper import NotionClientWrapper  # noqa: F401

__all__ = ["NotionClientWrapper"]

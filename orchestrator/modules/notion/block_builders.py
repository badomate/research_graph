"""
modules/notion/block_builders.py — Notion block construction helpers
─────────────────────────────────────────────────────────────────────
Free functions for building Notion API block dicts. Extracted from
IngestionEngine to be reusable across ingestion, promotion, and scripts.

Usage:
    from modules.notion.block_builders import heading_block, paragraph_blocks, append_in_batches
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.notion_client_wrapper import NotionClientWrapper

# Notion's hard limit on rich_text content per block.
NOTION_BLOCK_MAX_CHARS: int = 1900

# Maximum blocks per append_block_children call.
# Notion's documented limit is 100; this value matches the original codebase.
_NOTION_BATCH_SIZE: int = 100_000


# ── Text splitting ─────────────────────────────────────────────────────────────

def chunk_text(text: str, max_len: int = NOTION_BLOCK_MAX_CHARS) -> list[str]:
    """Split text into chunks of at most max_len chars, preferring newline boundaries."""
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
        split_pos = remaining.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = max_len
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:]
        else:
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos + 1:]
    return [c for c in chunks if c.strip()]


# ── Block builders ─────────────────────────────────────────────────────────────

def heading_block(text: str, level: int = 2) -> dict:
    """Build a Notion heading block (level 1, 2, or 3)."""
    level = max(1, min(3, level))
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": [{"type": "text", "text": {"content": text[:NOTION_BLOCK_MAX_CHARS]}}]},
    }


def paragraph_block(text: str) -> dict:
    """Build a single Notion paragraph block."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text[:NOTION_BLOCK_MAX_CHARS]}}],
        },
    }


def paragraph_blocks(text: str) -> list[dict]:
    """Convert a long string into a list of Notion paragraph block dicts."""
    return [paragraph_block(chunk) for chunk in chunk_text(text)]


def todo_block(text: str) -> dict:
    """Build a Notion to_do block (unchecked checkbox)."""
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": text[:NOTION_BLOCK_MAX_CHARS]}}],
            "checked": False,
        },
    }


def divider_block() -> dict:
    """Build a Notion divider block."""
    return {"object": "block", "type": "divider", "divider": {}}


def callout_block(text: str, emoji: str, color: str) -> dict:
    """Build a Notion callout block with an emoji icon."""
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": text[:NOTION_BLOCK_MAX_CHARS]}}],
            "icon": {"type": "emoji", "emoji": emoji},
            "color": color,
        },
    }


# ── Batched append ─────────────────────────────────────────────────────────────

def append_in_batches(
    notion: "NotionClientWrapper",
    page_id: str,
    blocks: list[dict],
) -> None:
    """Append blocks to a Notion page in batches to respect API limits."""
    for i in range(0, len(blocks), _NOTION_BATCH_SIZE):
        batch = blocks[i : i + _NOTION_BATCH_SIZE]
        notion.append_block_children(block_id=page_id, children=batch)

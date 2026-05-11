"""Tests for modules/notion/block_builders.py — Notion block construction."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.notion.block_builders import (
    NOTION_BLOCK_MAX_CHARS,
    callout_block,
    chunk_text,
    divider_block,
    heading_block,
    paragraph_block,
    paragraph_blocks,
    todo_block,
)


class TestChunkText:
    def test_short_text_returned_as_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n  ") == []

    def test_long_text_splits_into_multiple_chunks(self):
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= NOTION_BLOCK_MAX_CHARS

    def test_prefers_newline_boundary(self):
        text = "line one\n" + "x" * (NOTION_BLOCK_MAX_CHARS - 5) + "\nline two"
        chunks = chunk_text(text)
        assert chunks[0].endswith("line one")

    def test_custom_max_len(self):
        chunks = chunk_text("abcde fghij", max_len=5)
        assert all(len(c) <= 5 for c in chunks)


class TestHeadingBlock:
    def test_level_2_default(self):
        block = heading_block("Introduction")
        assert block["type"] == "heading_2"
        assert "heading_2" in block
        rt = block["heading_2"]["rich_text"]
        assert rt[0]["text"]["content"] == "Introduction"

    def test_level_1(self):
        block = heading_block("Title", level=1)
        assert block["type"] == "heading_1"
        assert "heading_1" in block

    def test_level_3(self):
        block = heading_block("Subheading", level=3)
        assert block["type"] == "heading_3"

    def test_level_clamped_at_3(self):
        block = heading_block("X", level=5)
        assert block["type"] == "heading_3"

    def test_level_clamped_at_1(self):
        block = heading_block("X", level=0)
        assert block["type"] == "heading_1"

    def test_object_field_is_block(self):
        assert heading_block("X")["object"] == "block"

    def test_long_text_truncated(self):
        long_text = "a" * (NOTION_BLOCK_MAX_CHARS + 100)
        block = heading_block(long_text)
        content = block["heading_2"]["rich_text"][0]["text"]["content"]
        assert len(content) <= NOTION_BLOCK_MAX_CHARS


class TestParagraphBlock:
    def test_basic_structure(self):
        block = paragraph_block("Hello world")
        assert block["type"] == "paragraph"
        assert block["object"] == "block"
        rt = block["paragraph"]["rich_text"]
        assert rt[0]["text"]["content"] == "Hello world"

    def test_long_text_truncated(self):
        long_text = "b" * (NOTION_BLOCK_MAX_CHARS + 200)
        block = paragraph_block(long_text)
        content = block["paragraph"]["rich_text"][0]["text"]["content"]
        assert len(content) <= NOTION_BLOCK_MAX_CHARS


class TestParagraphBlocks:
    def test_returns_list_of_blocks(self):
        blocks = paragraph_blocks("short text")
        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"

    def test_long_text_multiple_blocks(self):
        text = "sentence. " * 300  # ~3000 chars
        blocks = paragraph_blocks(text)
        assert len(blocks) > 1
        for b in blocks:
            assert b["type"] == "paragraph"


class TestTodoBlock:
    def test_structure(self):
        block = todo_block("Review this")
        assert block["type"] == "to_do"
        assert block["object"] == "block"
        assert block["to_do"]["checked"] is False
        assert block["to_do"]["rich_text"][0]["text"]["content"] == "Review this"


class TestDividerBlock:
    def test_structure(self):
        block = divider_block()
        assert block["type"] == "divider"
        assert block["object"] == "block"
        assert block["divider"] == {}


class TestCalloutBlock:
    def test_structure(self):
        block = callout_block("Note text", "📝", "gray_background")
        assert block["type"] == "callout"
        assert block["object"] == "block"
        cb = block["callout"]
        assert cb["icon"]["emoji"] == "📝"
        assert cb["color"] == "gray_background"
        assert cb["rich_text"][0]["text"]["content"] == "Note text"

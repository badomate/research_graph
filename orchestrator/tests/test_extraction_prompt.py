"""Tests for the Stage-1 extraction user-message builder.

Locks the prompt format so the instruction block stays well-separated — a
regression guard against the adjacent-string-literal bug where the bullets
collapsed into 'INSTRUCTIONS- Follow the schema strictly.' with no newlines.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.ingestion.extractor import _build_extraction_user_message


def test_instructions_header_is_separated_from_bullets():
    msg = _build_extraction_user_message("paper body")
    # The header must not be glued to the first bullet.
    assert "INSTRUCTIONS- " not in msg
    assert "INSTRUCTIONS\n- Follow the schema strictly." in msg


def test_each_instruction_bullet_is_on_its_own_line():
    msg = _build_extraction_user_message("paper body")
    for bullet in (
        "- Follow the schema strictly.",
        "- Prefer 3–12 high-value concepts.",
        "- Do not output theorem/lemma numbers as titles.",
        "- Do not include proof-only microlemmas.",
    ):
        assert f"\n{bullet}\n" in msg or msg.rstrip().endswith(bullet) or f"\n{bullet}" in msg
    # No two bullets share a line.
    assert "titles.- " not in msg
    assert "concepts.- " not in msg


def test_markdown_is_appended_and_capped():
    body = "X" * 200_000
    msg = _build_extraction_user_message(body)
    assert "PAPER MARKDOWN:\n\n" in msg
    # Body is truncated to 100k chars; total message stays near that bound.
    assert msg.count("X") == 100_000

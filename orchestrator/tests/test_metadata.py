"""Offline tests for intake metadata parsers (no network)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.metadata import parse_arxiv_atom, parse_crossref
from modules.zotero_intake import parse_zotero_items


_ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Mean Field Games and the Master Equation</title>
    <summary>We prove existence and uniqueness.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Roe</name></author>
  </entry>
</feed>"""


def test_parse_arxiv_atom():
    m = parse_arxiv_atom(_ATOM)
    assert m["title"] == "Mean Field Games and the Master Equation"
    assert m["authors"] == "Jane Doe, John Roe"
    assert "existence" in m["abstract"]


def test_parse_arxiv_atom_empty_returns_none():
    assert parse_arxiv_atom(b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>") is None
    assert parse_arxiv_atom(b"not xml") is None


def test_parse_crossref():
    payload = {"message": {
        "title": ["A DOI-Indexed Paper"],
        "author": [{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Roe"}],
        "abstract": "<jats:p>Result.</jats:p>",
    }}
    m = parse_crossref(payload)
    assert m["title"] == "A DOI-Indexed Paper"
    assert m["authors"] == "Jane Doe, John Roe"


def test_parse_crossref_missing_title_returns_none():
    assert parse_crossref({"message": {"author": []}}) is None
    assert parse_crossref({}) is None


def test_parse_zotero_items_skips_non_papers_and_builds_uri():
    items = [
        {"key": "ABC12345", "data": {
            "itemType": "journalArticle", "title": "Zotero Paper",
            "creators": [{"firstName": "Jane", "lastName": "Doe"}], "DOI": "10.1/x"}},
        {"key": "NOTE0001", "data": {"itemType": "note"}},
        {"key": "ATT00001", "data": {"itemType": "attachment"}},
    ]
    papers = parse_zotero_items(items, user_id="999")
    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "Zotero Paper"
    assert p["authors"] == "Jane Doe"
    assert p["zotero_key"] == "ABC12345"
    assert p["zotero_uri"] == "http://zotero.org/users/999/items/ABC12345"
    assert p["status"] == "s0-inbox"

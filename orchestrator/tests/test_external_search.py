"""Offline tests for external-search parsers + merge/dedup."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import external_search as es

_ARXIV_ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Mean Field Games with Common Noise</title>
    <summary>We study an MFG with common noise and prove existence.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>Pierre Lions</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" type="application/pdf"/>
  </entry>
</feed>"""


def test_parse_arxiv_feed():
    res = es.parse_arxiv_feed(_ARXIV_ATOM)
    assert len(res) == 1
    r = res[0]
    assert r.title == "Mean Field Games with Common Noise"
    assert r.arxiv_id == "2401.00001v1"
    assert r.year == "2024"
    assert "Pierre Lions" in r.authors
    assert r.pdf_url.endswith("2401.00001v1")


def test_parse_semantic_scholar():
    payload = {"data": [{
        "title": "A Master Equation Approach",
        "year": 2023,
        "abstract": "We derive a master equation.",
        "authors": [{"name": "A. Author"}],
        "externalIds": {"DOI": "10.1000/xyz", "ArXiv": "2301.12345"},
        "openAccessPdf": {"url": "https://example.org/p.pdf"},
        "venue": "SIAM",
        "url": "https://s2.org/paper/1",
    }]}
    res = es.parse_semantic_scholar(payload)
    assert res[0].doi == "10.1000/xyz"
    assert res[0].arxiv_id == "2301.12345"
    assert res[0].pdf_url == "https://example.org/p.pdf"


def test_parse_crossref_search_strips_jats():
    payload = {"message": {"items": [{
        "title": ["Adjoint Methods for PDEs"],
        "author": [{"given": "B.", "family": "Researcher"}],
        "issued": {"date-parts": [[2022, 5]]},
        "container-title": ["J. Comp. Phys."],
        "abstract": "<jats:p>An adjoint method.</jats:p>",
        "DOI": "10.1016/abc",
        "URL": "https://doi.org/10.1016/abc",
    }]}}
    res = es.parse_crossref_search(payload)
    assert res[0].abstract == "An adjoint method."
    assert res[0].year == "2022"
    assert res[0].venue == "J. Comp. Phys."


def test_merge_dedups_by_arxiv_and_fills_blanks():
    a = es.ExternalResult(source="arxiv", title="X", arxiv_id="2401.00001", pdf_url="p.pdf")
    b = es.ExternalResult(source="semantic_scholar", title="X", arxiv_id="2401.00001",
                          doi="10.1/x", abstract="filled")
    merged = es.merge_results([[a], [b]])
    assert len(merged) == 1
    assert merged[0].doi == "10.1/x"          # filled from the S2 record
    assert merged[0].pdf_url == "p.pdf"       # kept from the arXiv record


def test_merge_keeps_distinct_papers():
    a = es.ExternalResult(source="arxiv", title="A", arxiv_id="1")
    b = es.ExternalResult(source="arxiv", title="B", arxiv_id="2")
    assert len(es.merge_results([[a, b]])) == 2


def test_bibtex_contains_core_fields():
    r = es.ExternalResult(source="crossref", title="T", authors="Jane Doe, Pierre Lions",
                          year="2024", doi="10.1/x", venue="SIAM")
    bib = es.to_bibtex(r)
    assert "title = {T}" in bib
    assert "Jane Doe and Pierre Lions" in bib
    assert "doi = {10.1/x}" in bib


def test_external_search_empty_query_returns_empty():
    assert es.external_search("   ") == []

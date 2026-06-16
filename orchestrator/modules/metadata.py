"""
modules/metadata.py — fetch paper metadata for in-app "Add Paper".

Given an arXiv ID (arXiv Atom API) or a DOI (Crossref), return title / authors /
abstract. Uses only the stdlib (urllib) so the web app needs no extra deps. Parse
functions are split out from the network calls so they can be unit-tested offline.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_CROSSREF_API = "https://api.crossref.org/works/"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


# ── Pure parsers (offline-testable) ───────────────────────────────────────────


def parse_arxiv_atom(content: bytes) -> dict | None:
    """Parse an arXiv Atom response (single entry) → metadata dict, or None."""
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return None
    entry = root.find("atom:entry", _ATOM_NS)
    if entry is None:
        return None
    title_el = entry.find("atom:title", _ATOM_NS)
    summary_el = entry.find("atom:summary", _ATOM_NS)
    if title_el is None:
        return None
    authors = [
        " ".join((n.text or "").split())
        for n in entry.findall("atom:author/atom:name", _ATOM_NS)
        if n.text
    ]
    return {
        "title": " ".join((title_el.text or "").split()),
        "authors": ", ".join(authors),
        "abstract": " ".join((summary_el.text or "").split()) if summary_el is not None else "",
    }


def parse_crossref(payload: dict) -> dict | None:
    """Parse a Crossref /works/{doi} JSON payload → metadata dict, or None."""
    msg = payload.get("message")
    if not isinstance(msg, dict):
        return None
    title_list = msg.get("title") or []
    title = title_list[0] if title_list else ""
    if not title:
        return None
    authors = [
        " ".join(p for p in (a.get("given", ""), a.get("family", "")) if p).strip()
        for a in msg.get("author", [])
    ]
    abstract = msg.get("abstract", "")
    return {
        "title": title,
        "authors": ", ".join(a for a in authors if a),
        "abstract": abstract,
    }


# ── Network wrappers ───────────────────────────────────────────────────────────


def _get(url: str, timeout: int = 20) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        logger.warning("metadata: request failed for %s", url, exc_info=True)
        return None


def fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return None
    url = f"{_ARXIV_API}?{urllib.parse.urlencode({'id_list': arxiv_id, 'max_results': 1})}"
    content = _get(url)
    return parse_arxiv_atom(content) if content else None


def fetch_doi_metadata(doi: str) -> dict | None:
    doi = doi.strip()
    if not doi:
        return None
    content = _get(_CROSSREF_API + urllib.parse.quote(doi))
    if not content:
        return None
    try:
        return parse_crossref(json.loads(content))
    except (ValueError, TypeError):
        return None

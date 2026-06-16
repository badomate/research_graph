"""
modules/external_search.py — search external sources for *unsaved* papers.

Queries arXiv (Atom), Semantic Scholar (Graph API) and Crossref (works search),
normalizes results to a common shape, and merges/de-dups across sources. Network
wrappers use only the stdlib (urllib) so the web app needs no extra deps; the
parse functions are split out so they unit-test offline.

Nothing here saves anything — results are candidates the user explicitly saves
from the UI (requirement: never auto-save every external result).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_CROSSREF_API = "https://api.crossref.org/works"
_SERPAPI_API = "https://serpapi.com/search"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

SOURCES = ("arxiv", "semantic_scholar", "crossref")


def available_sources() -> tuple[str, ...]:
    """Sources usable right now. SerpAPI (Google Scholar) only when keyed."""
    if os.environ.get("SERPAPI_KEY"):
        return SOURCES + ("serpapi",)
    return SOURCES


@dataclass
class ExternalResult:
    source: str
    title: str
    authors: str = ""
    year: str = ""
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    arxiv_id: str = ""
    doi: str = ""
    venue: str = ""

    def dedup_key(self) -> str:
        """Identity for cross-source merge: arXiv id, else DOI, else norm title."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return "title:" + "".join(c for c in self.title.lower() if c.isalnum())

    def as_dict(self) -> dict:
        return asdict(self)


# ── Pure parsers (offline-testable) ──────────────────────────────────────────────


def parse_arxiv_feed(content: bytes) -> list[ExternalResult]:
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return []
    out: list[ExternalResult] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title_el = entry.find("atom:title", _ATOM_NS)
        if title_el is None or not (title_el.text or "").strip():
            continue
        summary_el = entry.find("atom:summary", _ATOM_NS)
        id_el = entry.find("atom:id", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        authors = [
            " ".join((n.text or "").split())
            for n in entry.findall("atom:author/atom:name", _ATOM_NS) if n.text
        ]
        abs_url = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = abs_url.rsplit("/abs/", 1)[-1] if "/abs/" in abs_url else ""
        pdf_url = ""
        for link in entry.findall("atom:link", _ATOM_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
        year = (published_el.text or "")[:4] if published_el is not None else ""
        out.append(ExternalResult(
            source="arxiv",
            title=" ".join((title_el.text or "").split()),
            authors=", ".join(authors),
            year=year,
            abstract=" ".join((summary_el.text or "").split()) if summary_el is not None else "",
            url=abs_url, pdf_url=pdf_url, arxiv_id=arxiv_id,
        ))
    return out


def parse_semantic_scholar(payload: dict) -> list[ExternalResult]:
    if not isinstance(payload, dict):
        return []
    out: list[ExternalResult] = []
    for p in payload.get("data", []) or []:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        ext = p.get("externalIds") or {}
        oa = p.get("openAccessPdf") or {}
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or []) if a.get("name"))
        out.append(ExternalResult(
            source="semantic_scholar",
            title=title, authors=authors,
            year=str(p.get("year") or ""),
            abstract=p.get("abstract") or "",
            url=p.get("url", ""), pdf_url=oa.get("url", ""),
            arxiv_id=str(ext.get("ArXiv", "") or ""),
            doi=str(ext.get("DOI", "") or ""),
            venue=p.get("venue", "") or "",
        ))
    return out


def _strip_jats(text: str) -> str:
    """Crossref abstracts arrive as JATS XML; strip tags for display."""
    import re
    return re.sub(r"<[^>]+>", "", text or "").strip()


def parse_crossref_search(payload: dict) -> list[ExternalResult]:
    msg = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(msg, dict):
        return []
    out: list[ExternalResult] = []
    for item in msg.get("items", []) or []:
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            continue
        authors = [
            " ".join(p for p in (a.get("given", ""), a.get("family", "")) if p).strip()
            for a in item.get("author", []) or []
        ]
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else ""
        venue = (item.get("container-title") or [""])[0]
        out.append(ExternalResult(
            source="crossref",
            title=title, authors=", ".join(a for a in authors if a),
            year=year, abstract=_strip_jats(item.get("abstract", "")),
            url=item.get("URL", ""), doi=item.get("DOI", "") or "", venue=venue,
        ))
    return out


def parse_serpapi_scholar(payload: dict) -> list[ExternalResult]:
    """Parse a SerpAPI Google Scholar response → results."""
    if not isinstance(payload, dict):
        return []
    out: list[ExternalResult] = []
    for item in payload.get("organic_results", []) or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        info = item.get("publication_info") or {}
        summary = info.get("summary", "")
        # summary looks like "A Author, B Author - Journal, 2023 - publisher"
        year = ""
        for tok in summary.replace(",", " ").split():
            if tok.isdigit() and len(tok) == 4 and tok.startswith(("19", "20")):
                year = tok
                break
        resources = item.get("resources") or []
        pdf_url = next((r.get("link", "") for r in resources if r.get("file_format") == "PDF"), "")
        out.append(ExternalResult(
            source="serpapi", title=title,
            authors=summary.split(" - ")[0] if " - " in summary else "",
            year=year, abstract=item.get("snippet", ""),
            url=item.get("link", ""), pdf_url=pdf_url,
        ))
    return out


def merge_results(groups: list[list[ExternalResult]]) -> list[ExternalResult]:
    """Merge across sources, de-duping by identity and filling blank fields."""
    merged: dict[str, ExternalResult] = {}
    order: list[str] = []
    for group in groups:
        for r in group:
            key = r.dedup_key()
            if key not in merged:
                merged[key] = r
                order.append(key)
            else:
                cur = merged[key]
                for f in ("authors", "year", "abstract", "pdf_url", "doi", "arxiv_id", "venue", "url"):
                    if not getattr(cur, f) and getattr(r, f):
                        setattr(cur, f, getattr(r, f))
    return [merged[k] for k in order]


# ── Network wrappers ─────────────────────────────────────────────────────────────


def _get(url: str, timeout: int = 20) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper_pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        logger.warning("external_search: request failed for %s", url, exc_info=True)
        return None


def search_arxiv(query: str, limit: int = 10) -> list[ExternalResult]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit,
              "sortBy": "relevance"}
    content = _get(f"{_ARXIV_API}?{urllib.parse.urlencode(params)}")
    return parse_arxiv_feed(content) if content else []


def search_semantic_scholar(query: str, limit: int = 10) -> list[ExternalResult]:
    fields = "title,authors,year,abstract,externalIds,openAccessPdf,venue,url"
    params = {"query": query, "limit": limit, "fields": fields}
    content = _get(f"{_S2_API}?{urllib.parse.urlencode(params)}")
    if not content:
        return []
    try:
        return parse_semantic_scholar(json.loads(content))
    except (ValueError, TypeError):
        return []


def search_crossref(query: str, limit: int = 10) -> list[ExternalResult]:
    params = {"query": query, "rows": limit}
    content = _get(f"{_CROSSREF_API}?{urllib.parse.urlencode(params)}")
    if not content:
        return []
    try:
        return parse_crossref_search(json.loads(content))
    except (ValueError, TypeError):
        return []


def search_serpapi(query: str, limit: int = 10) -> list[ExternalResult]:
    api_key = os.environ.get("SERPAPI_KEY", "")
    if not api_key:
        return []
    params = {"engine": "google_scholar", "q": query, "num": min(limit, 20), "api_key": api_key}
    content = _get(f"{_SERPAPI_API}?{urllib.parse.urlencode(params)}")
    if not content:
        return []
    try:
        return parse_serpapi_scholar(json.loads(content))
    except (ValueError, TypeError):
        return []


_SEARCHERS = {
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "crossref": search_crossref,
    "serpapi": search_serpapi,
}


def external_search(query: str, sources: list[str] | None = None, limit: int = 10) -> list[ExternalResult]:
    """Search the requested sources (in preference order) and merge results."""
    query = (query or "").strip()
    if not query:
        return []
    chosen = [s for s in (sources or available_sources()) if s in _SEARCHERS]
    groups = []
    for src in chosen:
        try:
            groups.append(_SEARCHERS[src](query, limit))
        except Exception:
            logger.warning("external_search: %s failed", src, exc_info=True)
            groups.append([])
    return merge_results(groups)


def to_bibtex(r: ExternalResult) -> str:
    """Minimal BibTeX for a result (best-effort from available fields)."""
    first_author = (r.authors.split(",")[0] if r.authors else "anon").split()
    key = (first_author[-1] if first_author else "anon").lower() + (r.year or "")
    kind = "article"
    lines = [f"@{kind}{{{key},", f"  title = {{{r.title}}},"]
    if r.authors:
        lines.append(f"  author = {{{r.authors.replace(', ', ' and ')}}},")
    if r.year:
        lines.append(f"  year = {{{r.year}}},")
    if r.venue:
        lines.append(f"  journal = {{{r.venue}}},")
    if r.doi:
        lines.append(f"  doi = {{{r.doi}}},")
    if r.arxiv_id:
        lines.append(f"  eprint = {{{r.arxiv_id}}},")
    lines.append("}")
    return "\n".join(lines)

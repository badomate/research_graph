"""
webapp/main.py — the paper_pipeline web UI (replaces Notion).

Server-rendered FastAPI + Jinja + HTMX. Reads/writes the same SQLite database the
orchestrator uses (the ``Store`` from orchestrator/modules/store). Designed to be
fast and low-friction: one-click verify/reject, inline edits, keyboard review.

Run locally:
    DATABASE_URL=sqlite:///./app.db uvicorn webapp.main:app --reload
    # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the orchestrator's modules importable (Store lives there).
_ORCH = (Path(__file__).resolve().parent.parent / "orchestrator").resolve()
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# Local-friendly default so `uvicorn webapp.main:app` just works without env.
os.environ.setdefault("DATABASE_URL", "sqlite:///./app.db")

from fastapi import FastAPI, Form, Request, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from modules.metadata import fetch_arxiv_metadata, fetch_doi_metadata  # noqa: E402
from modules.store import (  # noqa: E402
    ConceptState,
    EdgeStatus,
    PaperStatus,
    Store,
    VerificationStatus,
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
store = Store()

app = FastAPI(title="paper_pipeline")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.on_event("startup")
def _startup() -> None:
    store.create_all()


# ── Display metadata (kept tiny + muted; the UI should not be noisy) ───────────

STATUS_META: dict[str, dict] = {
    "s0-inbox": {"label": "Inbox", "color": "#8d99ae"},
    "s1-skim": {"label": "Queued", "color": "#e0a458"},
    "s1-processing": {"label": "Processing", "color": "#d6b34a"},
    "s1b-waiting-attachment": {"label": "Waiting PDF", "color": "#c98a4b"},
    "blocked-extraction": {"label": "Blocked", "color": "#c45b5b"},
    "s2-extracted": {"label": "To review", "color": "#4a8fbf"},
    "s2-reextract": {"label": "Re-extract", "color": "#7a6cc4"},
    "s2-read": {"label": "Promoting", "color": "#4a8fbf"},
    "s3-distilled": {"label": "Done", "color": "#5aa469"},
}

# Order papers columns flow in on the dashboard.
STATUS_ORDER = [
    "s0-inbox", "s1-skim", "s1-processing", "s1b-waiting-attachment",
    "blocked-extraction", "s2-extracted", "s2-reextract", "s2-read", "s3-distilled",
]

VERIF_META = {
    "unverified": {"label": "Unverified", "color": "#9aa0a6"},
    "verified": {"label": "Verified", "color": "#5aa469"},
    "rejected": {"label": "Rejected", "color": "#c45b5b"},
}


def _ctx(request: Request, **extra) -> dict:
    """Common template context (nav counts etc.)."""
    pending = sum(
        1 for c in store.list_concepts(state=ConceptState.INBOX.value)
        if c.verification_status == VerificationStatus.UNVERIFIED.value
    )
    base = {
        "request": request,
        "status_meta": STATUS_META,
        "verif_meta": VERIF_META,
        "pending_review": pending,
    }
    base.update(extra)
    return base


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _edge_view(edge) -> dict:
    """Resolve an edge into a flat dict the templates can render directly."""
    target = store.get_concept(edge.target_concept_id) if edge.target_concept_id else None
    source = store.get_concept(edge.source_concept_id)
    return {
        "edge": edge,
        "source_title": source.effective_title if source else "?",
        "target_title": (target.effective_title if target else edge.target_title_raw) or "?",
        "target_id": edge.target_concept_id,
    }


# ── Papers dashboard ───────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    papers = store.list_papers()
    by_status: dict[str, list] = {s: [] for s in STATUS_ORDER}
    for p in papers:
        by_status.setdefault(p.status, []).append(p)
    columns = [
        {"status": s, "meta": STATUS_META.get(s, {"label": s, "color": "#888"}), "papers": by_status.get(s, [])}
        for s in STATUS_ORDER
        if by_status.get(s)
    ]
    return templates.TemplateResponse(
        "papers.html",
        _ctx(request, columns=columns, total=len(papers)),
    )


@app.get("/papers/new", response_class=HTMLResponse)
def add_paper_form(request: Request):
    return templates.TemplateResponse("add_paper.html", _ctx(request))


@app.post("/papers")
async def create_paper(
    request: Request,
    title: str = Form(""),
    arxiv_id: str = Form(""),
    doi: str = Form(""),
    pdf: UploadFile | None = None,
):
    pdf_path = ""
    authors = ""
    abstract = ""
    if pdf is not None and pdf.filename:
        uploads = Path(os.environ.get("UPLOADS_DIR", "./uploads"))
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / pdf.filename
        dest.write_bytes(await pdf.read())
        pdf_path = str(dest)
        if not title:
            title = pdf.filename.rsplit(".", 1)[0]

    # Enrich from arXiv/DOI metadata (off the event loop) when no PDF was uploaded.
    if not pdf_path and (arxiv_id.strip() or doi.strip()):
        meta = None
        if arxiv_id.strip():
            meta = await run_in_threadpool(fetch_arxiv_metadata, arxiv_id)
        elif doi.strip():
            meta = await run_in_threadpool(fetch_doi_metadata, doi)
        if meta:
            title = title or meta.get("title", "")
            authors = meta.get("authors", "")
            abstract = meta.get("abstract", "")

    if not title:
        title = arxiv_id or doi or "Untitled paper"
    paper = store.create_paper(
        title=title,
        authors=authors,
        arxiv_id=arxiv_id.strip(),
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id.strip()}" if arxiv_id.strip() else "",
        doi=doi.strip(),
        one_liner=abstract[:500],
        pdf_path=pdf_path,
        source="manual",
        status=PaperStatus.S1_SKIM.value,
    )
    return RedirectResponse(url=f"/papers/{paper.id}", status_code=303)


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
def paper_detail(request: Request, paper_id: str):
    paper = store.get_paper(paper_id)
    if paper is None:
        return RedirectResponse(url="/", status_code=303)
    concepts = store.concepts_for_paper(paper_id)
    return templates.TemplateResponse(
        "paper_detail.html",
        _ctx(request, paper=paper, concepts=concepts),
    )


@app.post("/papers/{paper_id}/status")
def set_paper_status(request: Request, paper_id: str, status: str = Form(...)):
    store.set_paper_status(paper_id, status)
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


# ── Review queue (the core workflow) ───────────────────────────────────────────


@app.get("/review", response_class=HTMLResponse)
def review_all(request: Request):
    concepts = [
        c for c in store.list_concepts(state=ConceptState.INBOX.value)
        if c.verification_status == VerificationStatus.UNVERIFIED.value
    ]
    return _render_review(request, concepts, title="Review queue", paper=None)


@app.get("/papers/{paper_id}/review", response_class=HTMLResponse)
def review_paper(request: Request, paper_id: str):
    paper = store.get_paper(paper_id)
    concepts = store.concepts_for_paper(paper_id, state=ConceptState.INBOX.value)
    return _render_review(request, concepts, title=paper.title if paper else "Review", paper=paper)


def _render_review(request: Request, concepts, title: str, paper):
    cards = [
        {"concept": c, "edges": [_edge_view(e) for e in store.proposed_edges_for_concept(c.id)]}
        for c in concepts
    ]
    return templates.TemplateResponse(
        "review.html",
        _ctx(request, cards=cards, review_title=title, paper=paper),
    )


def _card_response(request: Request, concept_id: str):
    concept = store.get_concept(concept_id)
    edges = [_edge_view(e) for e in store.proposed_edges_for_concept(concept_id)]
    return templates.TemplateResponse(
        "partials/_concept_card.html",
        _ctx(request, card={"concept": concept, "edges": edges}),
    )


@app.post("/concepts/{concept_id}/verify", response_class=HTMLResponse)
def verify_concept(request: Request, concept_id: str):
    store.set_verification(concept_id, VerificationStatus.VERIFIED.value)
    return _card_response(request, concept_id)


@app.post("/concepts/{concept_id}/reject", response_class=HTMLResponse)
def reject_concept(request: Request, concept_id: str):
    store.set_verification(concept_id, VerificationStatus.REJECTED.value)
    return _card_response(request, concept_id)


@app.get("/concepts/{concept_id}/card", response_class=HTMLResponse)
def concept_card(request: Request, concept_id: str):
    return _card_response(request, concept_id)


@app.get("/concepts/{concept_id}/edit", response_class=HTMLResponse)
def edit_concept_form(request: Request, concept_id: str):
    concept = store.get_concept(concept_id)
    return templates.TemplateResponse(
        "partials/_concept_edit.html",
        _ctx(request, concept=concept),
    )


@app.post("/concepts/{concept_id}", response_class=HTMLResponse)
def save_concept(
    request: Request,
    concept_id: str,
    corrected_title: str = Form(""),
    statement_latex: str = Form(""),
    assumptions: str = Form(""),
    conclusion: str = Form(""),
    reviewer_notes: str = Form(""),
):
    store.update_concept(
        concept_id,
        corrected_title=corrected_title.strip(),
        statement_latex=statement_latex,
        assumptions=assumptions,
        conclusion=conclusion,
        reviewer_notes=reviewer_notes,
    )
    return _card_response(request, concept_id)


# ── Edge accept / reject ───────────────────────────────────────────────────────


def _edge_response(request: Request, edge_id: str):
    edge = store.get_edge(edge_id)
    return templates.TemplateResponse(
        "partials/_edge.html",
        _ctx(request, ev=_edge_view(edge)),
    )


@app.post("/edges/{edge_id}/accept", response_class=HTMLResponse)
def accept_edge(request: Request, edge_id: str):
    store.set_edge_status(edge_id, EdgeStatus.VERIFIED.value)
    return _edge_response(request, edge_id)


@app.post("/edges/{edge_id}/reject", response_class=HTMLResponse)
def reject_edge(request: Request, edge_id: str):
    store.set_edge_status(edge_id, EdgeStatus.REJECTED.value)
    return _edge_response(request, edge_id)


# ── Second Brain ───────────────────────────────────────────────────────────────


@app.get("/brain", response_class=HTMLResponse)
def brain(request: Request, q: str = ""):
    concepts = store.list_concepts(state=ConceptState.PROMOTED.value)
    if q:
        ql = q.lower()
        concepts = [
            c for c in concepts
            if ql in c.effective_title.lower()
            or any(ql in k.lower() for k in c.canonical_keywords)
        ]
    groups: dict[str, list] = {}
    for c in concepts:
        groups.setdefault(c.suggested_hub or "Uncategorized", []).append(c)
    grouped = sorted(groups.items(), key=lambda kv: kv[0].lower())
    return templates.TemplateResponse(
        "brain.html",
        _ctx(request, grouped=grouped, q=q, count=len(concepts)),
    )


@app.get("/concepts/{concept_id}", response_class=HTMLResponse)
def concept_detail(request: Request, concept_id: str):
    concept = store.get_concept(concept_id)
    if concept is None:
        return RedirectResponse(url="/brain", status_code=303)
    edges = store.edges_for_concept(concept_id)
    outgoing = [_edge_view(e) for e in edges if e.source_concept_id == concept_id]
    incoming = [_edge_view(e) for e in edges if e.target_concept_id == concept_id]
    paper = store.get_paper(concept.paper_id) if concept.paper_id else None
    return templates.TemplateResponse(
        "concept_detail.html",
        _ctx(request, concept=concept, outgoing=outgoing, incoming=incoming, paper=paper),
    )


# ── Graph ──────────────────────────────────────────────────────────────────────


@app.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request):
    return templates.TemplateResponse("graph.html", _ctx(request))


@app.get("/api/graph.json")
def graph_json(verified_only: bool = True):
    return JSONResponse(store.graph_data(verified_only=verified_only))


# ── Search ───────────────────────────────────────────────────────────────────


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = ""):
    results = store.search_concepts(q) if q.strip() else []
    return templates.TemplateResponse(
        "search.html",
        _ctx(request, q=q, results=results),
    )


@app.get("/health")
def health():
    return {"status": "ok"}

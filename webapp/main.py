"""
webapp/main.py — the paper_pipeline web UI (replaces Notion).

Server-rendered FastAPI + Jinja + HTMX over the shared SQLite ``Store``. Built for
daily use: filtered browsing, bulk review, one-click promotion, and manual
authoring of concepts and edges.

Run locally:
    DATABASE_URL=sqlite:///./app.db uvicorn webapp.main:app --reload
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ORCH = (Path(__file__).resolve().parent.parent / "orchestrator").resolve()
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
os.environ.setdefault("DATABASE_URL", "sqlite:///./app.db")

from fastapi import FastAPI, Form, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from modules import cost  # noqa: E402
from modules.metadata import fetch_arxiv_metadata, fetch_doi_metadata  # noqa: E402
from modules.parsing import scope_utils  # noqa: E402
from modules.config import get_config  # noqa: E402
from modules.store import (  # noqa: E402
    AnalysisType,
    ConceptState,
    EdgeStatus,
    JobStatus,
    PaperRole,
    PaperStatus,
    RelationType,
    ScopePurpose,
    Store,
    SuggestionStatus,
    SuggestionType,
    VerificationStatus,
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
store = Store()

_template_response = templates.TemplateResponse


def render(name: str, ctx: dict):
    """Render via the modern Starlette signature (request, name, context).

    The legacy positional (name, context) form was removed in newer Starlette, so
    we always pass the request first; every ctx here is built by ``_ctx`` and
    includes "request".
    """
    return _template_response(ctx["request"], name, ctx)

app = FastAPI(title="paper_pipeline")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

CONCEPT_TYPES = ["Definition", "Theorem", "Lemma", "Algorithm", "Assumption", "Proof", "ProofTechnique"]
RELATION_TYPES = [r.value for r in RelationType]


@app.on_event("startup")
def _startup() -> None:
    store.create_all()


# ── Display metadata ───────────────────────────────────────────────────────────

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
STATUS_ORDER = list(STATUS_META)
VERIF_META = {
    "unverified": {"label": "Unverified", "color": "#9aa0a6"},
    "verified": {"label": "Verified", "color": "#5aa469"},
    "rejected": {"label": "Rejected", "color": "#c45b5b"},
}


PAPER_ROLES = [r.value for r in PaperRole]
ANALYSIS_TYPES = [a.value for a in AnalysisType]
SCOPE_PURPOSES = [p.value for p in ScopePurpose]
_CFG = get_config()


def _ctx(request: Request, **extra) -> dict:
    pending = len(store.query_concepts(
        states=[ConceptState.INBOX.value], verification=VerificationStatus.UNVERIFIED.value
    ))
    base = {
        "request": request, "status_meta": STATUS_META, "verif_meta": VERIF_META,
        "pending_review": pending, "concept_types": CONCEPT_TYPES, "relation_types": RELATION_TYPES,
        "pending_suggestions": len(store.list_suggestions(status=SuggestionStatus.PENDING.value)),
        "paper_roles": PAPER_ROLES, "analysis_types": ANALYSIS_TYPES, "scope_purposes": SCOPE_PURPOSES,
    }
    base.update(extra)
    return base


def _edge_view(edge) -> dict:
    target = store.get_concept(edge.target_concept_id) if edge.target_concept_id else None
    source = store.get_concept(edge.source_concept_id)
    return {
        "edge": edge,
        "source_title": source.effective_title if source else "?",
        "target_title": (target.effective_title if target else edge.target_title_raw) or "?",
        "target_id": edge.target_concept_id,
    }


def _parse_keywords(raw: str) -> list[str]:
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


# ── Papers dashboard ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, q: str = "", imported: str = ""):
    papers = store.list_papers()
    if q.strip():
        ql = q.lower()
        papers = [p for p in papers if ql in p.title.lower() or ql in (p.authors or "").lower()]
    by_status: dict[str, list] = {}
    for p in papers:
        by_status.setdefault(p.status, []).append(p)
    columns = [
        {"status": s, "meta": STATUS_META.get(s, {"label": s, "color": "#888"}), "papers": by_status.get(s, [])}
        for s in STATUS_ORDER if by_status.get(s)
    ]
    zotero_configured = bool(os.environ.get("ZOTERO_USER_ID") and os.environ.get("ZOTERO_API_KEY"))
    return render("papers.html", _ctx(
        request, columns=columns, total=len(papers), q=q,
        zotero_configured=zotero_configured, imported=imported))


@app.post("/zotero/sync")
def zotero_sync():
    """Import new Zotero items now (the Sync button)."""
    from modules.zotero_intake import ZoteroIntake
    try:
        n = ZoteroIntake().run()
    except Exception:  # never 500 the dashboard over an intake hiccup
        n = -1
    return RedirectResponse(url=f"/?imported={n}", status_code=303)


@app.get("/papers/new", response_class=HTMLResponse)
def add_paper_form(request: Request):
    return render("add_paper.html", _ctx(request))


@app.post("/papers")
async def create_paper(
    request: Request, title: str = Form(""), arxiv_id: str = Form(""),
    doi: str = Form(""), zotero_uri: str = Form(""), pdf: UploadFile | None = None,
):
    pdf_path = authors = abstract = ""
    if pdf is not None and pdf.filename:
        uploads = Path(os.environ.get("UPLOADS_DIR", "./uploads"))
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / pdf.filename
        dest.write_bytes(await pdf.read())
        pdf_path = str(dest)
        title = title or pdf.filename.rsplit(".", 1)[0]
    if not pdf_path and (arxiv_id.strip() or doi.strip()):
        meta = (await run_in_threadpool(fetch_arxiv_metadata, arxiv_id) if arxiv_id.strip()
                else await run_in_threadpool(fetch_doi_metadata, doi))
        if meta:
            title = title or meta.get("title", "")
            authors, abstract = meta.get("authors", ""), meta.get("abstract", "")
    title = title or arxiv_id or doi or "Untitled paper"
    paper = store.create_paper(
        title=title, authors=authors, arxiv_id=arxiv_id.strip(),
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id.strip()}" if arxiv_id.strip() else "",
        doi=doi.strip(), zotero_uri=zotero_uri.strip(), one_liner=abstract[:500],
        pdf_path=pdf_path, source="manual", status=PaperStatus.S1_SKIM.value,
    )
    return RedirectResponse(url=f"/papers/{paper.id}", status_code=303)


@app.post("/papers/{paper_id}/link")
def set_paper_link(paper_id: str, zotero_uri: str = Form("")):
    """Attach a Zotero URI to a manually-added paper so its Koofr PDF resolves."""
    store.update_paper(paper_id, zotero_uri=zotero_uri.strip())
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
def paper_detail(request: Request, paper_id: str):
    paper = store.get_paper(paper_id)
    if paper is None:
        return RedirectResponse(url="/", status_code=303)
    concepts = store.concepts_for_paper(paper_id)
    return render("paper_detail.html", _ctx(
        request, paper=paper, concepts=concepts, counts=store.paper_review_counts(paper_id),
        orgs=_paper_orgs(paper_id), all_tags=store.list_tags(),
        all_collections=store.list_collections(), all_projects=store.list_projects(),
        suggestion_count=len(store.list_suggestions(paper_id=paper_id, status=SuggestionStatus.PENDING.value))))


@app.get("/papers/{paper_id}/pdf")
def paper_pdf(paper_id: str):
    """View the paper: serve an uploaded PDF inline, else redirect to arXiv/DOI."""
    paper = store.get_paper(paper_id)
    if paper is None:
        return RedirectResponse(url="/", status_code=303)
    if paper.pdf_path and Path(paper.pdf_path).exists():
        return FileResponse(
            paper.pdf_path, media_type="application/pdf",
            filename=Path(paper.pdf_path).name, content_disposition_type="inline",
        )
    if paper.arxiv_id:
        return RedirectResponse(url=f"https://arxiv.org/pdf/{paper.arxiv_id}", status_code=307)
    if paper.arxiv_url:
        return RedirectResponse(url=paper.arxiv_url, status_code=307)
    if paper.doi:
        return RedirectResponse(url=f"https://doi.org/{paper.doi}", status_code=307)
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


@app.post("/papers/{paper_id}/status")
def set_paper_status(paper_id: str, status: str = Form(...)):
    store.set_paper_status(paper_id, status)
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


@app.post("/papers/{paper_id}/promote")
def promote_paper(paper_id: str):
    store.promote_paper(paper_id)
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


# ── Manual concept authoring ────────────────────────────────────────────────────

@app.get("/papers/{paper_id}/concepts/new", response_class=HTMLResponse)
def new_concept_form(request: Request, paper_id: str):
    paper = store.get_paper(paper_id)
    return render("concept_new.html", _ctx(request, paper=paper))


@app.post("/papers/{paper_id}/concepts")
def create_concept(
    paper_id: str, title: str = Form(...), type: str = Form("Definition"),
    statement_latex: str = Form(""), assumptions: str = Form(""), conclusion: str = Form(""),
    suggested_hub: str = Form(""), keywords: str = Form(""),
):
    c = store.create_concept(
        paper_id=paper_id, title=title, type=type, statement_latex=statement_latex,
        assumptions=assumptions, conclusion=conclusion, suggested_hub=suggested_hub,
        canonical_keywords=_parse_keywords(keywords),
        verification_status=VerificationStatus.VERIFIED.value,  # hand-authored = trusted
    )
    return RedirectResponse(url=f"/concepts/{c.id}", status_code=303)


# ── Review queue (filters + bulk) ────────────────────────────────────────────────

def _render_review(request: Request, *, paper, filters: dict):
    concepts = store.query_concepts(
        states=[ConceptState.INBOX.value],
        paper_id=paper.id if paper else None,
        type=filters.get("type") or None,
        hub=filters.get("hub") or None,
        verification=filters.get("verification") or None,
        min_confidence=filters.get("min_confidence"),
        q=filters.get("q") or None,
        sort=filters.get("sort") or "confidence",
    )
    cards = [{"concept": c, "edges": [_edge_view(e) for e in store.proposed_edges_for_concept(c.id)]}
             for c in concepts]
    ctx = _ctx(
        request, cards=cards, paper=paper, filters=filters,
        base_url=f"/papers/{paper.id}/review" if paper else "/review",
        hubs=store.distinct_hubs(), types=store.distinct_types(),
        counts=store.paper_review_counts(paper.id) if paper else None,
    )
    return ctx


def _filters_from_query(type: str, hub: str, verification: str, min_confidence: str, q: str, sort: str) -> dict:
    mc = None
    try:
        mc = float(min_confidence) if min_confidence else None
    except ValueError:
        mc = None
    return {"type": type, "hub": hub, "verification": verification, "min_confidence": mc, "q": q, "sort": sort}


@app.get("/review", response_class=HTMLResponse)
def review_all(request: Request, type: str = "", hub: str = "",
               verification: str = "unverified", min_confidence: str = "", q: str = "", sort: str = "confidence"):
    filters = _filters_from_query(type, hub, verification, min_confidence, q, sort)
    return render("review.html", _render_review(request, paper=None, filters=filters))


@app.get("/papers/{paper_id}/review", response_class=HTMLResponse)
def review_paper(request: Request, paper_id: str, type: str = "", hub: str = "",
                 verification: str = "", min_confidence: str = "", q: str = "", sort: str = "confidence"):
    paper = store.get_paper(paper_id)
    filters = _filters_from_query(type, hub, verification, min_confidence, q, sort)
    return render("review.html", _render_review(request, paper=paper, filters=filters))


@app.post("/concepts/bulk", response_class=HTMLResponse)
def bulk_action(
    request: Request, action: str = Form(...), ids: list[str] = Form(default=[]),
    paper_id: str = Form(""), type: str = Form(""), hub: str = Form(""),
    verification: str = Form("unverified"), min_confidence: str = Form(""),
    q: str = Form(""), sort: str = Form("confidence"),
):
    status = VerificationStatus.VERIFIED.value if action == "verify" else VerificationStatus.REJECTED.value
    store.set_verification_bulk(ids, status)
    paper = store.get_paper(paper_id) if paper_id else None
    filters = _filters_from_query(type, hub, verification, min_confidence, q, sort)
    return render("partials/_review_list.html", _render_review(request, paper=paper, filters=filters))


def _card_response(request: Request, concept_id: str):
    concept = store.get_concept(concept_id)
    edges = [_edge_view(e) for e in store.proposed_edges_for_concept(concept_id)]
    return render("partials/_concept_card.html",
                                      _ctx(request, card={"concept": concept, "edges": edges}))


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
    return render("partials/_concept_edit.html",
                                      _ctx(request, concept=store.get_concept(concept_id)))


@app.post("/concepts/{concept_id}", response_class=HTMLResponse)
def save_concept(
    request: Request, concept_id: str, corrected_title: str = Form(""), type: str = Form(""),
    suggested_hub: str = Form(""), statement_latex: str = Form(""), assumptions: str = Form(""),
    conclusion: str = Form(""), interpretation: str = Form(""), keywords: str = Form(""),
    reviewer_notes: str = Form(""),
):
    fields = dict(
        corrected_title=corrected_title.strip(), statement_latex=statement_latex,
        assumptions=assumptions, conclusion=conclusion, interpretation=interpretation,
        canonical_keywords=_parse_keywords(keywords), reviewer_notes=reviewer_notes,
    )
    if type:
        fields["type"] = type
    fields["suggested_hub"] = suggested_hub
    store.update_concept(concept_id, **fields)
    if request.headers.get("HX-Request") == "true":
        return _card_response(request, concept_id)
    return RedirectResponse(url=f"/concepts/{concept_id}", status_code=303)


@app.post("/concepts/{concept_id}/delete")
def delete_concept(concept_id: str):
    c = store.get_concept(concept_id)
    back = f"/papers/{c.paper_id}" if (c and c.paper_id) else "/brain"
    store.delete_concept(concept_id)
    return RedirectResponse(url=back, status_code=303)


@app.post("/concepts/{concept_id}/merge")
def merge_concept(concept_id: str, into_id: str = Form(...)):
    store.merge_concepts(concept_id, into_id)
    return RedirectResponse(url=f"/concepts/{into_id}", status_code=303)


# ── Edges ───────────────────────────────────────────────────────────────────────

def _edge_response(request: Request, edge_id: str):
    return render("partials/_edge.html", _ctx(request, ev=_edge_view(store.get_edge(edge_id))))


@app.post("/edges/{edge_id}/accept", response_class=HTMLResponse)
def accept_edge(request: Request, edge_id: str):
    store.set_edge_status(edge_id, EdgeStatus.VERIFIED.value)
    return _edge_response(request, edge_id)


@app.post("/edges/{edge_id}/reject", response_class=HTMLResponse)
def reject_edge(request: Request, edge_id: str):
    store.set_edge_status(edge_id, EdgeStatus.REJECTED.value)
    return _edge_response(request, edge_id)


@app.post("/concepts/{concept_id}/edges")
def add_edge(concept_id: str, target_id: str = Form(...), relation_type: str = Form("related"),
             rationale: str = Form("")):
    store.add_manual_edge(concept_id, target_id, relation_type, rationale)
    return RedirectResponse(url=f"/concepts/{concept_id}", status_code=303)


@app.post("/edges/{edge_id}/delete")
def delete_edge(edge_id: str, back: str = Form("/brain")):
    store.delete_edge(edge_id)
    return RedirectResponse(url=back, status_code=303)


# ── Second Brain ─────────────────────────────────────────────────────────────────

@app.get("/brain", response_class=HTMLResponse)
def brain(request: Request, q: str = "", type: str = "", hub: str = "", sort: str = "title"):
    concepts = store.query_concepts(
        states=[ConceptState.PROMOTED.value, ConceptState.HUB.value],
        type=type or None, hub=hub or None, q=q or None, sort=sort,
    )
    groups: dict[str, list] = {}
    for c in concepts:
        groups.setdefault(c.suggested_hub or "Uncategorized", []).append(c)
    grouped = sorted(groups.items(), key=lambda kv: kv[0].lower())
    return render("brain.html", _ctx(
        request, grouped=grouped, q=q, count=len(concepts), sort=sort,
        cur_type=type, cur_hub=hub, hubs=store.distinct_hubs(), types=store.distinct_types()))


@app.get("/concepts/{concept_id}", response_class=HTMLResponse)
def concept_detail(request: Request, concept_id: str):
    concept = store.get_concept(concept_id)
    if concept is None:
        return RedirectResponse(url="/brain", status_code=303)
    edges = store.edges_for_concept(concept_id)
    outgoing = [_edge_view(e) for e in edges if e.source_concept_id == concept_id]
    incoming = [_edge_view(e) for e in edges if e.target_concept_id == concept_id]
    paper = store.get_paper(concept.paper_id) if concept.paper_id else None
    others = [c for c in store.list_concepts() if c.id != concept_id]
    others.sort(key=lambda c: c.effective_title.lower())
    return render("concept_detail.html", _ctx(
        request, concept=concept, outgoing=outgoing, incoming=incoming, paper=paper, others=others))


# ── Graph / search ───────────────────────────────────────────────────────────────

@app.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request):
    return render("graph.html", _ctx(request))


@app.get("/api/graph.json")
def graph_json(verified_only: bool = True):
    return JSONResponse(store.graph_data(verified_only=verified_only))


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = ""):
    results = store.query_concepts(q=q, sort="confidence") if q.strip() else []
    return render("search.html", _ctx(request, q=q, results=results))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — organization, unified search, selective parsing/analysis, AI review
# ═══════════════════════════════════════════════════════════════════════════════


def _paper_orgs(paper_id: str) -> dict:
    """Tags / collections / projects (with roles) attached to a paper."""
    return {
        "tags": store.tags_for_paper(paper_id),
        "collections": store.collections_for_paper(paper_id),
        "projects": store.projects_for_paper(paper_id),
    }


# ── Projects ─────────────────────────────────────────────────────────────────────

@app.get("/projects", response_class=HTMLResponse)
def projects_page(request: Request):
    projects = sorted(store.list_projects(), key=lambda p: (-getattr(p, "priority", 0), p.name.lower()))
    rows = [{"project": p, "paper_count": len(store.papers_in_project(p.id))} for p in projects]
    return render("projects.html", _ctx(request, rows=rows))


@app.post("/projects")
def create_project(name: str = Form(...), description: str = Form(""), priority: str = Form("0")):
    try:
        prio = int(priority)
    except ValueError:
        prio = 0
    p = store.create_project(name=name, description=description, priority=prio)
    return RedirectResponse(url=f"/projects/{p.id}", status_code=303)


@app.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(request: Request, project_id: str):
    project = store.get_project(project_id)
    if project is None:
        return RedirectResponse(url="/projects", status_code=303)
    members = store.papers_in_project(project_id)
    by_role: dict[str, list] = {}
    for paper_id, role in members:
        paper = store.get_paper(paper_id)
        if paper is not None:
            by_role.setdefault(role, []).append(paper)
    grouped = sorted(by_role.items(), key=lambda kv: PAPER_ROLES.index(kv[0]) if kv[0] in PAPER_ROLES else 99)
    member_ids = {pid for pid, _ in members}
    candidates = [p for p in store.list_papers() if p.id not in member_ids]
    return render("project_detail.html", _ctx(
        request, project=project, grouped=grouped, total=len(members), candidates=candidates))


@app.post("/projects/{project_id}/update")
def update_project(project_id: str, name: str = Form(...), description: str = Form(""),
                   priority: str = Form("0"), status: str = Form("active")):
    try:
        prio = int(priority)
    except ValueError:
        prio = 0
    store.update_project(project_id, name=name, description=description, priority=prio, status=status)
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@app.post("/projects/{project_id}/papers")
def project_add_paper(project_id: str, paper_id: str = Form(...),
                      role: str = Form("maybe_relevant"), note: str = Form("")):
    store.add_paper_to_project(paper_id, project_id, role=role, note=note)
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@app.post("/projects/{project_id}/papers/{paper_id}/remove")
def project_remove_paper(project_id: str, paper_id: str):
    store.remove_paper_from_project(paper_id, project_id)
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@app.get("/projects/{project_id}/novelty", response_class=HTMLResponse)
def project_novelty(request: Request, project_id: str):
    """Phase-3 dashboard scaffold — aggregates reviewed signals for a project."""
    project = store.get_project(project_id)
    if project is None:
        return RedirectResponse(url="/projects", status_code=303)
    buckets: dict[str, list] = {}
    for paper_id, role in store.papers_in_project(project_id):
        paper = store.get_paper(paper_id)
        if paper is not None:
            buckets.setdefault(role, []).append(paper)
    math_objs = store.math_objects_for_project(project_id)
    return render("project_novelty.html", _ctx(
        request, project=project, buckets=buckets, math_objs=math_objs))


# ── Collections & tags ───────────────────────────────────────────────────────────

@app.get("/collections", response_class=HTMLResponse)
def collections_page(request: Request):
    return render("collections.html", _ctx(
        request, tree=store.collection_tree(), flat=store.list_collections()))


@app.post("/collections")
def create_collection(name: str = Form(...), description: str = Form(""), parent_id: str = Form("")):
    store.create_collection(name=name, description=description, parent_id=parent_id or None)
    return RedirectResponse(url="/collections", status_code=303)


@app.post("/collections/{collection_id}/delete")
def delete_collection(collection_id: str):
    store.delete_collection(collection_id)
    return RedirectResponse(url="/collections", status_code=303)


@app.get("/tags", response_class=HTMLResponse)
def tags_page(request: Request):
    rows = [{"tag": t, "count": len(store.papers_for_tag(t.id))} for t in store.list_tags()]
    return render("tags.html", _ctx(request, rows=rows))


@app.post("/tags")
def create_tag(name: str = Form(...), color: str = Form("")):
    store.create_tag(name=name, color=color)
    return RedirectResponse(url="/tags", status_code=303)


@app.post("/tags/{tag_id}/delete")
def delete_tag(tag_id: str):
    store.delete_tag(tag_id)
    return RedirectResponse(url="/tags", status_code=303)


# Assign organization from the paper detail page.
@app.post("/papers/{paper_id}/organize")
def organize_paper(paper_id: str, kind: str = Form(...), target_id: str = Form(""),
                   new_name: str = Form(""), role: str = Form("maybe_relevant")):
    if kind == "tag":
        tag_id = target_id or store.create_tag(new_name).id
        if tag_id:
            store.add_paper_tag(paper_id, tag_id)
    elif kind == "collection":
        col_id = target_id or (store.create_collection(new_name).id if new_name else "")
        if col_id:
            store.add_paper_to_collection(paper_id, col_id, role=role)
    elif kind == "project":
        proj_id = target_id or (store.create_project(new_name).id if new_name else "")
        if proj_id:
            store.add_paper_to_project(paper_id, proj_id, role=role)
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


@app.post("/papers/{paper_id}/organize/remove")
def organize_remove(paper_id: str, kind: str = Form(...), target_id: str = Form(...)):
    if kind == "tag":
        store.remove_paper_tag(paper_id, target_id)
    elif kind == "collection":
        store.remove_paper_from_collection(paper_id, target_id)
    elif kind == "project":
        store.remove_paper_from_project(paper_id, target_id)
    return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)


# ── Unified local search (papers + parsed chunks) ────────────────────────────────

@app.get("/find", response_class=HTMLResponse)
def find_page(request: Request, q: str = ""):
    results = store.search_papers(q) if q.strip() else []
    return render("find.html", _ctx(request, q=q, results=results))


# ── Cost estimate (AJAX for the scope builder) ───────────────────────────────────

@app.get("/api/cost/estimate")
def cost_estimate(pages: int = 0, analysis_type: str = "triage_summary", input_tokens: int = 0):
    marker = cost.estimate_marker_cost(pages, _CFG.marker_price_per_page)
    # ~600 tokens/page is a rough prose density if the caller didn't pass tokens.
    in_tok = input_tokens or pages * 600
    claude = cost.estimate_claude_cost(
        input_tokens=in_tok, analysis_type=analysis_type,
        input_price_per_mtok=_CFG.claude_input_price_per_mtok,
        output_price_per_mtok=_CFG.claude_output_price_per_mtok,
    )
    return JSONResponse({
        "pages": marker.pages, "marker_cost": marker.cost,
        "claude_input_tokens": claude.input_tokens,
        "claude_cost_low": claude.cost_low, "claude_cost_high": claude.cost_high,
        "total_low": round(marker.cost + claude.cost_low, 4),
        "total_high": round(marker.cost + claude.cost_high, 4),
    })


# ── Parse scopes / jobs ──────────────────────────────────────────────────────────

def _parse_ranges(raw: str) -> list[list[int]]:
    """Parse '1-8, 35-40, 12' into [[1,8],[35,40],[12,12]] (1-indexed)."""
    ranges: list[list[int]] = []
    for part in (raw or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            if a.isdigit() and b.isdigit():
                ranges.append([int(a), int(b)])
        elif part.isdigit():
            ranges.append([int(part), int(part)])
    return ranges


@app.get("/papers/{paper_id}/scope", response_class=HTMLResponse)
def scope_builder(request: Request, paper_id: str):
    paper = store.get_paper(paper_id)
    if paper is None:
        return RedirectResponse(url="/", status_code=303)
    return render("scope.html", _ctx(
        request, paper=paper, scopes=store.list_parse_scopes(paper_id),
        parse_jobs=store.list_parse_jobs(paper_id),
        analysis_jobs=store.list_analysis_jobs(paper_id),
        marker_price=_CFG.marker_price_per_page))


@app.post("/papers/{paper_id}/scopes")
def create_scope_and_parse(paper_id: str, name: str = Form("scope"),
                           purpose: str = Form("manual"), kind: str = Form("page_range"),
                           pages: str = Form(""), enqueue: str = Form("1")):
    ranges = _parse_ranges(pages)
    scope_json = {"kind": "full" if kind == "full" else ("page_range" if ranges else "full"),
                  "page_ranges": ranges, "regions": []}
    scope = store.create_parse_scope(paper_id, name=name, purpose=purpose, scope_json=scope_json)
    selected = scope_utils.page_count(scope_json) or len(ranges)
    if enqueue == "1":
        store.create_parse_job(
            paper_id=paper_id, parse_scope_id=scope.id, selected_pages=selected,
            cost_estimate=cost.estimate_marker_cost(selected, _CFG.marker_price_per_page).cost,
        )
    return RedirectResponse(url=f"/papers/{paper_id}/scope", status_code=303)


@app.post("/papers/{paper_id}/analyze")
def enqueue_analysis(paper_id: str, analysis_type: str = Form("triage_summary"),
                     project_id: str = Form(""), instruction: str = Form("")):
    chunks = store.chunks_for_paper(paper_id)
    chunk_ids = [c.id for c in chunks]
    est_tokens = sum(c.token_estimate for c in chunks)
    est = cost.estimate_claude_cost(
        input_tokens=est_tokens, analysis_type=analysis_type,
        input_price_per_mtok=_CFG.claude_input_price_per_mtok,
        output_price_per_mtok=_CFG.claude_output_price_per_mtok,
    )
    store.create_analysis_job(
        paper_id=paper_id, project_id=project_id or None, analysis_type=analysis_type,
        chunk_ids=chunk_ids, instruction=instruction,
        input_token_estimate=est.input_tokens, output_token_estimate=est.output_tokens,
        cost_estimate=est.cost_mid,
    )
    return RedirectResponse(url=f"/papers/{paper_id}/scope", status_code=303)


# ── AI suggestion review (quarantine) + regeneration ─────────────────────────────

def _suggestion_view(sug) -> dict:
    versions = store.suggestion_versions(sug.id)
    return {"sug": sug, "versions": versions, "is_latest": versions[-1].id == sug.id if versions else True}


@app.get("/suggestions", response_class=HTMLResponse)
def suggestions_page(request: Request, paper_id: str = "", type: str = "", status: str = "pending"):
    sugs = store.list_suggestions(
        paper_id=paper_id or None, suggestion_type=type or None, status=status or None)
    cards = [_suggestion_view(s) for s in sugs]
    paper = store.get_paper(paper_id) if paper_id else None
    return render("suggestions.html", _ctx(
        request, cards=cards, paper=paper, cur_type=type, cur_status=status,
        suggestion_types=[t.value for t in SuggestionType]))


def _suggestion_card(request: Request, suggestion_id: str):
    sug = store.get_suggestion(suggestion_id)
    if sug is None:
        return HTMLResponse("")
    return render("partials/_suggestion.html", _ctx(request, card=_suggestion_view(sug)))


@app.post("/suggestions/{suggestion_id}/accept", response_class=HTMLResponse)
def accept_suggestion(request: Request, suggestion_id: str, payload: str = Form("")):
    edited = None
    if payload.strip():
        try:
            import json
            edited = json.loads(payload)
        except ValueError:
            edited = None
    store.accept_suggestion(suggestion_id, edited_payload=edited)
    return _suggestion_card(request, suggestion_id)


@app.post("/suggestions/{suggestion_id}/reject", response_class=HTMLResponse)
def reject_suggestion(request: Request, suggestion_id: str):
    store.reject_suggestion(suggestion_id)
    return _suggestion_card(request, suggestion_id)


@app.post("/suggestions/{suggestion_id}/regenerate", response_class=HTMLResponse)
def regenerate_suggestion(request: Request, suggestion_id: str, instruction: str = Form("")):
    store.regenerate_suggestion(suggestion_id, instruction=instruction)
    # The new version is produced asynchronously by the analysis worker; re-render
    # the card so the reviewer sees the queued state + lineage.
    return _suggestion_card(request, suggestion_id)


@app.get("/health")
def health():
    return {"status": "ok"}

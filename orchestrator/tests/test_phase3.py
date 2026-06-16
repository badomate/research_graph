"""Tests for Phase-3 aggregation, graph, version lineage, and SerpAPI parsing."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import external_search as es
from modules.store import (
    PaperRole,
    Store,
    SuggestionStatus,
    SuggestionType,
    make_engine,
)


@pytest.fixture()
def store(tmp_path):
    st = Store(make_engine(f"sqlite:///{tmp_path / 'test.db'}"))
    st.create_all()
    return st


# ── Novelty aggregation ──────────────────────────────────────────────────────────


def test_project_novelty_groups_papers_and_accepted_suggestions(store):
    proj = store.create_project("MFG-UQ")
    competitor = store.create_paper(title="Competitor")
    tool = store.create_paper(title="Tool")
    store.add_paper_to_project(competitor.id, proj.id, role=PaperRole.DIRECT_COMPETITOR.value)
    store.add_paper_to_project(tool.id, proj.id, role=PaperRole.THEORY_TOOL.value)

    # An accepted novelty-risk suggestion + a pending one (pending must NOT show).
    s1 = store.create_suggestion(paper_id=competitor.id, suggestion_type=SuggestionType.NOVELTY_RISK.value,
                                 payload_json={"risk_level": "high", "assessment": "big overlap"})
    store.accept_suggestion(s1.id)
    store.create_suggestion(paper_id=competitor.id, suggestion_type=SuggestionType.LIMITATION.value,
                            payload_json={"text": "pending"})  # stays pending

    data = store.project_novelty(proj.id)
    assert data["paper_count"] == 2
    assert [p.id for p in data["papers_by_role"]["direct_competitor"]] == [competitor.id]
    assert len(data["suggestions"].get("novelty_risk", [])) == 1
    assert "limitation" not in data["suggestions"]   # pending excluded


def test_project_novelty_includes_linked_math_objects(store):
    proj = store.create_project("P")
    paper = store.create_paper(title="X")
    store.add_paper_to_project(paper.id, proj.id, role=PaperRole.CORE.value)
    mo = store.create_math_object(paper_id=paper.id, type="theorem", title="Key thm")
    store.link_math_object_project(mo.id, proj.id, "competing_result")
    data = store.project_novelty(proj.id)
    assert data["math_objects"][0][0].id == mo.id
    assert data["math_objects"][0][1] == "competing_result"


# ── Project knowledge graph ──────────────────────────────────────────────────────


def test_project_graph_nodes_and_edges(store):
    proj = store.create_project("P")
    paper = store.create_paper(title="A paper")
    store.add_paper_to_project(paper.id, proj.id, role=PaperRole.CORE.value)
    mo = store.create_math_object(paper_id=paper.id, type="theorem", title="Thm")
    g = store.project_graph(proj.id)
    ids = {n["id"] for n in g["nodes"]}
    assert f"proj:{proj.id}" in ids
    assert f"paper:{paper.id}" in ids
    assert f"mo:{mo.id}" in ids
    # project→paper and paper→math_object edges exist.
    pairs = {(e["from"], e["to"]) for e in g["edges"]}
    assert (f"proj:{proj.id}", f"paper:{paper.id}") in pairs
    assert (f"paper:{paper.id}", f"mo:{mo.id}") in pairs


# ── Version lineage: accept retires others; restore reverts ──────────────────────


def _chain_of_three(store):
    p = store.create_paper(title="P")
    v1 = store.create_suggestion(paper_id=p.id, suggestion_type=SuggestionType.SUMMARY.value,
                                 payload_json={"text": "v1"})
    v2 = store.create_suggestion(paper_id=p.id, suggestion_type=SuggestionType.SUMMARY.value,
                                 payload_json={"text": "v2"}, parent_generation_id=v1.id)
    v3 = store.create_suggestion(paper_id=p.id, suggestion_type=SuggestionType.SUMMARY.value,
                                 payload_json={"text": "v3"}, parent_generation_id=v2.id)
    return v1, v2, v3


def test_accept_retires_all_other_versions(store):
    v1, v2, v3 = _chain_of_three(store)
    store.accept_suggestion(v3.id)
    assert store.get_suggestion(v3.id).status == SuggestionStatus.ACCEPTED.value
    assert store.get_suggestion(v1.id).status == SuggestionStatus.SUPERSEDED.value
    assert store.get_suggestion(v2.id).status == SuggestionStatus.SUPERSEDED.value


def test_restore_older_version_flips_accepted(store):
    v1, v2, v3 = _chain_of_three(store)
    store.accept_suggestion(v3.id)
    store.restore_suggestion_version(v1.id)
    assert store.get_suggestion(v1.id).status == SuggestionStatus.ACCEPTED.value
    assert store.get_suggestion(v3.id).status == SuggestionStatus.SUPERSEDED.value


# ── SerpAPI parser + key-gated source list ───────────────────────────────────────


def test_parse_serpapi_scholar():
    payload = {"organic_results": [{
        "title": "Adjoint methods in mean field games",
        "snippet": "We develop adjoint methods…",
        "link": "https://example.org/a",
        "publication_info": {"summary": "J Doe, P Lions - SIAM Journal, 2023 - siam.org"},
        "resources": [{"file_format": "PDF", "link": "https://example.org/a.pdf"}],
    }]}
    res = es.parse_serpapi_scholar(payload)
    assert res[0].title.startswith("Adjoint methods")
    assert res[0].year == "2023"
    assert res[0].pdf_url.endswith(".pdf")
    assert res[0].authors == "J Doe, P Lions"


def test_available_sources_gated_on_serpapi_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    assert "serpapi" not in es.available_sources()
    monkeypatch.setenv("SERPAPI_KEY", "x")
    assert "serpapi" in es.available_sources()

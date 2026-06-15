"""Tests for the SQLite Store repository (the Notion replacement)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.store import (
    ConceptState,
    EdgeStatus,
    PaperStatus,
    Store,
    VerificationStatus,
    make_engine,
    normalize_title,
)


@pytest.fixture()
def store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    st = Store(engine)
    st.create_all()
    return st


def test_paper_crud_and_status_query(store):
    p = store.create_paper(title="A Paper", status=PaperStatus.S1_SKIM.value)
    assert store.get_paper(p.id).title == "A Paper"

    skim = store.get_papers_by_status(PaperStatus.S1_SKIM.value)
    assert [x.id for x in skim] == [p.id]

    store.set_paper_status(p.id, PaperStatus.S2_EXTRACTED.value)
    assert store.get_paper(p.id).status == "s2-extracted"
    assert store.get_papers_by_status(PaperStatus.S1_SKIM.value) == []


def test_paper_json_columns_round_trip(store):
    p = store.create_paper(
        title="JSON",
        active_themes=["mfg", "control"],
        rejected_concepts=[{"title": "x", "reasons": ["too short"]}],
    )
    got = store.get_paper(p.id)
    assert got.active_themes == ["mfg", "control"]
    assert got.rejected_concepts[0]["reasons"] == ["too short"]


def test_find_paper_by_external(store):
    store.create_paper(title="arxiv one", arxiv_id="2401.00001")
    hit = store.find_paper_by_external(arxiv_id="2401.00001")
    assert hit is not None and hit.title == "arxiv one"
    assert store.find_paper_by_external(arxiv_id="nope") is None


def test_concept_crud_and_verification(store):
    paper = store.create_paper(title="P")
    c = store.create_concept(
        paper_id=paper.id, title="Existence of MFG equilibrium",
        type="Theorem", canonical_keywords=["mean field game", "existence"],
        ai_confidence=0.9,
    )
    assert store.concepts_for_paper(paper.id)[0].id == c.id
    assert store.get_concept(c.id).canonical_keywords == ["mean field game", "existence"]

    store.set_verification(c.id, VerificationStatus.VERIFIED.value)
    assert store.get_concept(c.id).verification_status == "verified"


def test_promote_dedups_by_effective_title(store):
    p1 = store.create_paper(title="P1")
    p2 = store.create_paper(title="P2")
    a = store.create_concept(paper_id=p1.id, title="Banach Fixed Point Theorem")
    b = store.create_concept(paper_id=p2.id, title="banach fixed point theorem")

    promoted_a = store.promote_concept(a.id)
    assert promoted_a.state == ConceptState.PROMOTED.value

    # Same effective title → returns the already-promoted concept, not a 2nd node.
    promoted_b = store.promote_concept(b.id)
    assert promoted_b.id == a.id


def test_corrected_title_drives_promotion(store):
    p = store.create_paper(title="P")
    c = store.create_concept(paper_id=p.id, title="Thm 3.1", corrected_title="Lasry-Lions Monotonicity")
    promoted = store.promote_concept(c.id)
    assert promoted.id == c.id
    assert store.find_promoted_by_title("Lasry-Lions Monotonicity").id == c.id


def test_edge_status_transitions(store):
    p = store.create_paper(title="P")
    a = store.create_concept(paper_id=p.id, title="A")
    b = store.create_concept(paper_id=p.id, title="B")
    e = store.create_edge(
        source_concept_id=a.id, target_concept_id=b.id,
        relation_type="depends_on", status=EdgeStatus.PROPOSED.value,
    )
    assert len(store.proposed_edges_for_concept(a.id)) == 1
    store.set_edge_status(e.id, EdgeStatus.VERIFIED.value)
    assert store.get_edge(e.id).needs_review is False
    assert [x.id for x in store.list_edges(status=EdgeStatus.VERIFIED.value)] == [e.id]
    assert store.proposed_edges_for_concept(a.id) == []


def test_resolve_deferred_edges(store):
    p = store.create_paper(title="P")
    src = store.create_concept(paper_id=p.id, title="Source")
    target = store.create_concept(paper_id=p.id, title="Itô's Lemma")
    store.promote_concept(target.id)

    deferred = store.create_edge(
        source_concept_id=src.id, target_concept_id=None,
        target_title_raw="Itô's Lemma", deferred=True,
    )
    assert store.resolve_deferred_edges() == 1
    resolved = store.get_edge(deferred.id)
    assert resolved.deferred is False
    assert resolved.target_concept_id == target.id


def test_graph_data_verified_only(store):
    p = store.create_paper(title="P")
    a = store.create_concept(paper_id=p.id, title="A")
    b = store.create_concept(paper_id=p.id, title="B")
    store.promote_concept(a.id)
    store.promote_concept(b.id)
    store.create_edge(
        source_concept_id=a.id, target_concept_id=b.id,
        relation_type="enables", status=EdgeStatus.VERIFIED.value,
    )
    # A proposed edge must NOT appear in the verified-only graph.
    store.create_edge(
        source_concept_id=b.id, target_concept_id=a.id,
        relation_type="related", status=EdgeStatus.PROPOSED.value,
    )
    g = store.graph_data(verified_only=True)
    assert len(g["nodes"]) == 2
    assert len(g["edges"]) == 1
    assert g["edges"][0]["relation_type"] == "enables"


def test_normalize_title():
    assert normalize_title("Itô's Lemma") == normalize_title("Itôs Lemma")
    assert normalize_title("  Foo   Bar ") == "foo bar"

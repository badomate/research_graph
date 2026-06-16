"""Tests for chunk semantic search (TF-IDF fallback + resolution to SQLite rows)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import semantic_search as ss
from modules.store import Store, make_engine


def test_tfidf_rank_orders_by_relevance():
    docs = [
        "mean field games with common noise and the master equation",
        "a study of convolutional neural networks for images",
        "uncertainty quantification for stochastic gradient descent",
    ]
    ranked = ss.tfidf_rank("mean field game master equation", docs)
    assert ranked[0][0] == 0                 # the MFG doc ranks first
    assert all(s > 0 for _, s in ranked)


def test_tfidf_rank_empty_inputs():
    assert ss.tfidf_rank("", ["a"]) == []
    assert ss.tfidf_rank("a", []) == []


def test_tfidf_drops_irrelevant_docs():
    ranked = ss.tfidf_rank("adjoint method pde", ["adjoint method for pdes", "completely unrelated text"])
    returned = {i for i, _ in ranked}
    assert 0 in returned          # relevant doc present
    # the unrelated doc shares no tokens → score 0 → dropped
    assert 1 not in returned


@pytest.fixture()
def store(tmp_path):
    st = Store(make_engine(f"sqlite:///{tmp_path / 'test.db'}"))
    st.create_all()
    return st


def test_semantic_search_resolves_to_paper_and_chunk(store, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)   # force TF-IDF fallback
    p1 = store.create_paper(title="MFG paper")
    p2 = store.create_paper(title="Vision paper")
    j1 = store.create_parse_job(paper_id=p1.id)
    j2 = store.create_parse_job(paper_id=p2.id)
    store.add_chunk(paper_id=p1.id, parse_job_id=j1.id, ordinal=0, heading="Results",
                    text="We prove existence for a mean field game with common noise.", content_hash="a")
    store.add_chunk(paper_id=p2.id, parse_job_id=j2.id, ordinal=0, heading="Method",
                    text="A convolutional neural network classifies images.", content_hash="b")

    results = ss.semantic_search(store, "mean field game existence")
    assert results, "expected at least one semantic hit"
    top = results[0]
    assert top["paper"].id == p1.id
    assert top["chunk"].heading == "Results"
    assert top["backend"] == "tfidf"
    assert top["score"] > 0


def test_semantic_search_empty_when_no_chunks(store):
    store.create_paper(title="unparsed")
    assert ss.semantic_search(store, "anything") == []

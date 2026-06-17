"""Tests for debug-console status aggregation."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import debug_ops
from modules.store import EdgeStatus, Store, make_engine


@pytest.fixture()
def store(tmp_path):
    st = Store(make_engine(f"sqlite:///{tmp_path / 'test.db'}"))
    st.create_all()
    return st


def test_system_stats_counts(store):
    p = store.create_paper(title="P", status="s2-extracted")
    a = store.create_concept(paper_id=p.id, title="A")
    b = store.create_concept(paper_id=p.id, title="B")
    store.create_edge(source_concept_id=a.id, target_concept_id=b.id,
                      relation_type="related", status=EdgeStatus.PROPOSED.value)
    store.create_parse_job(paper_id=p.id)
    store.create_suggestion(paper_id=p.id, suggestion_type="summary", payload_json={"text": "x"})
    store.create_math_object(paper_id=p.id, type="theorem", title="T")

    stats = debug_ops.system_stats(store)
    assert stats["papers"] == 1
    assert stats["papers_by_status"]["s2-extracted"] == 1
    assert stats["concepts"] == 2
    assert stats["edges"] == 1
    assert stats["edges_by_status"]["proposed"] == 1
    assert stats["parse_jobs"]["pending"] == 1
    assert stats["suggestions"]["pending"] == 1
    assert stats["math_objects"] == 1


def test_system_stats_empty(store):
    stats = debug_ops.system_stats(store)
    assert stats["papers"] == 0 and stats["concepts"] == 0 and stats["edges"] == 0

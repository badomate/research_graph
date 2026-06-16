"""Tests for the Phase-1 research Store layer + analysis worker lineage."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.store import (
    JobStatus,
    PaperRole,
    ScopePurpose,
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


# ── Organization ────────────────────────────────────────────────────────────────


def test_tags_are_idempotent_and_linkable(store):
    p = store.create_paper(title="P")
    t1 = store.create_tag("mfg")
    t2 = store.create_tag("mfg")          # same name → same row
    assert t1.id == t2.id
    store.add_paper_tag(p.id, t1.id)
    store.add_paper_tag(p.id, t1.id)      # idempotent
    assert [t.name for t in store.tags_for_paper(p.id)] == ["mfg"]
    store.remove_paper_tag(p.id, t1.id)
    assert store.tags_for_paper(p.id) == []


def test_nested_collections_and_delete_reparents(store):
    root = store.create_collection("UQ")
    child = store.create_collection("MFG-UQ", parent_id=root.id)
    grand = store.create_collection("drafts", parent_id=child.id)
    tree = store.collection_tree()
    assert tree[0]["collection"].id == root.id
    assert tree[0]["children"][0]["collection"].id == child.id

    # Deleting the middle node reparents its child up to root.
    store.delete_collection(child.id)
    assert store.get_collection(child.id) is None
    assert store.get_collection(grand.id).parent_id == root.id


def test_paper_belongs_to_multiple_groups_with_roles(store):
    p = store.create_paper(title="P")
    proj = store.create_project("MFG-UQ")
    col = store.create_collection("Competitors")
    store.add_paper_to_project(p.id, proj.id, role=PaperRole.DIRECT_COMPETITOR.value, note="overlaps thm 2")
    store.add_paper_to_collection(p.id, col.id, role=PaperRole.BASELINE.value)

    projs = store.projects_for_paper(p.id)
    assert projs[0][1] == "direct_competitor"
    cols = store.collections_for_paper(p.id)
    assert cols[0][1] == "baseline"
    assert store.papers_in_project(proj.id, role="direct_competitor") == [(p.id, "direct_competitor")]

    # Role is updatable in place.
    store.add_paper_to_project(p.id, proj.id, role=PaperRole.CORE.value)
    assert store.projects_for_paper(p.id)[0][1] == "core"


# ── Parse scopes / jobs / artifacts / chunks ────────────────────────────────────


def test_parse_job_dedup_by_input_hash(store):
    p = store.create_paper(title="P")
    scope = store.create_parse_scope(
        p.id, "intro", ScopePurpose.TRIAGE.value, {"kind": "page_range", "page_ranges": [[1, 8]]}
    )
    j1 = store.create_parse_job(paper_id=p.id, parse_scope_id=scope.id, input_hash="abc:123",
                                status=JobStatus.SUCCEEDED.value)
    assert store.find_parse_job_by_input_hash("abc:123").id == j1.id
    # Pending/failed jobs of the same hash are not considered a cache hit.
    assert store.find_parse_job_by_input_hash("nope") is None


def test_claim_next_parse_job_is_exclusive(store):
    p = store.create_paper(title="P")
    store.create_parse_job(paper_id=p.id)
    job = store.claim_next_parse_job()
    assert job.status == JobStatus.RUNNING.value and job.attempts == 1
    assert store.claim_next_parse_job() is None   # nothing left pending


def test_chunks_round_trip_and_selection(store):
    p = store.create_paper(title="P")
    job = store.create_parse_job(paper_id=p.id)
    rows = store.add_chunks([
        dict(paper_id=p.id, parse_job_id=job.id, ordinal=0, heading="Intro", text="a", content_hash="h0"),
        dict(paper_id=p.id, parse_job_id=job.id, ordinal=1, heading="Thm", text="b", content_hash="h1"),
    ])
    ids = [r.id for r in rows]
    assert [c.heading for c in store.get_chunks(ids)] == ["Intro", "Thm"]
    assert [c.heading for c in store.get_chunks(list(reversed(ids)))] == ["Thm", "Intro"]


# ── AI-suggestion quarantine + promotion ────────────────────────────────────────


def test_accept_math_object_suggestion_promotes_and_keeps_provenance(store):
    p = store.create_paper(title="P")
    sug = store.create_suggestion(
        paper_id=p.id, suggestion_type=SuggestionType.THEOREM.value,
        payload_json={"title": "Existence of MFG equilibrium", "statement_latex": "$u$",
                      "conclusion": "exists", "confidence": 0.8},
    )
    accepted = store.accept_suggestion(sug.id)
    assert accepted.status == SuggestionStatus.ACCEPTED.value
    assert accepted.promoted_ref_table == "math_objects"
    mo = store.get_math_object(accepted.promoted_ref_id)
    assert mo.title == "Existence of MFG equilibrium"
    assert mo.source_suggestion_id == sug.id   # provenance back to the suggestion


def test_edit_then_accept_uses_edited_payload(store):
    p = store.create_paper(title="P")
    sug = store.create_suggestion(
        paper_id=p.id, suggestion_type=SuggestionType.MATH_OBJECT.value,
        payload_json={"title": "raw", "type": "definition"},
    )
    accepted = store.accept_suggestion(sug.id, edited_payload={"title": "edited", "type": "definition"})
    assert store.get_math_object(accepted.promoted_ref_id).title == "edited"


def test_reject_keeps_row_for_provenance(store):
    p = store.create_paper(title="P")
    sug = store.create_suggestion(paper_id=p.id, suggestion_type=SuggestionType.SUMMARY.value,
                                  payload_json={"text": "x"})
    store.reject_suggestion(sug.id)
    assert store.get_suggestion(sug.id).status == SuggestionStatus.REJECTED.value


def test_project_link_suggestion_promotes_to_membership(store):
    p = store.create_paper(title="P")
    proj = store.create_project("MFG-UQ")
    sug = store.create_suggestion(
        paper_id=p.id, project_id=proj.id, suggestion_type=SuggestionType.PROJECT_LINK.value,
        payload_json={"role": "direct_competitor", "note": "same setting"},
    )
    store.accept_suggestion(sug.id)
    assert store.projects_for_paper(p.id)[0][1] == "direct_competitor"


# ── Regeneration lineage (end-to-end through the analysis worker) ────────────────


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeUsage:
    input_tokens = 1000
    output_tokens = 200


class _FakeResp:
    def __init__(self, payload):
        self.content = [_FakeBlock(json.dumps(payload))]
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, payload):
        self._payload = payload

    def create(self, **_kwargs):
        return _FakeResp(self._payload)


class _FakeClient:
    def __init__(self, payload):
        self.messages = _FakeMessages(payload)


def _worker(store, payload):
    from modules.analysis.analysis_worker import AnalysisWorker
    from modules.config import Config

    cfg = Config(anthropic_api_key="test-key")
    return AnalysisWorker(store, config=cfg, client=_FakeClient(payload))


def test_analysis_worker_quarantines_then_regenerates_with_lineage(store):
    p = store.create_paper(title="P")
    job = store.create_parse_job(paper_id=p.id)
    chunk = store.add_chunk(paper_id=p.id, parse_job_id=job.id, ordinal=0,
                            text="We prove existence.", content_hash="h0")

    # First generation: a triage summary.
    aj = store.create_analysis_job(paper_id=p.id, analysis_type="triage_summary", chunk_ids=[chunk.id])
    _worker(store, {"text": "v1 summary"}).run_job(aj.id)

    aj = store.get_analysis_job(aj.id)
    assert aj.status == JobStatus.SUCCEEDED.value
    assert aj.input_tokens_actual == 1000 and aj.cost_actual is not None
    sugs = store.list_suggestions(paper_id=p.id, suggestion_type=SuggestionType.SUMMARY.value)
    assert len(sugs) == 1 and sugs[0].status == SuggestionStatus.PENDING.value
    original = sugs[0]

    # Regenerate that suggestion with an instruction → a new pending job.
    regen_job = store.regenerate_suggestion(original.id, instruction="be more mathematical")
    assert regen_job.target_suggestion_id == original.id
    assert regen_job.instruction == "be more mathematical"
    assert regen_job.chunk_ids == [chunk.id]

    _worker(store, {"text": "v2 summary, more mathematical"}).run_job(regen_job.id)

    versions = store.suggestion_versions(original.id)
    assert [v.payload_json["text"] for v in versions] == ["v1 summary", "v2 summary, more mathematical"]
    new = versions[-1]
    assert new.parent_generation_id == original.id
    # The original is NOT superseded until the user accepts the new version.
    assert store.get_suggestion(original.id).status == SuggestionStatus.PENDING.value

    # Accept v2 → v1 becomes superseded.
    store.accept_suggestion(new.id)
    assert store.get_suggestion(original.id).status == SuggestionStatus.SUPERSEDED.value


# ── Math objects + local search ─────────────────────────────────────────────────


def test_math_object_project_relevance_link(store):
    p = store.create_paper(title="P")
    proj = store.create_project("MFG-UQ")
    mo = store.create_math_object(paper_id=p.id, type="theorem", title="Lasry-Lions monotonicity")
    store.link_math_object_project(mo.id, proj.id, "direct_tool")
    linked = store.math_objects_for_project(proj.id)
    assert linked[0][0].id == mo.id and linked[0][1] == "direct_tool"


def test_search_papers_matches_metadata_and_chunks(store):
    p1 = store.create_paper(title="Mean field games and control", authors="Lasry")
    p2 = store.create_paper(title="Unrelated topic")
    job = store.create_parse_job(paper_id=p2.id)
    store.add_chunk(paper_id=p2.id, parse_job_id=job.id, ordinal=0,
                    text="We study a mean field game with common noise.", content_hash="h")

    res = store.search_papers("mean field")
    ids = {r["paper"].id for r in res}
    assert p1.id in ids and p2.id in ids
    # The chunk-only hit reports it was matched in parsed text and is 'parsed'.
    p2_res = next(r for r in res if r["paper"].id == p2.id)
    assert p2_res["parsed"] is True
    assert "mean field" in p2_res["snippet"].lower()

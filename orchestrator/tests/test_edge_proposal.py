"""Tests for the rethought edge proposal: accepted-graph corpus + TF-IDF no-LLM path.

Covered units:
  - retriever tags TF-IDF candidates and honours restrict_to_ids (SQLite truth);
  - the linker routes pure-TF-IDF candidates to the no-LLM suggest path and routes
    real (cross-paper / same-paper) candidates to the LLM path.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.extraction_schema import CrossPaperLinkResult, MathObject
from modules.ingestion.linker import ConceptLinker
from modules.ingestion.retriever import CandidateRetriever
from modules.scoring.candidate_scorer import ConceptData


# ── Retriever: corpus + tagging ──────────────────────────────────────────────────


def _retriever():
    return CandidateRetriever(MagicMock(), vector_index=None, config=None)  # no Qdrant → TF-IDF


def _sb_index():
    return [
        {"id": "c1", "title": "Mean field game existence", "hub": "MFG", "summary": "",
         "keywords_bag": {"mean", "field", "game", "existence"}},
        {"id": "c2", "title": "Convolutional network", "hub": "ML", "summary": "",
         "keywords_bag": {"convolutional", "network"}},
    ]


def _concept():
    return MathObject(type="Theorem", title="Existence in mean field games",
                      statement_latex="u(x)=0", canonical_keywords=["mean field game", "existence"])


def test_tfidf_candidates_tagged_and_overlap_filtered():
    cands = _retriever().retrieve_candidates_for_concept(_concept(), _sb_index())
    ids = {c["id"] for c in cands}
    assert ids == {"c1"}                                  # only the overlapping concept
    assert all(c["_source"] == "tfidf" for c in cands)    # tagged for the linker


def test_restrict_to_ids_limits_corpus_to_accepted():
    # Accepted graph = {c2} only → the overlapping c1 is excluded (SQLite is truth).
    cands = _retriever().retrieve_candidates_for_concept(
        _concept(), _sb_index(), restrict_to_ids={"c2"})
    assert cands == []


def test_cand_id_handles_both_candidate_shapes():
    assert CandidateRetriever._cand_id({"id": "x"}) == "x"
    cd = ConceptData(notion_page_id="y", title="", concept_type="", statement_latex="",
                     assumptions="", conclusion="", setting=[], named_tools=[], keywords=[])
    assert CandidateRetriever._cand_id({"_concept_data": cd, "id": "ignored"}) == "y"


def test_restrict_is_noop_without_ids():
    cands = [{"id": "a"}, {"id": "b"}]
    assert CandidateRetriever._restrict(cands, None) == cands


# ── Linker: TF-IDF skip vs LLM path ──────────────────────────────────────────────


def _src():
    return MathObject(type="Theorem", title="Source", statement_latex="x=y")


def test_run_stage_link_skips_llm_for_tfidf_candidates():
    client = MagicMock()
    linker = ConceptLinker(client, config=None)
    candidates = [{"id": "a", "title": "A", "_source": "tfidf", "score": 0.3}]

    result = linker.run_stage_link(_src(), candidates, "r1")

    client.messages.create.assert_not_called()            # no LLM call
    assert isinstance(result, CrossPaperLinkResult)
    assert [p.target_notion_page_id for p in result.proposals] == ["a"]
    assert result.proposals[0].channel == "suggest"
    assert "no llm" in result.proposals[0].justification.lower()


def test_run_stage_link_routes_cross_paper_to_llm_not_tfidf():
    linker = ConceptLinker(MagicMock(), config=None)
    sentinel = CrossPaperLinkResult()
    linker._call_claude_link_v2 = lambda concept, candidates: sentinel  # type: ignore[assignment]
    cd = ConceptData(notion_page_id="t1", title="T", concept_type="Lemma", statement_latex="",
                     assumptions="", conclusion="", setting=[], named_tools=[], keywords=[])
    candidates = [{"id": "t1", "title": "T", "_concept_data": cd, "_score_obj": None}]

    out = linker.run_stage_link(_src(), candidates, "r1")
    assert out is sentinel                                 # cross-paper → v2 LLM path


def test_run_stage_link_routes_same_paper_to_v1_not_tfidf():
    linker = ConceptLinker(MagicMock(), config=None)
    from modules.extraction_schema import ConceptLinkResult
    sentinel = ConceptLinkResult()
    linker._call_claude_link_v1 = lambda concept, candidates: sentinel  # type: ignore[assignment]
    # Untagged, no _concept_data → a same-paper candidate (v1), not TF-IDF.
    out = linker.run_stage_link(_src(), [{"id": "s1", "title": "S1"}], "r1")
    assert out is sentinel


def test_batch_skips_llm_for_tfidf_items():
    linker = ConceptLinker(MagicMock(), config=None)
    submitted = {}
    linker._submit_and_poll = lambda requests, run_id: submitted.update(reqs=requests) or {}  # type: ignore
    concept_candidates = [(_src(), "ki-0", [{"id": "a", "_source": "tfidf"}])]

    results = linker.run_stage_link_batch(concept_candidates, "run-1")
    # TF-IDF item resolved locally (suggest-only), never submitted to the batch.
    assert submitted.get("reqs") in (None, [])
    assert isinstance(results["ki-0"], CrossPaperLinkResult)
    assert results["ki-0"].proposals[0].channel == "suggest"

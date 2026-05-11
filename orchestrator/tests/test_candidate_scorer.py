"""Tests for modules/scoring/candidate_scorer.py — pure scoring functions."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.scoring.candidate_scorer import (
    CandidateScore,
    ConceptData,
    _jaccard,
    _normalize_for_fuzzy,
    _tokenize_for_overlap,
    route_edge_proposals,
    score_candidate_pair,
)


def _concept(
    page_id="id-a",
    title="Gradient Descent",
    concept_type="Algorithm",
    statement_latex="",
    assumptions="convex function f",
    conclusion="converges to minimum",
    setting=None,
    named_tools=None,
    keywords=None,
) -> ConceptData:
    return ConceptData(
        notion_page_id=page_id,
        title=title,
        concept_type=concept_type,
        statement_latex=statement_latex,
        assumptions=assumptions,
        conclusion=conclusion,
        setting=setting or [],
        named_tools=named_tools or [],
        keywords=keywords or [],
    )


class TestJaccardAndHelpers:
    def test_jaccard_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_jaccard_disjoint_sets(self):
        assert _jaccard({"a"}, {"b"}) == pytest.approx(0.0)

    def test_jaccard_partial_overlap(self):
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_jaccard_empty_sets(self):
        assert _jaccard(set(), set()) == pytest.approx(0.0)

    def test_normalize_strips_punct_and_lowercases(self):
        assert _normalize_for_fuzzy("Lipschitz-Continuity!") == "lipschitz continuity"

    def test_tokenize_strips_latex(self):
        tokens = _tokenize_for_overlap(r"\alpha convex function")
        assert "convex" in tokens
        assert "function" in tokens
        # \alpha should be gone (replaced with space, then empty after strip)
        assert "alpha" not in tokens


class TestScoreCandidatePair:
    def test_returns_candidate_score_type(self):
        a = _concept()
        b = _concept(page_id="id-b", title="SGD")
        result = score_candidate_pair(a, b, qdrant_similarity=0.5)
        assert isinstance(result, CandidateScore)
        assert result.candidate_id == "id-b"

    def test_qdrant_similarity_is_stored(self):
        a = _concept()
        b = _concept(page_id="id-b")
        result = score_candidate_pair(a, b, qdrant_similarity=0.75)
        assert result.qdrant_similarity == pytest.approx(0.75)

    def test_matching_keywords_increase_score(self):
        shared_kw = ["optimization", "convex"]
        a = _concept(keywords=shared_kw)
        b = _concept(page_id="id-b", keywords=shared_kw)
        result_match = score_candidate_pair(a, b, qdrant_similarity=0.0)
        result_no_match = score_candidate_pair(
            _concept(keywords=[]), _concept(page_id="id-b", keywords=[]), qdrant_similarity=0.0
        )
        assert result_match.keyword_jaccard > result_no_match.keyword_jaccard

    def test_assumption_conclusion_overlap_detected(self):
        # A's assumptions overlap with B's conclusion
        a = _concept(assumptions="convex lipschitz function")
        b = _concept(page_id="id-b", conclusion="convex lipschitz bound")
        result = score_candidate_pair(a, b, qdrant_similarity=0.0)
        assert result.assumption_conclusion_overlap > 0.0

    def test_named_tool_match_prevents_drop(self):
        # If named_tool_match fires, should_drop must be False regardless of composite score
        a = _concept(named_tools=["SGD"])
        b = _concept(page_id="id-b", title="SGD")
        result = score_candidate_pair(a, b, qdrant_similarity=0.0)
        if result.named_tool_match:
            assert result.should_drop is False

    def test_zero_signals_low_composite(self):
        a = _concept(assumptions="", conclusion="", setting=[], named_tools=[], keywords=[])
        b = _concept(page_id="id-b", assumptions="", conclusion="", setting=[], named_tools=[], keywords=[])
        result = score_candidate_pair(a, b, qdrant_similarity=0.0)
        assert result.composite_score == pytest.approx(0.0)
        assert result.should_drop is True

    def test_composite_bounded_0_to_1(self):
        a = _concept(assumptions="f g h", conclusion="f g h", keywords=["opt"], setting=["R^d"])
        b = _concept(page_id="id-b", assumptions="f g h", conclusion="f g h",
                     keywords=["opt"], setting=["R^d"])
        result = score_candidate_pair(a, b, qdrant_similarity=1.0)
        assert 0.0 <= result.composite_score <= 1.0 + 1e-9


class TestRouteEdgeProposals:
    """route_edge_proposals with mock proposal objects."""

    class _FakeProposal:
        def __init__(self, confidence, channel, driving_fields=None, falsifiability="",
                     relation_type="depends_on", source_type=None, target_type=None):
            self.confidence = confidence
            self.channel = channel
            self.driving_fields = driving_fields or ["named_tools"]
            self.falsifiability = falsifiability
            self.relation_type = relation_type
            self.source_type = source_type
            self.target_type = target_type
            self.needs_review = False
            self.demoted_from_auto = False

    def test_low_confidence_auto_is_demoted(self):
        p = self._FakeProposal(confidence=0.5, channel="auto",
                               falsifiability="x " * 10)
        auto, suggest = route_edge_proposals([p], scores={})
        assert p in suggest
        assert p not in auto
        assert p.channel == "suggest"
        assert p.demoted_from_auto is True

    def test_high_confidence_with_valid_fields_and_falsifiability_stays_auto(self):
        p = self._FakeProposal(
            confidence=0.90,
            channel="auto",
            driving_fields=["named_tools"],
            falsifiability="If the bound does not hold then the theorem conclusion fails completely.",
            relation_type="depends_on",
        )
        auto, suggest = route_edge_proposals([p], scores={})
        assert p in auto
        assert p not in suggest
        assert p.needs_review is False

    def test_suggest_channel_below_threshold_is_excluded(self):
        p = self._FakeProposal(confidence=0.3, channel="suggest")
        auto, suggest = route_edge_proposals([p], scores={})
        assert p not in suggest
        assert p not in auto

    def test_suggest_channel_above_threshold_is_included(self):
        p = self._FakeProposal(confidence=0.6, channel="suggest")
        auto, suggest = route_edge_proposals([p], scores={})
        assert p in suggest
        assert p.needs_review is True

    def test_auto_cap_at_3(self):
        proposals = [
            self._FakeProposal(
                confidence=0.9,
                channel="auto",
                driving_fields=["named_tools"],
                falsifiability="word " * 10,
            )
            for _ in range(5)
        ]
        auto, _ = route_edge_proposals(proposals, scores={})
        assert len(auto) <= 3

    def test_suggest_cap_at_4(self):
        proposals = [
            self._FakeProposal(confidence=0.6, channel="suggest")
            for _ in range(6)
        ]
        _, suggest = route_edge_proposals(proposals, scores={})
        assert len(suggest) <= 4

    def test_empty_proposals(self):
        auto, suggest = route_edge_proposals([], scores={})
        assert auto == []
        assert suggest == []

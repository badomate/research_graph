"""Tests for the Stage-3 batch-linking helpers in ConceptLinker.

These exercise the pure request-building / parsing / dispatch logic without
touching the Anthropic Batches API (which is stubbed out in the test env).
"""
import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.extraction_schema import (
    ConceptLinkResult,
    CrossPaperLinkResult,
    MathObject,
)
from modules.ingestion.linker import ConceptLinker
from modules.scoring.candidate_scorer import ConceptData


def _linker() -> ConceptLinker:
    return ConceptLinker(MagicMock(), config=None)


def _concept(title="Source Theorem") -> MathObject:
    return MathObject(type="Theorem", title=title, statement_latex="x = y for all admissible x")


def _cross_candidate(page_id="tgt-1") -> dict:
    cd = ConceptData(
        notion_page_id=page_id,
        title="Target Lemma",
        concept_type="Lemma",
        statement_latex="",
        assumptions="",
        conclusion="",
        setting=[],
        named_tools=[],
        keywords=[],
    )
    return {"id": page_id, "title": "Target Lemma", "_concept_data": cd, "_score_obj": None}


def _tool_message(input_dict: dict):
    block = types.SimpleNamespace(type="tool_use", input=input_dict)
    return types.SimpleNamespace(content=[block])


def test_structured_tool_exposes_model_schema():
    tool = ConceptLinker._structured_tool(ConceptLinkResult)
    assert tool["name"] == "record_links"
    assert tool["description"]
    props = tool["input_schema"]["properties"]
    for key in ("depends_on", "enables", "generalizes", "special_case_of", "related"):
        assert key in props


def test_parse_tool_message_validates_into_model():
    edge = {
        "source_concept_title": "A",
        "target_concept_title": "B",
        "target_notion_page_id": "tgt-1",
        "relation_type": "related",
        "direction": "A_to_B",
        "confidence": 0.9,
        "justification": "shares the same operator",
    }
    msg = _tool_message({"proposals": [edge], "low_confidence_suggestions": []})
    parsed = ConceptLinker._parse_tool_message(msg, CrossPaperLinkResult)
    assert isinstance(parsed, CrossPaperLinkResult)
    assert len(parsed.proposals) == 1


def test_parse_tool_message_empty_when_no_tool_block():
    msg = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="nope")])
    parsed = ConceptLinker._parse_tool_message(msg, ConceptLinkResult)
    assert isinstance(parsed, ConceptLinkResult)
    assert parsed.depends_on == []


def test_build_batch_params_v2_forces_the_tool_and_caches_system():
    params = _linker()._build_link_batch_params(_concept(), [_cross_candidate()], "v2")
    assert params["tool_choice"] == {"type": "tool", "name": "record_links"}
    assert len(params["tools"]) == 1
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][0]["role"] == "user"


def test_run_stage_link_batch_resolves_local_and_batched_concepts():
    linker = _linker()

    # Item 0: no candidates → local empty ConceptLinkResult (no batch request).
    # Item 1: TF-IDF only -> legacy v1 batch request.
    # Item 2: cross-paper -> v2 batch request.
    concept_candidates = [
        (_concept("c0"), "ki-0", []),
        (_concept("c1"), "ki-1", [{"id": "x", "title": "X"}]),
        (_concept("c2"), "ki-2", [_cross_candidate("tgt-2")]),
    ]

    edge = {
        "source_concept_title": "c2",
        "target_concept_title": "Target Lemma",
        "target_notion_page_id": "tgt-2",
        "relation_type": "related",
        "direction": "A_to_B",
        "confidence": 0.9,
        "justification": "shares the same fixed-point argument",
    }
    captured = {}

    def fake_submit(requests, run_id):
        captured["requests"] = requests
        return {
            "link_1": _tool_message({}),
            "link_2": _tool_message({"proposals": [edge], "low_confidence_suggestions": []}),
        }

    linker._submit_and_poll = fake_submit  # type: ignore[assignment]

    results = linker.run_stage_link_batch(concept_candidates, "run-xyz")

    assert [r["custom_id"] for r in captured["requests"]] == ["link_1", "link_2"]

    assert isinstance(results["ki-0"], ConceptLinkResult)
    assert results["ki-0"].depends_on == []

    assert isinstance(results["ki-1"], ConceptLinkResult)
    assert results["ki-1"].depends_on == []

    assert isinstance(results["ki-2"], CrossPaperLinkResult)
    assert len(results["ki-2"].proposals) == 1
    # Routing annotated the source type from the concept.
    assert results["ki-2"].proposals[0].source_type == "Theorem"


def test_run_stage_link_batch_skips_missing_results():
    linker = _linker()
    concept_candidates = [(_concept("c0"), "ki-0", [_cross_candidate("tgt-0")])]
    linker._submit_and_poll = lambda requests, run_id: {}  # type: ignore[assignment]
    results = linker.run_stage_link_batch(concept_candidates, "run-xyz")
    # No result for the only concept → key absent, not an empty/garbage value.
    assert "ki-0" not in results

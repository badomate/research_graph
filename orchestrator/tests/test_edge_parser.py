"""Tests for modules/promotion/edge_parser.py — all three Edge Suggestions formats."""
import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.promotion.edge_parser import parse_edge_suggestions


def _props(json_value: str) -> dict:
    """Wrap a JSON string as a Notion rich_text property dict."""
    return {
        "Edge Suggestions": {
            "rich_text": [{"plain_text": json_value}]
        }
    }


def _empty_props() -> dict:
    return {}


KI_PAGE_ID = "test-ki-page-001"


class TestFormat1CrossPaperLinkResult:
    """Format 1: {proposals: [...]} — new CrossPaperLinkResult."""

    def test_basic_proposal(self):
        data = {
            "proposals": [{
                "target_concept_title": "Lipschitz Continuity",
                "relation_type": "depends_on",
                "justification": "Uses Lipschitz bound in proof.",
                "confidence": 0.92,
                "needs_review": False,
                "driving_fields": ["assumptions"],
                "pre_filter_signal": "named_tool_match",
                "target_notion_page_id": "abc123",
                "channel": "auto",
                "falsifiability": "If the bound does not hold then the theorem fails.",
            }]
        }
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 1
        e = edges[0]
        assert e["relation_type"] == "depends_on"
        assert e["target_title"] == "Lipschitz Continuity"
        assert e["confidence"] == pytest.approx(0.92)
        assert e["channel"] == "auto"
        assert e["target_notion_page_id"] == "abc123"
        assert e["driving_fields"] == ["assumptions"]
        assert e["falsifiability"] == "If the bound does not hold then the theorem fails."

    def test_falls_back_to_target_title_key(self):
        data = {"proposals": [{"target_title": "Banach Space", "relation_type": "generalizes",
                                "confidence": 0.8, "channel": "auto"}]}
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 1
        assert edges[0]["target_title"] == "Banach Space"

    def test_skips_entries_without_target_title(self):
        data = {"proposals": [{"relation_type": "related", "confidence": 0.7}]}
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert edges == []

    def test_multiple_proposals(self):
        data = {"proposals": [
            {"target_concept_title": "A", "relation_type": "depends_on", "confidence": 0.9},
            {"target_concept_title": "B", "relation_type": "enables",    "confidence": 0.8},
        ]}
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 2
        assert {e["target_title"] for e in edges} == {"A", "B"}


class TestFormat2LegacyConceptLinkResult:
    """Format 2: {relation_type: [{target_title, rationale, confidence}]} dict."""

    def test_basic_dict_entries(self):
        data = {
            "depends_on": [
                {"target_title": "Convex Set", "rationale": "proof step", "confidence": 0.85}
            ],
            "enables": [
                {"target_title": "SGD Convergence", "rationale": "by lemma", "confidence": 0.70}
            ],
        }
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 2
        by_type = {e["relation_type"]: e for e in edges}
        assert by_type["depends_on"]["target_title"] == "Convex Set"
        assert by_type["depends_on"]["confidence"] == pytest.approx(0.85)
        assert by_type["depends_on"]["rationale"] == "proof step"
        assert by_type["enables"]["target_title"] == "SGD Convergence"

    def test_string_shorthand_entries(self):
        data = {"related": ["Gradient Descent", "Momentum"]}
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 2
        titles = {e["target_title"] for e in edges}
        assert titles == {"Gradient Descent", "Momentum"}
        for e in edges:
            assert e["confidence"] == 0.0
            assert e["channel"] == "auto"

    def test_skips_entries_without_target_title(self):
        data = {"depends_on": [{"rationale": "no title", "confidence": 0.9}]}
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert edges == []

    def test_non_list_value_is_skipped(self):
        data = {"depends_on": "not a list"}
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert edges == []


class TestFormat3LegacyFlatList:
    """Format 3: [{relation_type, target_name, ...}] flat list."""

    def test_target_name_key(self):
        data = [
            {"relation_type": "generalizes", "target_name": "Metric Space", "confidence": 0.75},
        ]
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 1
        assert edges[0]["target_title"] == "Metric Space"
        assert edges[0]["relation_type"] == "generalizes"

    def test_target_title_fallback(self):
        data = [{"relation_type": "related", "target_title": "Hilbert Space"}]
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 1
        assert edges[0]["target_title"] == "Hilbert Space"

    def test_defaults_relation_type_to_related(self):
        data = [{"target_name": "Norm", "confidence": 0.6}]
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert edges[0]["relation_type"] == "related"

    def test_skips_entries_without_target(self):
        data = [{"relation_type": "depends_on"}]
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert edges == []


class TestEdgeParserRobustness:
    def test_empty_props_returns_empty(self):
        assert parse_edge_suggestions(_empty_props(), KI_PAGE_ID) == []

    def test_empty_string_returns_empty(self):
        assert parse_edge_suggestions(_props(""), KI_PAGE_ID) == []

    def test_invalid_json_returns_empty(self):
        assert parse_edge_suggestions(_props("{not valid json}"), KI_PAGE_ID) == []

    def test_null_json_returns_empty(self):
        assert parse_edge_suggestions(_props("null"), KI_PAGE_ID) == []

    def test_number_json_returns_empty(self):
        assert parse_edge_suggestions(_props("42"), KI_PAGE_ID) == []

    def test_empty_proposals_list(self):
        data = {"proposals": []}
        assert parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID) == []

    def test_non_dict_entries_in_flat_list_are_skipped(self):
        data = ["not a dict", {"target_name": "Real", "relation_type": "related"}]
        edges = parse_edge_suggestions(_props(json.dumps(data)), KI_PAGE_ID)
        assert len(edges) == 1
        assert edges[0]["target_title"] == "Real"

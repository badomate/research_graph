"""
modules/promotion/edge_parser.py — Edge Suggestions JSON parser.

Pure function with no I/O. Handles all three historical formats written by
the ingestion pipeline's Stage 3 LLM linking step.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def parse_edge_suggestions(props: dict, ki_page_id: str) -> list[dict]:
    """
    Parse the Edge Suggestions JSON property into a flat list of edge dicts.

    Handles three formats:

    1. New CrossPaperLinkResult format (Stage 3 v2):
       {"proposals": [{source_concept_title, target_concept_title,
                       target_notion_page_id, relation_type, direction,
                       confidence, justification, driving_fields, ...}]}

    2. Legacy ConceptLinkResult dict format (Stage 3 v1):
       {"depends_on": [{"target_concept_id": "...", "target_title": "...",
                        "rationale": "...", "confidence": 0.9}], ...}

    3. Legacy flat-list format (very old):
       [{"relation_type": "...", "target_name": "..."}]

    Returns a list of dicts with at least:
      relation_type, target_title, rationale, confidence,
      needs_review, driving_fields, pre_filter_signal,
      justification, target_notion_page_id, channel.
    """
    raw = _get_text(props, "Edge Suggestions")
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "EdgeParser: invalid Edge Suggestions JSON on page %s", ki_page_id
        )
        return []

    edges: list[dict] = []

    # Format 1: New CrossPaperLinkResult
    if isinstance(parsed, dict) and "proposals" in parsed:
        for entry in parsed.get("proposals", []):
            if not isinstance(entry, dict):
                continue
            target_title = (
                entry.get("target_concept_title", "").strip()
                or entry.get("target_title", "").strip()
            )
            if not target_title:
                continue
            edges.append({
                "relation_type":         entry.get("relation_type", "related"),
                "target_title":          target_title,
                "rationale":             entry.get("justification", ""),
                "justification":         entry.get("justification", ""),
                "confidence":            float(entry.get("confidence", 0.0)),
                "needs_review":          bool(entry.get("needs_review", False)),
                "driving_fields":        entry.get("driving_fields", []),
                "pre_filter_signal":     entry.get("pre_filter_signal", ""),
                "target_notion_page_id": entry.get("target_notion_page_id", ""),
                "channel":               entry.get("channel", "auto"),
                "falsifiability":        entry.get("falsifiability", ""),
            })
        return edges

    # Format 2: Legacy ConceptLinkResult dict
    if isinstance(parsed, dict):
        for rel_type, targets in parsed.items():
            if not isinstance(targets, list):
                continue
            for entry in targets:
                if isinstance(entry, str):
                    t = entry.strip()
                    if t:
                        edges.append({
                            "relation_type": rel_type,
                            "target_title": t,
                            "rationale": "",
                            "justification": "",
                            "confidence": 0.0,
                            "needs_review": False,
                            "driving_fields": [],
                            "pre_filter_signal": "",
                            "target_notion_page_id": "",
                            "channel": "auto",
                        })
                elif isinstance(entry, dict):
                    target_title = entry.get("target_title", "").strip()
                    rationale    = entry.get("rationale", "")
                    confidence   = float(entry.get("confidence", 0.0))
                    if target_title:
                        edges.append({
                            "relation_type":         rel_type,
                            "target_title":          target_title,
                            "rationale":             rationale,
                            "justification":         rationale,
                            "confidence":            confidence,
                            "needs_review":          False,
                            "driving_fields":        [],
                            "pre_filter_signal":     "",
                            "target_notion_page_id": entry.get("target_concept_id", ""),
                            "channel":               "auto",
                        })
        return edges

    # Format 3: Legacy flat-list
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            rel_type   = entry.get("relation_type", "related")
            target     = entry.get("target_name", "") or entry.get("target_title", "")
            rationale  = entry.get("rationale", "")
            confidence = float(entry.get("confidence", 0.0))
            if target:
                edges.append({
                    "relation_type":         rel_type,
                    "target_title":          target.strip(),
                    "rationale":             rationale,
                    "justification":         rationale,
                    "confidence":            confidence,
                    "needs_review":          False,
                    "driving_fields":        [],
                    "pre_filter_signal":     "",
                    "target_notion_page_id": "",
                    "channel":               "auto",
                })

    return edges


def _get_text(props: dict, key: str) -> str:
    try:
        segments = props[key]["rich_text"]
        return "".join(seg.get("plain_text", "") for seg in segments)
    except (KeyError, TypeError):
        return ""

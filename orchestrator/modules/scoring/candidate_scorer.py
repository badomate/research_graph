"""
modules/scoring/candidate_scorer.py — Cross-paper candidate pre-filter scoring.

Pure functions with no I/O. All thresholds read from env at import time
(Config injection planned for Phase 5).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# -- Thresholds ----------------------------------------------------------------
NAMED_TOOL_MATCH_THRESHOLD: int = int(os.environ.get("NAMED_TOOL_MATCH_THRESHOLD", "85"))
SETTING_CONTAINMENT_THRESHOLD: int = int(os.environ.get("SETTING_CONTAINMENT_THRESHOLD", "80"))
ASSUMPTION_OVERLAP_DROP_THRESHOLD: float = float(
    os.environ.get("ASSUMPTION_OVERLAP_DROP_THRESHOLD", "0.05")
)
KEYWORD_JACCARD_DROP_THRESHOLD: float = float(
    os.environ.get("KEYWORD_JACCARD_DROP_THRESHOLD", "0.10")
)
QDRANT_SIMILARITY_DROP_THRESHOLD: float = float(
    os.environ.get("QDRANT_SIMILARITY_DROP_THRESHOLD", "0.75")
)

# -- Composite score weights (must sum to 1.0) ---------------------------------
WEIGHT_QDRANT: float = float(os.environ.get("WEIGHT_QDRANT", "0.40"))
WEIGHT_NAMED_TOOL: float = float(os.environ.get("WEIGHT_NAMED_TOOL", "0.25"))
WEIGHT_ASSUMPTION_OVERLAP: float = float(os.environ.get("WEIGHT_ASSUMPTION_OVERLAP", "0.20"))
WEIGHT_SETTING_CONTAINMENT: float = float(os.environ.get("WEIGHT_SETTING_CONTAINMENT", "0.10"))
WEIGHT_KEYWORD_JACCARD: float = float(os.environ.get("WEIGHT_KEYWORD_JACCARD", "0.05"))
COMPOSITE_DROP_THRESHOLD = 0.12

# -- Edge creation thresholds --------------------------------------------------
EDGE_AUTO_CREATE_CONFIDENCE: float = float(
    os.environ.get("EDGE_AUTO_CREATE_CONFIDENCE", "0.80")
)
EDGE_REVIEW_FLAG_CONFIDENCE: float = float(
    os.environ.get("EDGE_REVIEW_FLAG_CONFIDENCE", "0.65")
)


@dataclass
class ConceptData:
    """Fully-hydrated concept data fetched from a Notion page."""

    notion_page_id: str
    title: str
    concept_type: str
    statement_latex: str
    assumptions: str
    conclusion: str
    setting: list
    named_tools: list
    keywords: list


@dataclass
class CandidateScore:
    """Structural pre-filter scores for a single (C_A, C_B) candidate pair."""

    candidate_id: str
    qdrant_similarity: float
    named_tool_match: bool
    assumption_conclusion_overlap: float
    setting_containment: Optional[str]  # "A_in_B" | "B_in_A" | None
    keyword_jaccard: float
    composite_score: float
    should_drop: bool


# -- Normalisation helpers -----------------------------------------------------

_PUNCT_RE = re.compile(r'[^\w\s]')
_WS_RE = re.compile(r'\s+')


def _normalize_for_fuzzy(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(' ', s)
    s = _WS_RE.sub(' ', s).strip()
    return s


_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+')
_TOKEN_SEP_RE = re.compile(r'[\s\{\}\[\]\(\)\$,;:\.\|]+')


def _tokenize_for_overlap(text: str) -> set:
    text = text.lower()
    text = _LATEX_CMD_RE.sub(' ', text)
    tokens = _TOKEN_SEP_RE.split(text)
    return {t for t in tokens if t and len(t) > 1}


def _jaccard(s1: set, s2: set) -> float:
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


# -- Core scoring --------------------------------------------------------------

def score_candidate_pair(
    concept_a: ConceptData,
    concept_b: ConceptData,
    qdrant_similarity: float,
) -> CandidateScore:
    """
    Compute four structural signals for a (C_A, C_B) pair.

    Signal 1 — Named Tool Match (rapidfuzz token_sort_ratio, threshold 85).
    Signal 2 — Assumption-Conclusion Overlap (token Jaccard, max of both directions).
    Signal 3 — Setting Containment (rapidfuzz partial_ratio, threshold 80).
    Signal 4 — Keyword Jaccard (exact match after normalisation).
    """
    try:
        from rapidfuzz import fuzz as _fuzz
    except ImportError:
        logger.warning(
            "score_candidate_pair: rapidfuzz not installed — returning neutral score."
        )
        composite = WEIGHT_QDRANT * qdrant_similarity
        return CandidateScore(
            candidate_id=concept_b.notion_page_id,
            qdrant_similarity=qdrant_similarity,
            named_tool_match=False,
            assumption_conclusion_overlap=0.0,
            setting_containment=None,
            keyword_jaccard=0.0,
            composite_score=composite,
            should_drop=(qdrant_similarity < QDRANT_SIMILARITY_DROP_THRESHOLD),
        )

    # Signal 1: Named Tool Match
    a_title_norm = _normalize_for_fuzzy(concept_a.title)
    b_title_norm = _normalize_for_fuzzy(concept_b.title)
    named_tool_match = False

    for tool in concept_a.named_tools:
        if _fuzz.token_sort_ratio(_normalize_for_fuzzy(tool), b_title_norm) >= NAMED_TOOL_MATCH_THRESHOLD:
            named_tool_match = True
            break
    if not named_tool_match:
        for tool in concept_b.named_tools:
            if _fuzz.token_sort_ratio(_normalize_for_fuzzy(tool), a_title_norm) >= NAMED_TOOL_MATCH_THRESHOLD:
                named_tool_match = True
                break

    # Signal 2: Assumption-Conclusion Overlap
    a_assumptions = _tokenize_for_overlap(concept_a.assumptions)
    b_conclusion = _tokenize_for_overlap(concept_b.conclusion)
    b_assumptions = _tokenize_for_overlap(concept_b.assumptions)
    a_conclusion = _tokenize_for_overlap(concept_a.conclusion)
    assumption_conclusion_overlap = max(_jaccard(a_assumptions, b_conclusion), _jaccard(b_assumptions, a_conclusion))

    # Signal 3: Setting Containment
    setting_containment: Optional[str] = None
    a_setting_str = " ".join(concept_a.setting) if isinstance(concept_a.setting, list) else str(concept_a.setting or "")
    b_setting_str = " ".join(concept_b.setting) if isinstance(concept_b.setting, list) else str(concept_b.setting or "")
    if a_setting_str.strip() and b_setting_str.strip():
        ratio_a_in_b = _fuzz.partial_ratio(a_setting_str.lower(), b_setting_str.lower())
        ratio_b_in_a = _fuzz.partial_ratio(b_setting_str.lower(), a_setting_str.lower())
        if ratio_a_in_b >= SETTING_CONTAINMENT_THRESHOLD:
            setting_containment = "A_in_B"
        elif ratio_b_in_a >= SETTING_CONTAINMENT_THRESHOLD:
            setting_containment = "B_in_A"

    # Signal 4: Keyword Jaccard
    kw_a = {k.lower().strip() for k in concept_a.keywords if k}
    kw_b = {k.lower().strip() for k in concept_b.keywords if k}
    keyword_jaccard = _jaccard(kw_a, kw_b)

    # Composite score
    composite_score = (
        WEIGHT_QDRANT * qdrant_similarity
        + WEIGHT_NAMED_TOOL * float(named_tool_match)
        + WEIGHT_ASSUMPTION_OVERLAP * assumption_conclusion_overlap
        + WEIGHT_SETTING_CONTAINMENT * float(setting_containment is not None)
        + WEIGHT_KEYWORD_JACCARD * keyword_jaccard
    )

    if named_tool_match:
        should_drop = False
    else:
        should_drop = (
            assumption_conclusion_overlap < ASSUMPTION_OVERLAP_DROP_THRESHOLD
            and setting_containment is None
            and keyword_jaccard < KEYWORD_JACCARD_DROP_THRESHOLD
            and qdrant_similarity < QDRANT_SIMILARITY_DROP_THRESHOLD
        )

    return CandidateScore(
        candidate_id=concept_b.notion_page_id,
        qdrant_similarity=qdrant_similarity,
        named_tool_match=named_tool_match,
        assumption_conclusion_overlap=assumption_conclusion_overlap,
        setting_containment=setting_containment,
        keyword_jaccard=keyword_jaccard,
        composite_score=composite_score,
        should_drop=should_drop,
    )


def _dominant_signal(score: CandidateScore) -> str:
    """Return the name of the highest-firing pre-filter signal, or 'none'."""
    if score.named_tool_match:
        return "named_tool_match"
    if score.assumption_conclusion_overlap >= ASSUMPTION_OVERLAP_DROP_THRESHOLD:
        return "assumption_conclusion_overlap"
    if score.setting_containment is not None:
        return "setting_containment"
    if score.keyword_jaccard >= KEYWORD_JACCARD_DROP_THRESHOLD:
        return "keyword_jaccard"
    return "none"


def _assign_review_flag(proposal, score: CandidateScore):
    """Determine whether an edge needs human review. Mutates proposal in place."""
    has_structural_signal = (
        score.named_tool_match
        or score.assumption_conclusion_overlap >= 0.10
        or score.setting_containment is not None
    )
    fields_are_grounded = len(proposal.driving_fields) >= 1

    if not fields_are_grounded:
        proposal.needs_review = True
        return proposal

    high_confidence = proposal.confidence >= EDGE_AUTO_CREATE_CONFIDENCE
    medium_confidence = EDGE_REVIEW_FLAG_CONFIDENCE <= proposal.confidence < EDGE_AUTO_CREATE_CONFIDENCE

    if high_confidence and has_structural_signal:
        proposal.needs_review = False
    else:
        proposal.needs_review = True

    return proposal


def _relation_type_valid(p) -> bool:
    """Enforce the relation type constraint table from the dual-channel prompt."""
    constraints: dict[tuple[str, str], set[str]] = {
        ("Theorem",    "Theorem"):    {"depends_on", "generalizes", "special_case_of"},
        ("Theorem",    "Definition"): {"depends_on"},
        ("Theorem",    "Lemma"):      {"depends_on"},
        ("Definition", "Definition"): {"generalizes", "special_case_of"},
        ("Definition", "Theorem"):    set(),
        ("Lemma",      "Theorem"):    {"enables"},
        ("Algorithm",  "Theorem"):    {"depends_on"},
        ("Assumption", "Theorem"):    {"enables"},
    }
    if p.source_type is None or p.target_type is None:
        return True
    key = (p.source_type, p.target_type)
    allowed = constraints.get(
        key,
        {"depends_on", "enables", "generalizes", "special_case_of", "related"},
    )
    return p.relation_type in allowed


def route_edge_proposals(
    proposals: list,
    scores: dict,
) -> tuple:
    """
    Apply hard validation on top of GPT's channel assignment.

    GPT can be demoted from auto → suggest but never promoted.
    Returns (auto_edges, suggest_edges). Count caps: ≤3 auto, ≤4 suggest.
    """
    auto_edges: list = []
    suggest_edges: list = []

    for p in proposals:
        if p.channel == "auto":
            valid = (
                p.confidence >= 0.75
                and any(f in p.driving_fields for f in ["named_tools", "assumptions", "conclusion"])
                and len(p.falsifiability.split()) >= 8
                and _relation_type_valid(p)
            )
            if not valid:
                p.channel = "suggest"
                p.needs_review = True
                p.demoted_from_auto = True
                suggest_edges.append(p)
            else:
                p.needs_review = False
                auto_edges.append(p)
        else:
            if p.confidence >= 0.50:
                p.needs_review = True
                suggest_edges.append(p)

    auto_edges = sorted(auto_edges, key=lambda x: x.confidence, reverse=True)[:3]
    suggest_edges = sorted(suggest_edges, key=lambda x: x.confidence, reverse=True)[:4]

    return auto_edges, suggest_edges

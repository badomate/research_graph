# modules/ingestion - Core Ingestion Engine package
from .extractor import is_dense_paper
from ..notion_client_wrapper import NotionClientWrapper
from . import engine as _engine_module
from ..scoring.candidate_scorer import (
    EDGE_AUTO_CREATE_CONFIDENCE,
    EDGE_REVIEW_FLAG_CONFIDENCE,
    CandidateScore,
    ConceptData,
    _assign_review_flag,
    _dominant_signal,
    _jaccard,
    _normalize_for_fuzzy,
    _relation_type_valid,
    _tokenize_for_overlap,
    route_edge_proposals,
    score_candidate_pair,
)

anthropic = _engine_module.anthropic
instructor = _engine_module.instructor

def __getattr__(name: str):
    if name == "IngestionEngine":
        from .engine import IngestionEngine

        return IngestionEngine
    raise AttributeError(name)


__all__ = [
    "IngestionEngine",
    "is_dense_paper",
    "EDGE_AUTO_CREATE_CONFIDENCE",
    "EDGE_REVIEW_FLAG_CONFIDENCE",
    "CandidateScore",
    "ConceptData",
    "_assign_review_flag",
    "_dominant_signal",
    "_jaccard",
    "_normalize_for_fuzzy",
    "_relation_type_valid",
    "_tokenize_for_overlap",
    "route_edge_proposals",
    "score_candidate_pair",
]

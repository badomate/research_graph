"""
modules/ingestion/retriever.py — Stage 2 candidate retrieval service.

Retrieves top-k candidate concepts via Qdrant (with pre-filter scoring)
or TF-IDF fallback.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Dict

from ..config import Config
from ..extraction_schema import MathObject
from ..store import Store
from ..scoring.candidate_scorer import (
    CandidateScore,
    ConceptData,
    _dominant_signal,
    score_candidate_pair,
)

logger = logging.getLogger(__name__)

RETRIEVE_CANDIDATES_K: int = int(os.environ.get("RETRIEVE_CANDIDATES_K", "30"))
EDGE_MAX_CANDIDATES_TO_GPT: int = int(os.environ.get("EDGE_MAX_CANDIDATES_TO_GPT", "20"))
NOTION_HYDRATION_CONCURRENCY: int = int(os.environ.get("NOTION_HYDRATION_CONCURRENCY", "5"))
NOTION_BLOCK_MAX_CHARS = 1900

_RELATION_CANDIDATE_MAP: dict[str, str] = {
    "depends_on":    "depends_on",
    "enables":       "enables",
    "related":       "related",
    "special_case_of": "related",
    "generalizes":   "related",
}


class CandidateRetriever:
    """Retrieves and scores candidate concepts for Stage 2."""

    def __init__(
        self,
        store: Store,
        vector_index=None,
        config: Config | None = None,
    ) -> None:
        self.store = store
        self._vector_index = vector_index
        self._retrieve_candidates_k = (
            config.retrieve_candidates_k if config is not None else RETRIEVE_CANDIDATES_K
        )
        self._edge_max_candidates_to_gpt = (
            config.edge_max_candidates_to_gpt
            if config is not None
            else EDGE_MAX_CANDIDATES_TO_GPT
        )
        self._hydration_concurrency = (
            config.notion_hydration_concurrency
            if config is not None
            else NOTION_HYDRATION_CONCURRENCY
        )

    def retrieve_candidates_for_concept(
        self,
        concept: MathObject,
        sb_index: list[dict],
        k: int | None = None,
        current_page_id: str | None = None,
        same_paper_ids: set | None = None,
    ) -> list[dict]:
        """
        Return top-k candidate concepts for linking.

        Uses Qdrant + pre-filter scoring when available; falls back to TF-IDF.
        Same-paper candidates bypass the pre-filter.
        """
        k = k or self._retrieve_candidates_k
        if not (self._vector_index and self._vector_index.available):
            return self._tfidf_retrieve(concept, sb_index, k)

        hints = self._vector_index.retrieve_candidates(concept, verified_only=False)
        if current_page_id:
            hints = [h for h in hints if h.notion_page_id != current_page_id]

        same_paper_ids = same_paper_ids or set()
        same_paper_hints = [h for h in hints if h.notion_page_id in same_paper_ids]
        cross_paper_hints = [h for h in hints if h.notion_page_id not in same_paper_ids]

        cross_paper_dicts: list[dict] = []
        if cross_paper_hints:
            cross_ids = [h.notion_page_id for h in cross_paper_hints]
            logger.debug(
                "Pre-filter: hydrating %d cross-paper candidate(s) for '%s'.",
                len(cross_ids), concept.title,
            )
            hydrated = self.hydrate_candidates(cross_ids)

            concept_a_data = ConceptData(
                notion_page_id=current_page_id or "",
                title=concept.title,
                concept_type=concept.type,
                statement_latex=concept.statement_latex,
                assumptions=concept.assumptions or "",
                conclusion=concept.conclusion or "",
                setting=list(concept.setting) if concept.setting else [],
                named_tools=list(concept.named_tools) if concept.named_tools else [],
                keywords=list(concept.canonical_keywords) if concept.canonical_keywords else [],
            )

            scored: list[tuple[float, dict]] = []
            n_before = len(cross_paper_hints)
            dropped = 0

            for hint in cross_paper_hints:
                concept_b_data = hydrated.get(hint.notion_page_id)
                if concept_b_data is None:
                    d = hint.to_dict()
                    d["_pre_filter_signal"] = "none"
                    scored.append((hint.score, d))
                    continue

                score = score_candidate_pair(concept_a_data, concept_b_data, hint.score)
                if score.should_drop:
                    dropped += 1
                    continue

                d = hint.to_dict()
                d["_concept_data"] = concept_b_data
                d["_pre_filter_signal"] = _dominant_signal(score)
                d["_score_obj"] = score
                scored.append((score.composite_score, d))

            logger.debug(
                "Pre-filter '%s': %d → %d candidate(s) (%d dropped).",
                concept.title, n_before, len(scored), dropped,
            )
            scored.sort(key=lambda x: x[0], reverse=True)
            cross_paper_dicts = [d for _, d in scored[:self._edge_max_candidates_to_gpt]]

        same_paper_dicts = [h.to_dict() for h in same_paper_hints]
        return same_paper_dicts + cross_paper_dicts

    def _tfidf_retrieve(
        self, concept: MathObject, sb_index: list[dict], k: int = RETRIEVE_CANDIDATES_K
    ) -> list[dict]:
        def _toks(s: str) -> set:
            return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

        concept_tokens: set = set()
        for kw_list in (concept.canonical_keywords, concept.prereq_keywords, concept.downstream_keywords):
            for kw in kw_list:
                concept_tokens |= _toks(kw)
        concept_tokens |= _toks(concept.title)

        scored: list[tuple[float, dict]] = []
        for record in sb_index:
            bag = record.get("keywords_bag", set())
            overlap = len(concept_tokens & bag)
            score = overlap / math.log(1.0 + len(bag))
            if record.get("hub") and record["hub"] == concept.suggested_hub:
                score += 0.2
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "hub": r.get("hub", ""),
                "summary": r.get("summary", ""),
                "score": round(float(score), 4),
            }
            for score, r in scored[:k]
        ]

    def hydrate_candidates(self, candidate_ids: list[str]) -> Dict[str, ConceptData]:
        """Load full ConceptData for candidate concept IDs from the Store."""
        results: Dict[str, ConceptData] = {}
        for cid in candidate_ids:
            c = self.store.get_concept(cid)
            if c is None:
                continue
            results[cid] = ConceptData(
                notion_page_id=c.id,
                title=c.effective_title or "(unknown)",
                concept_type=c.type or "Definition",
                statement_latex=c.statement_latex,
                assumptions=c.assumptions,
                conclusion=c.conclusion or c.interpretation,
                setting=list(c.setting or []),
                named_tools=list(c.named_tools or []),
                keywords=list(c.canonical_keywords or []),
            )
        return results

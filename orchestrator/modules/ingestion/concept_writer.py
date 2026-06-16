"""
modules/ingestion/concept_writer.py — writes extracted concepts and edges to the Store.

Replaces the Notion-page/block builder (ki_writer.py). A MathObject becomes one
`concepts` row (state=inbox); Stage-3 link results become `edges` rows
(status=proposed) — the review UI flips those to verified/rejected.
"""
from __future__ import annotations

import logging

from ..extraction_schema import (
    ConceptLinkResult,
    CrossPaperLinkResult,
    MathObject,
)
from ..store import Store

logger = logging.getLogger(__name__)


class ConceptWriter:
    """Persists concepts + their proposed edges via the Store."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def create_concept_row(
        self,
        paper_id: str,
        concept: MathObject,
        flag_reasons: list[str] | None = None,
    ) -> str:
        row = self.store.create_concept(
            paper_id=paper_id,
            type=concept.type,
            title=concept.title,
            statement_latex=concept.statement_latex,
            assumptions=concept.assumptions,
            variables=concept.variables,
            conclusion=concept.conclusion,
            interpretation=concept.interpretation,
            proof_idea=concept.proof_idea,
            source_quote=concept.source_quotes or "",
            source_pages=list(concept.source_pages or []),
            source_anchors=concept.source_anchors,
            aliases=concept.aliases,
            suggested_hub=concept.suggested_hub or "",
            result_category=concept.result_category,
            named_tools=list(concept.named_tools or []),
            setting=list(concept.setting or []),
            canonical_keywords=list(concept.canonical_keywords or []),
            prereq_keywords=list(concept.prereq_keywords or []),
            downstream_keywords=list(concept.downstream_keywords or []),
            ai_confidence=concept.confidence,
            flag_reasons=list(flag_reasons or []),
        )
        logger.info("Created concept %s for '%s'.", row.id, concept.title)
        return row.id

    def write_edges(
        self,
        source_concept_id: str,
        link_result: ConceptLinkResult | CrossPaperLinkResult,
    ) -> int:
        """Create proposed edge rows from a Stage-3 link result. Returns count."""
        created = 0
        if isinstance(link_result, CrossPaperLinkResult):
            for p in link_result.proposals:
                if not p.target_notion_page_id:
                    continue
                self.store.create_edge(
                    source_concept_id=source_concept_id,
                    target_concept_id=p.target_notion_page_id,
                    relation_type=p.relation_type,
                    direction=p.direction,
                    channel=p.channel,
                    ai_confidence=p.confidence,
                    justification=p.justification,
                    rationale=p.justification,
                    falsifiability=p.falsifiability,
                    driving_fields=list(p.driving_fields or []),
                    needs_review=p.needs_review,
                    demoted_from_auto=p.demoted_from_auto,
                )
                created += 1
        else:  # ConceptLinkResult — legacy/same-paper TF-IDF path
            for rel in ("depends_on", "enables", "generalizes", "special_case_of", "related"):
                for e in getattr(link_result, rel, []):
                    if not e.target_concept_id:
                        continue
                    self.store.create_edge(
                        source_concept_id=source_concept_id,
                        target_concept_id=e.target_concept_id,
                        relation_type=rel,
                        channel="suggest",
                        ai_confidence=e.confidence,
                        justification=e.rationale,
                        rationale=e.rationale,
                        needs_review=True,
                    )
                    created += 1

        status = "linked-ai" if created else "unlinked"
        self.store.update_concept(source_concept_id, graph_link_status=status)
        return created

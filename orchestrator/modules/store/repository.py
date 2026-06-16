"""
modules/store/repository.py — the Store: typed data-access layer.

Every pipeline module and the web app go through this class instead of talking to
Notion. Methods open a short-lived session, commit, and return detached rows
(sessions use expire_on_commit=False so the returned objects stay readable).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy.engine import Engine
from sqlmodel import select

from .db import get_engine, init_db, new_session
from .models import (
    Concept,
    ConceptState,
    Edge,
    EdgeStatus,
    Paper,
    PaperStatus,
    Project,
    VerificationStatus,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def normalize_title(title: str) -> str:
    """Lower-case, strip math/punctuation — used for promotion + edge dedup."""
    t = (title or "").lower()
    t = re.sub(r"\$[^$]*\$", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


class Store:
    """Repository over the SQLite database."""

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()

    def create_all(self) -> None:
        init_db(self._engine)

    # ── Papers ─────────────────────────────────────────────────────────────────

    def create_paper(self, **fields: Any) -> Paper:
        paper = Paper(**fields)
        with new_session(self._engine) as s:
            s.add(paper)
            s.commit()
            s.refresh(paper)
        return paper

    def get_paper(self, paper_id: str) -> Paper | None:
        with new_session(self._engine) as s:
            return s.get(Paper, paper_id)

    def list_papers(self) -> list[Paper]:
        with new_session(self._engine) as s:
            return list(s.exec(select(Paper).order_by(Paper.created_at.desc())))

    def get_papers_by_status(self, status: str) -> list[Paper]:
        with new_session(self._engine) as s:
            return list(s.exec(select(Paper).where(Paper.status == status)))

    def update_paper(self, paper_id: str, **fields: Any) -> Paper | None:
        with new_session(self._engine) as s:
            paper = s.get(Paper, paper_id)
            if paper is None:
                return None
            for key, value in fields.items():
                setattr(paper, key, value)
            paper.updated_at = _now()
            s.add(paper)
            s.commit()
            s.refresh(paper)
            return paper

    def set_paper_status(self, paper_id: str, status: str) -> Paper | None:
        return self.update_paper(paper_id, status=status)

    def find_paper_by_external(
        self, *, zotero_key: str = "", arxiv_id: str = "", doi: str = ""
    ) -> Paper | None:
        """De-dup intake: match an existing paper by any external identifier."""
        with new_session(self._engine) as s:
            for field, value in (
                (Paper.zotero_key, zotero_key),
                (Paper.arxiv_id, arxiv_id),
                (Paper.doi, doi),
            ):
                if value:
                    hit = s.exec(select(Paper).where(field == value)).first()
                    if hit:
                        return hit
        return None

    # ── Concepts ───────────────────────────────────────────────────────────────

    def create_concept(self, **fields: Any) -> Concept:
        concept = Concept(**fields)
        with new_session(self._engine) as s:
            s.add(concept)
            s.commit()
            s.refresh(concept)
        return concept

    def get_concept(self, concept_id: str) -> Concept | None:
        with new_session(self._engine) as s:
            return s.get(Concept, concept_id)

    def update_concept(self, concept_id: str, **fields: Any) -> Concept | None:
        with new_session(self._engine) as s:
            concept = s.get(Concept, concept_id)
            if concept is None:
                return None
            for key, value in fields.items():
                setattr(concept, key, value)
            concept.updated_at = _now()
            s.add(concept)
            s.commit()
            s.refresh(concept)
            return concept

    def set_verification(self, concept_id: str, status: str) -> Concept | None:
        return self.update_concept(concept_id, verification_status=status)

    def concepts_for_paper(self, paper_id: str, state: str | None = None) -> list[Concept]:
        with new_session(self._engine) as s:
            stmt = select(Concept).where(Concept.paper_id == paper_id)
            if state is not None:
                stmt = stmt.where(Concept.state == state)
            return list(s.exec(stmt))

    def list_concepts(self, *, state: str | None = None) -> list[Concept]:
        with new_session(self._engine) as s:
            stmt = select(Concept)
            if state is not None:
                stmt = stmt.where(Concept.state == state)
            return list(s.exec(stmt.order_by(Concept.title)))

    def second_brain_index(self) -> list[Concept]:
        """Promoted concepts + hubs — the retrieval/candidate corpus."""
        with new_session(self._engine) as s:
            stmt = select(Concept).where(
                Concept.state.in_([ConceptState.PROMOTED.value, ConceptState.HUB.value])
            )
            return list(s.exec(stmt))

    def find_promoted_by_title(self, title: str) -> Concept | None:
        target = normalize_title(title)
        if not target:
            return None
        for c in self.second_brain_index():
            if normalize_title(c.effective_title) == target:
                return c
        return None

    def build_title_index(self) -> dict[str, str]:
        """normalized effective-title → concept id, over promoted concepts + hubs."""
        return {
            normalize_title(c.effective_title): c.id
            for c in self.second_brain_index()
            if c.effective_title
        }

    def promote_concept(self, concept_id: str) -> Concept | None:
        """Flip a verified inbox concept into the Second Brain.

        If a promoted concept with the same effective title already exists, return
        that one instead (cross-paper de-dup, mirroring the old two-pass cache).
        """
        concept = self.get_concept(concept_id)
        if concept is None:
            return None
        existing = self.find_promoted_by_title(concept.effective_title)
        if existing and existing.id != concept_id:
            return existing
        return self.update_concept(
            concept_id, state=ConceptState.PROMOTED.value, promoted_at=_now()
        )

    def hubs(self) -> dict[str, str]:
        """Return {hub title: concept id} for all hubs (the ALLOWED_HUBS source)."""
        return {
            c.effective_title: c.id
            for c in self.list_concepts(state=ConceptState.HUB.value)
            if c.effective_title
        }

    def upsert_hub(self, name: str) -> Concept:
        existing = self.find_promoted_by_title(name)
        if existing:
            return existing
        return self.create_concept(
            title=name, state=ConceptState.HUB.value, type="Hub",
            verification_status=VerificationStatus.VERIFIED.value,
        )

    def search_concepts(self, query: str, limit: int = 50) -> list[Concept]:
        like = f"%{query.strip()}%"
        with new_session(self._engine) as s:
            stmt = (
                select(Concept)
                .where(Concept.title.ilike(like) | Concept.statement_latex.ilike(like))
                .limit(limit)
            )
            return list(s.exec(stmt))

    # ── Edges ──────────────────────────────────────────────────────────────────

    def create_edge(self, **fields: Any) -> Edge:
        edge = Edge(**fields)
        with new_session(self._engine) as s:
            s.add(edge)
            s.commit()
            s.refresh(edge)
        return edge

    def get_edge(self, edge_id: str) -> Edge | None:
        with new_session(self._engine) as s:
            return s.get(Edge, edge_id)

    def update_edge(self, edge_id: str, **fields: Any) -> Edge | None:
        with new_session(self._engine) as s:
            edge = s.get(Edge, edge_id)
            if edge is None:
                return None
            for key, value in fields.items():
                setattr(edge, key, value)
            edge.updated_at = _now()
            s.add(edge)
            s.commit()
            s.refresh(edge)
            return edge

    def set_edge_status(self, edge_id: str, status: str) -> Edge | None:
        needs_review = status == EdgeStatus.PROPOSED.value
        return self.update_edge(edge_id, status=status, needs_review=needs_review)

    def edges_for_concept(self, concept_id: str) -> list[Edge]:
        with new_session(self._engine) as s:
            stmt = select(Edge).where(
                (Edge.source_concept_id == concept_id)
                | (Edge.target_concept_id == concept_id)
            )
            return list(s.exec(stmt))

    def proposed_edges_for_concept(self, concept_id: str) -> list[Edge]:
        with new_session(self._engine) as s:
            stmt = select(Edge).where(
                Edge.source_concept_id == concept_id,
                Edge.status == EdgeStatus.PROPOSED.value,
            )
            return list(s.exec(stmt))

    def list_edges(self, *, status: str | None = None) -> list[Edge]:
        with new_session(self._engine) as s:
            stmt = select(Edge)
            if status is not None:
                stmt = stmt.where(Edge.status == status)
            return list(s.exec(stmt))

    def verify_auto_edges_between_promoted(self) -> int:
        """Promote proposed auto-channel edges to verified once both endpoints
        are in the Second Brain (promoted or hub). Returns the count verified."""
        promoted_ids = {c.id for c in self.second_brain_index()}
        verified = 0
        with new_session(self._engine) as s:
            proposed = list(
                s.exec(
                    select(Edge).where(
                        Edge.status == EdgeStatus.PROPOSED.value,
                        Edge.channel == "auto",
                    )
                )
            )
            for edge in proposed:
                if (
                    edge.source_concept_id in promoted_ids
                    and edge.target_concept_id in promoted_ids
                ):
                    edge.status = EdgeStatus.VERIFIED.value
                    edge.needs_review = False
                    edge.updated_at = _now()
                    s.add(edge)
                    verified += 1
            if verified:
                s.commit()
        return verified

    def resolve_deferred_edges(self, title_index: dict[str, str] | None = None) -> int:
        """Link deferred edges whose raw target title now matches a promoted concept.

        Replaces the Deferred Edges DB resolution pass. Returns the count resolved.
        """
        index = title_index if title_index is not None else self.build_title_index()
        resolved = 0
        with new_session(self._engine) as s:
            deferred = list(
                s.exec(select(Edge).where(Edge.deferred == True))  # noqa: E712
            )
            for edge in deferred:
                target_id = index.get(normalize_title(edge.target_title_raw))
                if target_id:
                    edge.target_concept_id = target_id
                    edge.deferred = False
                    edge.updated_at = _now()
                    s.add(edge)
                    resolved += 1
            if resolved:
                s.commit()
        return resolved

    # ── Projects (minimal) ───────────────────────────────────────────────────────

    def list_projects(self) -> list[Project]:
        with new_session(self._engine) as s:
            return list(s.exec(select(Project)))

    # ── Graph ──────────────────────────────────────────────────────────────────

    def graph_data(self, *, verified_only: bool = True) -> dict[str, list[dict]]:
        """Nodes + edges for the front-end graph (vis-network)."""
        states: Iterable[str] = (
            (ConceptState.PROMOTED.value, ConceptState.HUB.value)
            if verified_only
            else (ConceptState.PROMOTED.value, ConceptState.HUB.value, ConceptState.INBOX.value)
        )
        edge_statuses = (
            (EdgeStatus.VERIFIED.value,)
            if verified_only
            else (EdgeStatus.VERIFIED.value, EdgeStatus.PROPOSED.value)
        )
        with new_session(self._engine) as s:
            concepts = list(s.exec(select(Concept).where(Concept.state.in_(list(states)))))
            node_ids = {c.id for c in concepts}
            edges = list(s.exec(select(Edge).where(Edge.status.in_(list(edge_statuses)))))

        nodes = [
            {
                "id": c.id,
                "label": c.effective_title,
                "type": c.type,
                "state": c.state,
                "hub": c.suggested_hub,
            }
            for c in concepts
        ]
        graph_edges = [
            {
                "id": e.id,
                "from": e.source_concept_id,
                "to": e.target_concept_id,
                "relation_type": e.relation_type,
                "status": e.status,
                "confidence": e.ai_confidence,
                "rationale": e.rationale or e.justification,
            }
            for e in edges
            if e.target_concept_id in node_ids and e.source_concept_id in node_ids
        ]
        return {"nodes": nodes, "edges": graph_edges}

    # ── Counts (dashboard) ───────────────────────────────────────────────────────

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {st.value: 0 for st in PaperStatus}
        for paper in self.list_papers():
            counts[paper.status] = counts.get(paper.status, 0) + 1
        return counts

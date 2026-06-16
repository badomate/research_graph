"""
modules/store/research_repo.py — Phase-1 data access (mixed into ``Store``).

Covers paper organization (tags / collections / projects with roles), selective
parsing (parse scopes, parse jobs, artifacts, chunks), analysis jobs, the
AI-suggestion quarantine with regeneration lineage, and math objects.

Kept in a mixin so the original ``repository.py`` stays focused; ``Store`` inherits
``ResearchStoreMixin`` and these methods read as part of the same repository.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlmodel import select

from .db import new_session
from .models import (
    AiSuggestion,
    AnalysisJob,
    Collection,
    JobStatus,
    MathObject,
    MathObjectProject,
    Paper,
    PaperArtifact,
    PaperChunk,
    PaperCollection,
    PaperProject,
    PaperTag,
    ParseJob,
    ParseScope,
    Project,
    SuggestionStatus,
    SuggestionType,
    Tag,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# Which first-class table an accepted suggestion type is promoted into.
_PROMOTABLE = {
    SuggestionType.MATH_OBJECT.value: "math_objects",
    SuggestionType.THEOREM.value: "math_objects",
    SuggestionType.ASSUMPTION.value: "math_objects",
    SuggestionType.CONCEPT.value: "concepts",
    SuggestionType.PROJECT_LINK.value: "paper_projects",
}

# Suggestion type → math_object type, for promotion.
_SUGGESTION_TO_MO_TYPE = {
    SuggestionType.THEOREM.value: "theorem",
    SuggestionType.ASSUMPTION.value: "assumption",
    SuggestionType.MATH_OBJECT.value: "definition",
}


class ResearchStoreMixin:
    """Phase-1 repository methods. Mixed into :class:`Store`."""

    # ── Tags ────────────────────────────────────────────────────────────────────

    def create_tag(self, name: str, color: str = "") -> Tag:
        with new_session(self._engine) as s:
            existing = s.exec(select(Tag).where(Tag.name == name.strip())).first()
            if existing:
                return existing
            tag = Tag(name=name.strip(), color=color)
            s.add(tag)
            s.commit()
            s.refresh(tag)
            return tag

    def list_tags(self) -> list[Tag]:
        with new_session(self._engine) as s:
            return list(s.exec(select(Tag).order_by(Tag.name)))

    def get_tag(self, tag_id: str) -> Tag | None:
        with new_session(self._engine) as s:
            return s.get(Tag, tag_id)

    def delete_tag(self, tag_id: str) -> None:
        with new_session(self._engine) as s:
            for link in s.exec(select(PaperTag).where(PaperTag.tag_id == tag_id)):
                s.delete(link)
            tag = s.get(Tag, tag_id)
            if tag is not None:
                s.delete(tag)
            s.commit()

    def add_paper_tag(self, paper_id: str, tag_id: str) -> None:
        with new_session(self._engine) as s:
            if s.get(PaperTag, (paper_id, tag_id)) is None:
                s.add(PaperTag(paper_id=paper_id, tag_id=tag_id))
                s.commit()

    def remove_paper_tag(self, paper_id: str, tag_id: str) -> None:
        with new_session(self._engine) as s:
            link = s.get(PaperTag, (paper_id, tag_id))
            if link is not None:
                s.delete(link)
                s.commit()

    def tags_for_paper(self, paper_id: str) -> list[Tag]:
        with new_session(self._engine) as s:
            ids = [pt.tag_id for pt in s.exec(select(PaperTag).where(PaperTag.paper_id == paper_id))]
            if not ids:
                return []
            return list(s.exec(select(Tag).where(Tag.id.in_(ids)).order_by(Tag.name)))

    def papers_for_tag(self, tag_id: str) -> list[str]:
        with new_session(self._engine) as s:
            return [pt.paper_id for pt in s.exec(select(PaperTag).where(PaperTag.tag_id == tag_id))]

    # ── Collections (nested) ────────────────────────────────────────────────────

    def create_collection(self, name: str, description: str = "", parent_id: str | None = None) -> Collection:
        with new_session(self._engine) as s:
            col = Collection(name=name.strip(), description=description, parent_id=parent_id or None)
            s.add(col)
            s.commit()
            s.refresh(col)
            return col

    def get_collection(self, collection_id: str) -> Collection | None:
        with new_session(self._engine) as s:
            return s.get(Collection, collection_id)

    def list_collections(self) -> list[Collection]:
        with new_session(self._engine) as s:
            return list(s.exec(select(Collection).order_by(Collection.name)))

    def update_collection(self, collection_id: str, **fields: Any) -> Collection | None:
        with new_session(self._engine) as s:
            col = s.get(Collection, collection_id)
            if col is None:
                return None
            for k, v in fields.items():
                setattr(col, k, v)
            s.add(col)
            s.commit()
            s.refresh(col)
            return col

    def collection_tree(self) -> list[dict]:
        """Nested collections as a tree of {collection, children:[...]}."""
        cols = self.list_collections()
        by_parent: dict[str | None, list[Collection]] = {}
        for c in cols:
            by_parent.setdefault(c.parent_id, []).append(c)

        def _build(parent_id: str | None) -> list[dict]:
            return [
                {"collection": c, "children": _build(c.id)}
                for c in by_parent.get(parent_id, [])
            ]

        return _build(None)

    def delete_collection(self, collection_id: str) -> None:
        """Delete a collection, reparent its children to the deleted node's parent."""
        with new_session(self._engine) as s:
            col = s.get(Collection, collection_id)
            if col is None:
                return
            for child in s.exec(select(Collection).where(Collection.parent_id == collection_id)):
                child.parent_id = col.parent_id
                s.add(child)
            for link in s.exec(
                select(PaperCollection).where(PaperCollection.collection_id == collection_id)
            ):
                s.delete(link)
            s.delete(col)
            s.commit()

    def add_paper_to_collection(self, paper_id: str, collection_id: str, role: str | None = None) -> None:
        with new_session(self._engine) as s:
            link = s.get(PaperCollection, (paper_id, collection_id))
            if link is None:
                link = PaperCollection(paper_id=paper_id, collection_id=collection_id)
                if role:
                    link.role = role
                s.add(link)
            elif role:
                link.role = role
                s.add(link)
            s.commit()

    def remove_paper_from_collection(self, paper_id: str, collection_id: str) -> None:
        with new_session(self._engine) as s:
            link = s.get(PaperCollection, (paper_id, collection_id))
            if link is not None:
                s.delete(link)
                s.commit()

    def collections_for_paper(self, paper_id: str) -> list[tuple[Collection, str]]:
        with new_session(self._engine) as s:
            links = list(s.exec(select(PaperCollection).where(PaperCollection.paper_id == paper_id)))
            out: list[tuple[Collection, str]] = []
            for link in links:
                col = s.get(Collection, link.collection_id)
                if col is not None:
                    out.append((col, link.role))
            return out

    def papers_in_collection(self, collection_id: str) -> list[str]:
        with new_session(self._engine) as s:
            return [
                link.paper_id
                for link in s.exec(
                    select(PaperCollection).where(PaperCollection.collection_id == collection_id)
                )
            ]

    # ── Projects (rich) ─────────────────────────────────────────────────────────

    def create_project(self, name: str, description: str = "", priority: int = 0) -> Project:
        with new_session(self._engine) as s:
            proj = Project(name=name.strip(), description=description, priority=priority)
            s.add(proj)
            s.commit()
            s.refresh(proj)
            return proj

    def get_project(self, project_id: str) -> Project | None:
        with new_session(self._engine) as s:
            return s.get(Project, project_id)

    def update_project(self, project_id: str, **fields: Any) -> Project | None:
        with new_session(self._engine) as s:
            proj = s.get(Project, project_id)
            if proj is None:
                return None
            for k, v in fields.items():
                setattr(proj, k, v)
            proj.updated_at = _now()
            s.add(proj)
            s.commit()
            s.refresh(proj)
            return proj

    def add_paper_to_project(self, paper_id: str, project_id: str, role: str | None = None, note: str = "") -> None:
        with new_session(self._engine) as s:
            link = s.get(PaperProject, (paper_id, project_id))
            if link is None:
                link = PaperProject(paper_id=paper_id, project_id=project_id, note=note)
                if role:
                    link.role = role
                s.add(link)
            else:
                if role:
                    link.role = role
                if note:
                    link.note = note
                s.add(link)
            s.commit()

    def remove_paper_from_project(self, paper_id: str, project_id: str) -> None:
        with new_session(self._engine) as s:
            link = s.get(PaperProject, (paper_id, project_id))
            if link is not None:
                s.delete(link)
                s.commit()

    def projects_for_paper(self, paper_id: str) -> list[tuple[Project, str]]:
        with new_session(self._engine) as s:
            links = list(s.exec(select(PaperProject).where(PaperProject.paper_id == paper_id)))
            out: list[tuple[Project, str]] = []
            for link in links:
                proj = s.get(Project, link.project_id)
                if proj is not None:
                    out.append((proj, link.role))
            return out

    def papers_in_project(self, project_id: str, role: str | None = None) -> list[tuple[str, str]]:
        """Return [(paper_id, role)] for a project, optionally filtered by role."""
        with new_session(self._engine) as s:
            links = list(s.exec(select(PaperProject).where(PaperProject.project_id == project_id)))
            return [(link.paper_id, link.role) for link in links if role is None or link.role == role]

    # ── Parse scopes ────────────────────────────────────────────────────────────

    def create_parse_scope(self, paper_id: str, name: str, purpose: str, scope_json: dict) -> ParseScope:
        with new_session(self._engine) as s:
            scope = ParseScope(paper_id=paper_id, name=name, purpose=purpose, scope_json=scope_json)
            s.add(scope)
            s.commit()
            s.refresh(scope)
            return scope

    def get_parse_scope(self, scope_id: str) -> ParseScope | None:
        with new_session(self._engine) as s:
            return s.get(ParseScope, scope_id)

    def list_parse_scopes(self, paper_id: str) -> list[ParseScope]:
        with new_session(self._engine) as s:
            return list(
                s.exec(
                    select(ParseScope)
                    .where(ParseScope.paper_id == paper_id)
                    .order_by(ParseScope.created_at.desc())
                )
            )

    def delete_parse_scope(self, scope_id: str) -> None:
        with new_session(self._engine) as s:
            scope = s.get(ParseScope, scope_id)
            if scope is not None:
                s.delete(scope)
                s.commit()

    # ── Parse jobs ──────────────────────────────────────────────────────────────

    def create_parse_job(self, **fields: Any) -> ParseJob:
        with new_session(self._engine) as s:
            job = ParseJob(**fields)
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    def get_parse_job(self, job_id: str) -> ParseJob | None:
        with new_session(self._engine) as s:
            return s.get(ParseJob, job_id)

    def update_parse_job(self, job_id: str, **fields: Any) -> ParseJob | None:
        with new_session(self._engine) as s:
            job = s.get(ParseJob, job_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = _now()
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    def list_parse_jobs(self, paper_id: str | None = None, status: str | None = None) -> list[ParseJob]:
        with new_session(self._engine) as s:
            stmt = select(ParseJob)
            if paper_id:
                stmt = stmt.where(ParseJob.paper_id == paper_id)
            if status:
                stmt = stmt.where(ParseJob.status == status)
            return list(s.exec(stmt.order_by(ParseJob.created_at.desc())))

    def find_parse_job_by_input_hash(self, input_hash: str) -> ParseJob | None:
        """Dedup: a succeeded job for the same (pdf + scope) means no re-parse."""
        if not input_hash:
            return None
        with new_session(self._engine) as s:
            return s.exec(
                select(ParseJob).where(
                    ParseJob.input_hash == input_hash,
                    ParseJob.status == JobStatus.SUCCEEDED.value,
                )
            ).first()

    def claim_next_parse_job(self) -> ParseJob | None:
        """Atomically flip the oldest pending parse job to running and return it."""
        with new_session(self._engine) as s:
            job = s.exec(
                select(ParseJob)
                .where(ParseJob.status == JobStatus.PENDING.value)
                .order_by(ParseJob.created_at)
            ).first()
            if job is None:
                return None
            job.status = JobStatus.RUNNING.value
            job.attempts += 1
            job.started_at = _now()
            job.updated_at = _now()
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    # ── Artifacts ───────────────────────────────────────────────────────────────

    def add_artifact(self, **fields: Any) -> PaperArtifact:
        with new_session(self._engine) as s:
            art = PaperArtifact(**fields)
            s.add(art)
            s.commit()
            s.refresh(art)
            return art

    def artifacts_for_paper(self, paper_id: str, kind: str | None = None) -> list[PaperArtifact]:
        with new_session(self._engine) as s:
            stmt = select(PaperArtifact).where(PaperArtifact.paper_id == paper_id)
            if kind:
                stmt = stmt.where(PaperArtifact.kind == kind)
            return list(s.exec(stmt.order_by(PaperArtifact.created_at)))

    def artifacts_for_job(self, parse_job_id: str) -> list[PaperArtifact]:
        with new_session(self._engine) as s:
            return list(
                s.exec(select(PaperArtifact).where(PaperArtifact.parse_job_id == parse_job_id))
            )

    # ── Chunks ──────────────────────────────────────────────────────────────────

    def add_chunk(self, **fields: Any) -> PaperChunk:
        with new_session(self._engine) as s:
            chunk = PaperChunk(**fields)
            s.add(chunk)
            s.commit()
            s.refresh(chunk)
            return chunk

    def add_chunks(self, chunks: list[dict]) -> list[PaperChunk]:
        out: list[PaperChunk] = []
        with new_session(self._engine) as s:
            for data in chunks:
                chunk = PaperChunk(**data)
                s.add(chunk)
                out.append(chunk)
            s.commit()
            for c in out:
                s.refresh(c)
        return out

    def chunks_for_paper(self, paper_id: str) -> list[PaperChunk]:
        with new_session(self._engine) as s:
            return list(
                s.exec(
                    select(PaperChunk)
                    .where(PaperChunk.paper_id == paper_id)
                    .order_by(PaperChunk.parse_job_id, PaperChunk.ordinal)
                )
            )

    def chunks_for_job(self, parse_job_id: str) -> list[PaperChunk]:
        with new_session(self._engine) as s:
            return list(
                s.exec(
                    select(PaperChunk)
                    .where(PaperChunk.parse_job_id == parse_job_id)
                    .order_by(PaperChunk.ordinal)
                )
            )

    def get_chunks(self, chunk_ids: list[str]) -> list[PaperChunk]:
        if not chunk_ids:
            return []
        with new_session(self._engine) as s:
            rows = {c.id: c for c in s.exec(select(PaperChunk).where(PaperChunk.id.in_(chunk_ids)))}
        return [rows[cid] for cid in chunk_ids if cid in rows]

    # ── Analysis jobs ───────────────────────────────────────────────────────────

    def create_analysis_job(self, **fields: Any) -> AnalysisJob:
        with new_session(self._engine) as s:
            job = AnalysisJob(**fields)
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    def get_analysis_job(self, job_id: str) -> AnalysisJob | None:
        with new_session(self._engine) as s:
            return s.get(AnalysisJob, job_id)

    def update_analysis_job(self, job_id: str, **fields: Any) -> AnalysisJob | None:
        with new_session(self._engine) as s:
            job = s.get(AnalysisJob, job_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = _now()
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    def list_analysis_jobs(self, paper_id: str | None = None, status: str | None = None) -> list[AnalysisJob]:
        with new_session(self._engine) as s:
            stmt = select(AnalysisJob)
            if paper_id:
                stmt = stmt.where(AnalysisJob.paper_id == paper_id)
            if status:
                stmt = stmt.where(AnalysisJob.status == status)
            return list(s.exec(stmt.order_by(AnalysisJob.created_at.desc())))

    def claim_next_analysis_job(self) -> AnalysisJob | None:
        with new_session(self._engine) as s:
            job = s.exec(
                select(AnalysisJob)
                .where(AnalysisJob.status == JobStatus.PENDING.value)
                .order_by(AnalysisJob.created_at)
            ).first()
            if job is None:
                return None
            job.status = JobStatus.RUNNING.value
            job.attempts += 1
            job.started_at = _now()
            job.updated_at = _now()
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    # ── AI suggestions (quarantine + regeneration lineage) ──────────────────────

    def create_suggestion(self, **fields: Any) -> AiSuggestion:
        with new_session(self._engine) as s:
            sug = AiSuggestion(**fields)
            s.add(sug)
            s.commit()
            s.refresh(sug)
            return sug

    def get_suggestion(self, suggestion_id: str) -> AiSuggestion | None:
        with new_session(self._engine) as s:
            return s.get(AiSuggestion, suggestion_id)

    def update_suggestion(self, suggestion_id: str, **fields: Any) -> AiSuggestion | None:
        with new_session(self._engine) as s:
            sug = s.get(AiSuggestion, suggestion_id)
            if sug is None:
                return None
            for k, v in fields.items():
                setattr(sug, k, v)
            sug.updated_at = _now()
            s.add(sug)
            s.commit()
            s.refresh(sug)
            return sug

    def list_suggestions(
        self,
        *,
        paper_id: str | None = None,
        project_id: str | None = None,
        suggestion_type: str | None = None,
        status: str | None = None,
        analysis_job_id: str | None = None,
    ) -> list[AiSuggestion]:
        with new_session(self._engine) as s:
            stmt = select(AiSuggestion)
            if paper_id:
                stmt = stmt.where(AiSuggestion.paper_id == paper_id)
            if project_id:
                stmt = stmt.where(AiSuggestion.project_id == project_id)
            if suggestion_type:
                stmt = stmt.where(AiSuggestion.suggestion_type == suggestion_type)
            if status:
                stmt = stmt.where(AiSuggestion.status == status)
            if analysis_job_id:
                stmt = stmt.where(AiSuggestion.analysis_job_id == analysis_job_id)
            return list(s.exec(stmt.order_by(AiSuggestion.created_at.desc())))

    def suggestion_versions(self, suggestion_id: str) -> list[AiSuggestion]:
        """Full regeneration lineage (oldest → newest) for a suggestion's chain."""
        sug = self.get_suggestion(suggestion_id)
        if sug is None:
            return []
        # Walk up to the root.
        root = sug
        seen = {root.id}
        while root.parent_generation_id and root.parent_generation_id not in seen:
            parent = self.get_suggestion(root.parent_generation_id)
            if parent is None:
                break
            seen.add(parent.id)
            root = parent
        # Walk down collecting all descendants by BFS.
        with new_session(self._engine) as s:
            all_sugs = list(s.exec(select(AiSuggestion)))
        children: dict[str, list[AiSuggestion]] = {}
        for cand in all_sugs:
            if cand.parent_generation_id:
                children.setdefault(cand.parent_generation_id, []).append(cand)
        chain: list[AiSuggestion] = []
        stack = [root]
        while stack:
            node = stack.pop(0)
            chain.append(node)
            stack.extend(sorted(children.get(node.id, []), key=lambda x: x.created_at))
        return chain

    def regenerate_suggestion(self, suggestion_id: str, instruction: str = "") -> AnalysisJob | None:
        """Queue a fresh analysis that re-derives one suggestion.

        Creates a child :class:`AnalysisJob` (status pending) seeded from the
        suggestion's originating job — same chunks/type/scope — carrying the
        optional ``instruction`` and lineage pointers. The old suggestion is NOT
        touched; the worker produces a new version linked back to it, and the old
        one is only superseded when the user accepts the new one.
        """
        sug = self.get_suggestion(suggestion_id)
        if sug is None:
            return None
        parent_job = self.get_analysis_job(sug.analysis_job_id) if sug.analysis_job_id else None
        return self.create_analysis_job(
            paper_id=sug.paper_id or (parent_job.paper_id if parent_job else None),
            project_id=sug.project_id or (parent_job.project_id if parent_job else None),
            analysis_type=(parent_job.analysis_type if parent_job else sug.suggestion_type),
            model=(parent_job.model if parent_job else sug.model),
            prompt_version=(parent_job.prompt_version if parent_job else sug.prompt_version),
            chunk_ids=(list(parent_job.chunk_ids) if parent_job else []),
            instruction=instruction,
            parent_generation_id=(parent_job.id if parent_job else None),
            target_suggestion_id=suggestion_id,
        )

    def accept_suggestion(self, suggestion_id: str, edited_payload: dict | None = None) -> AiSuggestion | None:
        """Accept (optionally after edit), promote to a first-class table, and
        supersede any older versions in the same lineage chain."""
        sug = self.get_suggestion(suggestion_id)
        if sug is None:
            return None
        if edited_payload is not None:
            sug = self.update_suggestion(suggestion_id, payload_json=edited_payload, status=SuggestionStatus.EDITED.value)
        ref_table, ref_id = self._promote_suggestion(sug)
        sug = self.update_suggestion(
            suggestion_id,
            status=SuggestionStatus.ACCEPTED.value,
            promoted_ref_table=ref_table,
            promoted_ref_id=ref_id,
        )
        self._supersede_ancestors(suggestion_id)
        return sug

    def reject_suggestion(self, suggestion_id: str) -> AiSuggestion | None:
        return self.update_suggestion(suggestion_id, status=SuggestionStatus.REJECTED.value)

    def _supersede_ancestors(self, suggestion_id: str) -> None:
        """Mark every older version in this lineage as superseded."""
        sug = self.get_suggestion(suggestion_id)
        if sug is None:
            return
        cur = sug.parent_generation_id
        seen: set[str] = {suggestion_id}
        while cur and cur not in seen:
            parent = self.get_suggestion(cur)
            if parent is None:
                break
            seen.add(parent.id)
            if parent.status != SuggestionStatus.ACCEPTED.value:
                self.update_suggestion(parent.id, status=SuggestionStatus.SUPERSEDED.value)
            cur = parent.parent_generation_id

    def _promote_suggestion(self, sug: AiSuggestion) -> tuple[str, str]:
        """Materialize an accepted suggestion into a first-class table.

        Returns (table, row_id). Types without a dedicated Phase-1 table are kept
        as accepted suggestions (provenance retained) and return ("", "").
        """
        payload = sug.payload_json or {}
        stype = sug.suggestion_type
        if stype in _SUGGESTION_TO_MO_TYPE:
            mo = self.create_math_object(
                paper_id=sug.paper_id,
                type=payload.get("type", _SUGGESTION_TO_MO_TYPE[stype]),
                title=payload.get("title", ""),
                statement_latex=payload.get("statement_latex", ""),
                assumptions=payload.get("assumptions", ""),
                variables=payload.get("variables", ""),
                conclusion=payload.get("conclusion", ""),
                source_pages=payload.get("source_pages", []) or [],
                source_quotes=payload.get("source_quotes", []) or [],
                confidence=payload.get("confidence", 1.0),
                source_suggestion_id=sug.id,
            )
            return "math_objects", mo.id
        if stype == SuggestionType.CONCEPT.value:
            # Promote into the existing concepts inbox (verified by the human accept).
            c = self.create_concept(  # type: ignore[attr-defined]
                paper_id=sug.paper_id,
                title=payload.get("title", ""),
                type=payload.get("type", "Definition"),
                statement_latex=payload.get("statement_latex", ""),
                assumptions=payload.get("assumptions", ""),
                conclusion=payload.get("conclusion", ""),
                suggested_hub=payload.get("suggested_hub", ""),
                canonical_keywords=payload.get("canonical_keywords", []) or [],
                verification_status="verified",
            )
            return "concepts", c.id
        if stype == SuggestionType.PROJECT_LINK.value and sug.project_id and sug.paper_id:
            self.add_paper_to_project(
                sug.paper_id, sug.project_id, role=payload.get("role"), note=payload.get("note", "")
            )
            return "paper_projects", f"{sug.paper_id}:{sug.project_id}"
        if stype == SuggestionType.SUMMARY.value and sug.paper_id:
            # A human-accepted summary becomes the paper's one-liner if empty.
            paper = self.get_paper(sug.paper_id)  # type: ignore[attr-defined]
            text = payload.get("text") or payload.get("summary") or ""
            if paper is not None and text and not (paper.one_liner or "").strip():
                self.update_paper(sug.paper_id, one_liner=text[:500])  # type: ignore[attr-defined]
            return "papers", sug.paper_id
        return "", ""

    # ── Math objects ────────────────────────────────────────────────────────────

    def create_math_object(self, **fields: Any) -> MathObject:
        with new_session(self._engine) as s:
            mo = MathObject(**fields)
            s.add(mo)
            s.commit()
            s.refresh(mo)
            return mo

    def get_math_object(self, math_object_id: str) -> MathObject | None:
        with new_session(self._engine) as s:
            return s.get(MathObject, math_object_id)

    def update_math_object(self, math_object_id: str, **fields: Any) -> MathObject | None:
        with new_session(self._engine) as s:
            mo = s.get(MathObject, math_object_id)
            if mo is None:
                return None
            for k, v in fields.items():
                setattr(mo, k, v)
            mo.updated_at = _now()
            s.add(mo)
            s.commit()
            s.refresh(mo)
            return mo

    def delete_math_object(self, math_object_id: str) -> None:
        with new_session(self._engine) as s:
            for link in s.exec(
                select(MathObjectProject).where(MathObjectProject.math_object_id == math_object_id)
            ):
                s.delete(link)
            mo = s.get(MathObject, math_object_id)
            if mo is not None:
                s.delete(mo)
            s.commit()

    def list_math_objects(
        self, *, paper_id: str | None = None, type: str | None = None, q: str | None = None
    ) -> list[MathObject]:
        with new_session(self._engine) as s:
            stmt = select(MathObject)
            if paper_id:
                stmt = stmt.where(MathObject.paper_id == paper_id)
            if type:
                stmt = stmt.where(MathObject.type == type)
            rows = list(s.exec(stmt.order_by(MathObject.title)))
        if q:
            ql = q.lower().strip()
            rows = [
                m for m in rows
                if ql in (m.title or "").lower()
                or ql in (m.statement_latex or "").lower()
                or ql in (m.conclusion or "").lower()
            ]
        return rows

    def link_math_object_project(self, math_object_id: str, project_id: str, relevance: str) -> None:
        with new_session(self._engine) as s:
            link = s.get(MathObjectProject, (math_object_id, project_id))
            if link is None:
                link = MathObjectProject(
                    math_object_id=math_object_id, project_id=project_id, relevance=relevance
                )
            else:
                link.relevance = relevance
            s.add(link)
            s.commit()

    def math_objects_for_project(self, project_id: str) -> list[tuple[MathObject, str]]:
        with new_session(self._engine) as s:
            links = list(
                s.exec(select(MathObjectProject).where(MathObjectProject.project_id == project_id))
            )
            out: list[tuple[MathObject, str]] = []
            for link in links:
                mo = s.get(MathObject, link.math_object_id)
                if mo is not None:
                    out.append((mo, link.relevance))
            return out

    # ── Unified local search ────────────────────────────────────────────────────

    def search_papers(self, query: str, limit: int = 50) -> list[dict]:
        """Local search across paper metadata + parsed chunks (single-user scale).

        Returns ranked dicts with the matched field and a short snippet so the UI
        can show *why* it matched. SQLite has no shared FULLTEXT index, so this
        ANDs lowercased terms over an in-memory haystack — fine at this scale and
        a clean seam to later swap for FTS5.
        """
        terms = [t for t in (query or "").lower().split() if t]
        if not terms:
            return []
        with new_session(self._engine) as s:
            papers = list(s.exec(select(Paper)))
            chunks_by_paper: dict[str, list[PaperChunk]] = {}
            for ch in s.exec(select(PaperChunk)):
                chunks_by_paper.setdefault(ch.paper_id, []).append(ch)

        results: list[dict] = []
        for p in papers:
            fields = {
                "title": p.title or "",
                "authors": p.authors or "",
                "abstract": p.one_liner or "",
                "notes": p.ai_notes or "",
                "themes": " ".join(p.active_themes or []),
            }
            chunk_text = " ".join(c.text or "" for c in chunks_by_paper.get(p.id, []))
            haystack = " ".join(fields.values()) + " " + chunk_text
            hay_l = haystack.lower()
            if not all(t in hay_l for t in terms):
                continue
            matched_field, snippet = self._best_match_field(fields, chunk_text, terms)
            score = sum(hay_l.count(t) for t in terms)
            results.append({
                "paper": p,
                "matched_field": matched_field,
                "snippet": snippet,
                "score": score,
                "parsed": bool(chunks_by_paper.get(p.id)),
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def _best_match_field(fields: dict[str, str], chunk_text: str, terms: list[str]) -> tuple[str, str]:
        candidates = list(fields.items()) + [("parsed text", chunk_text)]
        for name, value in candidates:
            vl = value.lower()
            for t in terms:
                idx = vl.find(t)
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(value), idx + 80)
                    snippet = ("…" if start > 0 else "") + value[start:end].strip() + ("…" if end < len(value) else "")
                    return name, snippet
        return "title", (fields.get("title", "")[:140])

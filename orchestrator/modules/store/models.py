"""
modules/store/models.py — SQLModel tables (the local source of truth).

Replaces the six Notion databases:
  Paper Tracker          → ``papers``
  Knowledge Inbox + Second Brain → ``concepts`` (unified, distinguished by ``state``)
  Edges DB + Deferred Edges + per-page "Edge Suggestions" → ``edges``
  Projects               → ``projects`` / ``project_concepts``

Status / type strings are kept identical to the old Notion values so the rest of
the pipeline (and the documented state machine) needs no re-mapping.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── String enums (subclass str so they compare/serialize as the wire value) ────


class PaperStatus(str, Enum):
    S0_INBOX = "s0-inbox"
    S1_SKIM = "s1-skim"
    S1_PROCESSING = "s1-processing"
    S1B_WAITING_ATTACHMENT = "s1b-waiting-attachment"
    BLOCKED_EXTRACTION = "blocked-extraction"
    S2_EXTRACTED = "s2-extracted"
    S2_REEXTRACT = "s2-reextract"
    S2_READ = "s2-read"
    S3_DISTILLED = "s3-distilled"


class PaperSource(str, Enum):
    ZOTERO = "zotero"
    ARXIV = "arxiv"
    MANUAL = "manual"


class ConceptState(str, Enum):
    INBOX = "inbox"
    PROMOTED = "promoted"   # Second Brain
    HUB = "hub"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"


class RelationType(str, Enum):
    DEPENDS_ON = "depends_on"
    ENABLES = "enables"
    GENERALIZES = "generalizes"
    SPECIAL_CASE_OF = "special_case_of"
    RELATED = "related"


class EdgeChannel(str, Enum):
    AUTO = "auto"
    SUGGEST = "suggest"


class EdgeStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"


# ── Tables ─────────────────────────────────────────────────────────────────────


class Paper(SQLModel, table=True):
    __tablename__ = "papers"

    id: str = Field(default_factory=_uid, primary_key=True)
    title: str = ""
    authors: str = ""
    status: str = Field(default=PaperStatus.S0_INBOX.value, index=True)
    source: str = Field(default=PaperSource.MANUAL.value)

    # Intake / attachment resolution
    zotero_key: str = ""
    attachment_key: str = ""
    zotero_uri: str = ""
    arxiv_id: str = ""
    arxiv_url: str = ""
    doi: str = ""
    primary_pdf_filename: str = ""
    pdf_path: str = ""          # set for in-app uploads (bypasses Koofr/Zotero)
    pdf_sha256: str = ""

    # Extraction output / bookkeeping
    one_liner: str = ""
    active_themes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    rejected_concepts: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    extraction_version: str = ""
    extraction_count: int = 0
    extraction_tokens: int = 0
    reextract_hints: str = ""
    extraction_error: str = ""
    ai_status: str = ""
    ai_notes: str = ""
    last_run_id: str = ""
    processed_at: datetime | None = None

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Concept(SQLModel, table=True):
    __tablename__ = "concepts"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str | None = Field(default=None, foreign_key="papers.id", index=True)
    state: str = Field(default=ConceptState.INBOX.value, index=True)

    type: str = ""
    title: str = Field(default="", index=True)
    corrected_title: str = ""

    statement_latex: str = ""
    assumptions: str = ""
    variables: str = ""
    conclusion: str = ""
    interpretation: str = ""
    proof_idea: str = ""
    source_quote: str = ""
    source_anchors: str = ""
    aliases: str = ""
    suggested_hub: str = ""
    result_category: str = ""

    source_pages: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    named_tools: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    setting: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    canonical_keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    prereq_keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    downstream_keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    ai_confidence: float = 1.0
    verification_status: str = Field(default=VerificationStatus.UNVERIFIED.value, index=True)
    graph_link_status: str = "unlinked"
    reviewer_notes: str = ""
    flag_reasons: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    promoted_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def effective_title(self) -> str:
        return (self.corrected_title or self.title).strip()


class Edge(SQLModel, table=True):
    __tablename__ = "edges"

    id: str = Field(default_factory=_uid, primary_key=True)
    source_concept_id: str = Field(foreign_key="concepts.id", index=True)
    target_concept_id: str | None = Field(default=None, foreign_key="concepts.id", index=True)
    # When an edge is proposed before its target is promoted, we keep the raw
    # target title and resolve it later (replaces the Deferred Edges DB).
    target_title_raw: str = ""

    relation_type: str = Field(default=RelationType.RELATED.value)
    direction: str = "A_to_B"
    rationale: str = ""
    justification: str = ""
    ai_confidence: float = 0.0
    channel: str = Field(default=EdgeChannel.SUGGEST.value)
    status: str = Field(default=EdgeStatus.PROPOSED.value, index=True)
    needs_review: bool = True
    deferred: bool = Field(default=False, index=True)
    demoted_from_auto: bool = False
    falsifiability: str = ""
    driving_fields: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Project(SQLModel, table=True):
    __tablename__ = "projects"

    id: str = Field(default_factory=_uid, primary_key=True)
    name: str = ""
    description: str = ""
    status: str = Field(default="active", index=True)   # active | archived
    priority: int = 0                                   # higher = more important
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ProjectConcept(SQLModel, table=True):
    __tablename__ = "project_concepts"

    project_id: str = Field(foreign_key="projects.id", primary_key=True)
    concept_id: str = Field(foreign_key="concepts.id", primary_key=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — paper organization, selective parsing/analysis, AI-suggestion
# quarantine + regeneration. New tables are additive; the Alembic migration in
# ``orchestrator/alembic/`` adds them (and the new ``projects`` columns) to
# existing databases, while ``create_all`` builds them on fresh ones.
# ═══════════════════════════════════════════════════════════════════════════════


# ── String enums ─────────────────────────────────────────────────────────────────


class PaperRole(str, Enum):
    """Role a paper plays inside a project/collection."""
    CORE = "core"
    DIRECT_COMPETITOR = "direct_competitor"
    BASELINE = "baseline"
    THEORY_TOOL = "theory_tool"
    BACKGROUND = "background"
    CITATION_ONLY = "citation_only"
    MAYBE_RELEVANT = "maybe_relevant"
    IRRELEVANT = "irrelevant"


class ScopeKind(str, Enum):
    FULL = "full"
    PAGE_RANGE = "page_range"
    REGIONS = "regions"
    MIXED = "mixed"


class ScopePurpose(str, Enum):
    TRIAGE = "triage"
    DEEP_ANALYSIS = "deep_analysis"
    THEOREM_EXTRACTION = "theorem_extraction"
    RELATED_WORK = "related_work"
    CITATION_CONTEXT = "citation_context"
    NOVELTY_RISK = "novelty_risk"
    MANUAL = "manual"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ParseBackend(str, Enum):
    MARKER_API = "marker_api"
    MARKER_LOCAL = "marker_local"
    OCR = "ocr"
    MANUAL = "manual"


class AnalysisType(str, Enum):
    TRIAGE_SUMMARY = "triage_summary"
    CLAIM_EXTRACTION = "claim_extraction"
    MATH_OBJECT_EXTRACTION = "math_object_extraction"
    THEOREM_ASSUMPTION_EXTRACTION = "theorem_assumption_extraction"
    NOVELTY_RISK = "novelty_risk"
    PROJECT_RELEVANCE = "project_relevance"
    CITATION_SUGGESTIONS = "citation_suggestions"
    BASELINE_DETECTION = "baseline_detection"
    LIMITATION_EXTRACTION = "limitation_extraction"


class SuggestionType(str, Enum):
    SUMMARY = "summary"
    CLAIM = "claim"
    MATH_OBJECT = "math_object"
    CONCEPT = "concept"
    EDGE = "edge"
    PROJECT_LINK = "project_link"
    CITATION_USE = "citation_use"
    NOVELTY_RISK = "novelty_risk"
    LIMITATION = "limitation"
    BASELINE_CANDIDATE = "baseline_candidate"
    THEOREM = "theorem"
    ASSUMPTION = "assumption"


class SuggestionStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    SUPERSEDED = "superseded"


class RegionLabel(str, Enum):
    THEOREM = "theorem"
    DEFINITION = "definition"
    ASSUMPTION = "assumption"
    EQUATION = "equation"
    ALGORITHM = "algorithm"
    PROOF_SKETCH = "proof_sketch"
    RELATED_WORK = "related_work"
    LIMITATION = "limitation"
    EXPERIMENT_RESULT = "experiment_result"
    CITATION_CONTEXT = "citation_context"
    OTHER = "other"


class ArtifactKind(str, Enum):
    ORIGINAL_PDF = "original_pdf"
    SUBSET_PDF = "subset_pdf"
    REGION_CROP = "region_crop"
    MARKER_MARKDOWN = "marker_markdown"
    MARKER_JSON = "marker_json"
    EXTRACTED_FIGURES = "extracted_figures"
    EXTRACTED_TABLES = "extracted_tables"


class MathObjectType(str, Enum):
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    ASSUMPTION = "assumption"
    ALGORITHM = "algorithm"
    EQUATION = "equation"
    PROOF_SKETCH = "proof_sketch"


class MathObjectRelevance(str, Enum):
    DIRECT_TOOL = "direct_tool"
    BACKGROUND = "background"
    COMPETING_RESULT = "competing_result"
    MISSING_ASSUMPTION = "missing_assumption"
    POSSIBLE_EXTENSION = "possible_extension"
    CITATION_ONLY = "citation_only"


# ── Organization: tags, collections (nested), project/collection membership ──────


class Tag(SQLModel, table=True):
    __tablename__ = "tags"

    id: str = Field(default_factory=_uid, primary_key=True)
    name: str = Field(default="", index=True)
    color: str = ""
    created_at: datetime = Field(default_factory=_now)


class Collection(SQLModel, table=True):
    __tablename__ = "collections"

    id: str = Field(default_factory=_uid, primary_key=True)
    name: str = ""
    description: str = ""
    parent_id: str | None = Field(default=None, foreign_key="collections.id", index=True)
    created_at: datetime = Field(default_factory=_now)


class PaperTag(SQLModel, table=True):
    __tablename__ = "paper_tags"

    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    tag_id: str = Field(foreign_key="tags.id", primary_key=True)
    created_at: datetime = Field(default_factory=_now)


class PaperCollection(SQLModel, table=True):
    __tablename__ = "paper_collections"

    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    collection_id: str = Field(foreign_key="collections.id", primary_key=True)
    role: str = Field(default=PaperRole.MAYBE_RELEVANT.value)
    created_at: datetime = Field(default_factory=_now)


class PaperProject(SQLModel, table=True):
    __tablename__ = "paper_projects"

    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    project_id: str = Field(foreign_key="projects.id", primary_key=True)
    role: str = Field(default=PaperRole.MAYBE_RELEVANT.value)
    note: str = ""
    created_at: datetime = Field(default_factory=_now)


# ── Selective parsing / analysis ─────────────────────────────────────────────────


class ParseScope(SQLModel, table=True):
    """What to extract from a PDF (full / page_range / regions / mixed).

    ``scope_json`` shape:
        {"kind": "mixed",
         "page_ranges": [[1, 8], [35, 40]],   # 1-indexed, inclusive
         "regions": [{"page": 3, "bbox": [x0, y0, x1, y1], "label": "theorem"}]}
    """
    __tablename__ = "parse_scopes"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    name: str = ""
    purpose: str = Field(default=ScopePurpose.MANUAL.value)
    scope_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)


class ParseJob(SQLModel, table=True):
    __tablename__ = "parse_jobs"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    parse_scope_id: str | None = Field(default=None, foreign_key="parse_scopes.id", index=True)

    status: str = Field(default=JobStatus.PENDING.value, index=True)
    backend: str = Field(default=ParseBackend.MARKER_API.value)
    selected_pages: int = 0
    input_hash: str = Field(default="", index=True)   # pdf_sha256 + scope → dedup key
    marker_version: str = ""
    marker_config: dict = Field(default_factory=dict, sa_column=Column(JSON))

    cost_estimate: float = 0.0
    cost_actual: float | None = None
    error: str = ""
    attempts: int = 0

    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class PaperArtifact(SQLModel, table=True):
    __tablename__ = "paper_artifacts"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    parse_job_id: str | None = Field(default=None, foreign_key="parse_jobs.id", index=True)
    kind: str = Field(default=ArtifactKind.MARKER_MARKDOWN.value, index=True)
    path: str = ""                 # filesystem path (subset_pdf / region_crop / figures)
    text: str = ""                 # inline text (marker_markdown / marker_json)
    content_hash: str = ""
    page_from: int | None = None
    page_to: int | None = None
    bbox: list | None = Field(default=None, sa_column=Column(JSON))
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)


class PaperChunk(SQLModel, table=True):
    """A unit of parsed content selectable for analysis (the AnalysisScope inputs)."""
    __tablename__ = "paper_chunks"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    parse_job_id: str | None = Field(default=None, foreign_key="parse_jobs.id", index=True)
    artifact_id: str | None = Field(default=None, foreign_key="paper_artifacts.id", index=True)

    ordinal: int = 0
    kind: str = "text"             # text | heading | theorem | ...
    heading: str = ""
    text: str = ""
    page_from: int | None = None
    page_to: int | None = None
    bbox: list | None = Field(default=None, sa_column=Column(JSON))
    token_estimate: int = 0
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_now)


class AnalysisJob(SQLModel, table=True):
    __tablename__ = "analysis_jobs"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    project_id: str | None = Field(default=None, foreign_key="projects.id", index=True)

    analysis_type: str = Field(default=AnalysisType.TRIAGE_SUMMARY.value, index=True)
    model: str = ""
    prompt_version: str = ""
    chunk_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    instruction: str = ""          # optional regeneration instruction
    # Regeneration lineage: the analysis job this one re-ran.
    parent_generation_id: str | None = Field(
        default=None, foreign_key="analysis_jobs.id", index=True
    )
    # When a regen targets a single suggestion (the per-item Regenerate button),
    # this points at it so the worker can link the new suggestion 1:1.
    target_suggestion_id: str | None = Field(default=None, foreign_key="ai_suggestions.id")

    status: str = Field(default=JobStatus.PENDING.value, index=True)
    input_token_estimate: int = 0
    output_token_estimate: int = 0
    cost_estimate: float = 0.0
    input_tokens_actual: int | None = None
    output_tokens_actual: int | None = None
    cost_actual: float | None = None
    error: str = ""
    attempts: int = 0

    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ── AI-suggestion quarantine + regeneration lineage ─────────────────────────────


class AiSuggestion(SQLModel, table=True):
    """Every Claude-generated item lands here first — never directly in the
    accepted research tables. Accept/reject/edit/regenerate all live here."""
    __tablename__ = "ai_suggestions"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str | None = Field(default=None, foreign_key="papers.id", index=True)
    project_id: str | None = Field(default=None, foreign_key="projects.id", index=True)
    parse_job_id: str | None = Field(default=None, foreign_key="parse_jobs.id")
    analysis_job_id: str | None = Field(default=None, foreign_key="analysis_jobs.id", index=True)

    suggestion_type: str = Field(default=SuggestionType.SUMMARY.value, index=True)
    payload_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    model: str = ""
    prompt_version: str = ""
    input_hash: str = ""
    output_hash: str = ""
    status: str = Field(default=SuggestionStatus.PENDING.value, index=True)

    # Regeneration lineage.
    parent_generation_id: str | None = Field(
        default=None, foreign_key="ai_suggestions.id", index=True
    )
    regeneration_reason: str = ""

    # Set once accepted + promoted to a first-class table (provenance pointer).
    promoted_ref_table: str = ""
    promoted_ref_id: str = ""

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ── Math objects (first-class, promotable from suggestions) ──────────────────────


class MathObject(SQLModel, table=True):
    __tablename__ = "math_objects"

    id: str = Field(default_factory=_uid, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    type: str = Field(default=MathObjectType.THEOREM.value, index=True)
    title: str = Field(default="", index=True)
    statement_latex: str = ""
    assumptions: str = ""
    variables: str = ""
    conclusion: str = ""
    source_pages: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    source_quotes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    confidence: float = 1.0
    source_suggestion_id: str | None = Field(
        default=None, foreign_key="ai_suggestions.id"
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class MathObjectProject(SQLModel, table=True):
    __tablename__ = "math_object_projects"

    math_object_id: str = Field(foreign_key="math_objects.id", primary_key=True)
    project_id: str = Field(foreign_key="projects.id", primary_key=True)
    relevance: str = Field(default=MathObjectRelevance.BACKGROUND.value)
    created_at: datetime = Field(default_factory=_now)

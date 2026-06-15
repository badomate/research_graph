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
    created_at: datetime = Field(default_factory=_now)


class ProjectConcept(SQLModel, table=True):
    __tablename__ = "project_concepts"

    project_id: str = Field(foreign_key="projects.id", primary_key=True)
    concept_id: str = Field(foreign_key="concepts.id", primary_key=True)

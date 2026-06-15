"""
modules/store — local SQLite source of truth (replaces Notion).

Public surface:
    from modules.store import Store
    store = Store()
    store.create_all()        # idempotent schema creation
"""
from __future__ import annotations

from .db import get_engine, init_db, make_engine, new_session
from .models import (
    Concept,
    ConceptState,
    Edge,
    EdgeChannel,
    EdgeStatus,
    Paper,
    PaperSource,
    PaperStatus,
    Project,
    ProjectConcept,
    RelationType,
    VerificationStatus,
)
from .repository import Store, normalize_title

__all__ = [
    "Store",
    "normalize_title",
    "get_engine",
    "make_engine",
    "init_db",
    "new_session",
    "Paper",
    "Concept",
    "Edge",
    "Project",
    "ProjectConcept",
    "PaperStatus",
    "PaperSource",
    "ConceptState",
    "VerificationStatus",
    "RelationType",
    "EdgeChannel",
    "EdgeStatus",
]

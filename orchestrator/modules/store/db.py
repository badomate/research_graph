"""
modules/store/db.py — SQLite engine / session factory.

A single SQLite file is shared by two processes (the web app and the
orchestrator scheduler), so WAL journaling + a generous busy-timeout are enabled
on every connection to allow concurrent reads/writes for this single-user,
low-write workload.
"""
from __future__ import annotations

import os

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# Default points at the shared Docker volume; override with DATABASE_URL.
DEFAULT_DATABASE_URL = "sqlite:////data/app.db"

_engine: Engine | None = None


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def _apply_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def make_engine(url: str | None = None) -> Engine:
    """Create a configured engine (used by tests to point at a temp DB)."""
    url = url or _database_url()
    connect_args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    if url.startswith("sqlite"):
        _apply_sqlite_pragmas(engine)
    return engine


def get_engine() -> Engine:
    """Return the process-wide singleton engine."""
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Create all tables if they do not exist (safe to call on every startup)."""
    # Import models so they register on SQLModel.metadata before create_all.
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(engine or get_engine())


def new_session(engine: Engine | None = None) -> Session:
    # expire_on_commit=False keeps returned rows usable after the session closes.
    return Session(engine or get_engine(), expire_on_commit=False)

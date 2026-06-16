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


# Additive columns that ``create_all`` cannot add to a pre-existing table.
# (New *tables* are handled by create_all; only altered tables need this.)
# Each entry: table → [(column, sqlite_type, default_sql_or_None)].
_ADDITIVE_COLUMNS: dict[str, list[tuple[str, str, str | None]]] = {
    "projects": [
        ("status", "VARCHAR", "'active'"),
        ("priority", "INTEGER", "0"),
        ("updated_at", "DATETIME", None),
    ],
}


def _ensure_additive_columns(engine: Engine) -> None:
    """Idempotently add new columns to pre-existing tables (lightweight migration).

    ``SQLModel.metadata.create_all`` is additive only at table granularity, so an
    older database that already has e.g. ``projects`` won't get columns added in a
    later release. This adds them via ``ALTER TABLE … ADD COLUMN`` if missing —
    safe to run on every startup, and tolerant of the two-process startup race
    (duplicate-column errors are ignored).
    """
    from sqlalchemy import inspect, text
    from sqlalchemy.exc import OperationalError

    insp = inspect(engine)
    tables = set(insp.get_table_names())
    for table, cols in _ADDITIVE_COLUMNS.items():
        if table not in tables:
            continue
        present = {c["name"] for c in insp.get_columns(table)}
        for name, sqltype, default in cols:
            if name in present:
                continue
            ddl = f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}"
            if default is not None:
                ddl += f" DEFAULT {default}"
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
            except OperationalError:
                # Another process added it first (race), or it now exists — fine.
                pass


def init_db(engine: Engine | None = None) -> None:
    """Create all tables if they do not exist (safe to call on every startup)."""
    # Import models so they register on SQLModel.metadata before create_all.
    from . import models  # noqa: F401

    engine = engine or get_engine()
    SQLModel.metadata.create_all(engine)
    _ensure_additive_columns(engine)


def new_session(engine: Engine | None = None) -> Session:
    # expire_on_commit=False keeps returned rows usable after the session closes.
    return Session(engine or get_engine(), expire_on_commit=False)

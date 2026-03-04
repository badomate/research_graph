"""
modules/job_ledger.py — SQLite-based idempotency / restart-safety ledger
─────────────────────────────────────────────────────────────────────────
Tracks every ingestion job so that:
  - A re-run of the scheduler never reprocesses a successfully completed job.
  - Partial failures can be resumed from the last successful milestone.
  - Operators can inspect job history directly via SQLite (e.g. DB Browser).

Status lifecycle:
  started → marker_done → openai_done → notion_done (success)
                                       └→ failed     (error path)

All SQL uses parameterized queries — never f-string or %-format SQL.
All public methods acquire a module-level threading.Lock for thread safety.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = "/tmp/pipeline/ingestion_jobs.db"

# Valid status values — enforce at the Python layer; SQLite has no enum.
# Pipeline flow: started → marker_done → extract_done → retrieve_done
#                        → link_done → notion_done
# openai_done retained for backward compatibility with v2 ledger rows.
VALID_STATUSES: frozenset[str] = frozenset(
    {
        "started",
        "marker_done",
        "openai_done",      # v2 legacy; kept for backward compat
        "extract_done",     # v3: Stage 1 LLM extraction complete
        "retrieve_done",    # v3: Stage 2 candidate retrieval complete
        "link_done",        # v3: Stage 3 LLM linking complete
        "notion_done",
        "failed",
    }
)

_lock = threading.Lock()


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ── Ledger class ──────────────────────────────────────────────────────────────


class JobLedger:
    """
    Thin SQLite wrapper for the ``ingestion_jobs`` table.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Resolved in this order:

        1. Explicit *db_path* constructor argument.
        2. ``PIPELINE_DB_PATH`` environment variable.
        3. Default: ``/tmp/pipeline/ingestion_jobs.db``.

    The parent directory is created automatically if it does not exist.
    :py:meth:`create_tables` is called in ``__init__``.
    """

    def __init__(self, db_path: str | None = None) -> None:
        resolved = (
            db_path
            or os.environ.get("PIPELINE_DB_PATH")
            or _DEFAULT_DB_PATH
        )
        self._db_path = Path(resolved)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("JobLedger: using database at %s", self._db_path)
        self.create_tables()

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Create the ``ingestion_jobs`` table if it does not already exist."""
        with _lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    zotero_key          TEXT NOT NULL,
                    pdf_sha256          TEXT NOT NULL,
                    extraction_version  TEXT NOT NULL,
                    status              TEXT NOT NULL DEFAULT 'started',
                    started_at          TEXT,
                    finished_at         TEXT,
                    error               TEXT,
                    retry_count         INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_lookup
                ON ingestion_jobs (zotero_key, pdf_sha256, extraction_version)
                """
            )
            conn.commit()

    # ── Write operations ──────────────────────────────────────────────────────

    def start_job(
        self,
        zotero_key: str,
        pdf_sha256: str,
        extraction_version: str,
    ) -> int:
        """
        Insert a new job row and return its auto-generated ``id``.

        If a previous job exists for the same (zotero_key, pdf_sha256,
        extraction_version) triple with status ``failed``, its ``retry_count``
        is incremented and the old row is reused (returned id is the existing
        row id).  This avoids unbounded table growth on repeated retries.
        """
        with _lock, sqlite3.connect(str(self._db_path)) as conn:
            # Check for an existing failed job to reuse
            row = conn.execute(
                """
                SELECT id, retry_count FROM ingestion_jobs
                WHERE zotero_key = ?
                  AND pdf_sha256 = ?
                  AND extraction_version = ?
                  AND status = 'failed'
                ORDER BY id DESC
                LIMIT 1
                """,
                (zotero_key, pdf_sha256, extraction_version),
            ).fetchone()

            if row is not None:
                job_id: int = row[0]
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'started',
                        started_at = ?,
                        finished_at = NULL,
                        error = NULL,
                        retry_count = retry_count + 1
                    WHERE id = ?
                    """,
                    (_utcnow(), job_id),
                )
                conn.commit()
                logger.debug(
                    "JobLedger: reusing job_id=%d (retry #%d) for zotero_key=%s",
                    job_id,
                    row[1] + 1,
                    zotero_key,
                )
                return job_id

            cursor = conn.execute(
                """
                INSERT INTO ingestion_jobs
                    (zotero_key, pdf_sha256, extraction_version, status, started_at)
                VALUES (?, ?, ?, 'started', ?)
                """,
                (zotero_key, pdf_sha256, extraction_version, _utcnow()),
            )
            conn.commit()
            job_id = cursor.lastrowid  # type: ignore[assignment]
            logger.debug(
                "JobLedger: started job_id=%d for zotero_key=%s", job_id, zotero_key
            )
            return job_id

    def update_status(
        self,
        job_id: int,
        status: str,
        error: str | None = None,
    ) -> None:
        """
        Update the status of *job_id*.

        Parameters
        ----------
        job_id: int
            Row id returned by :py:meth:`start_job`.
        status: str
            One of the VALID_STATUSES values.
        error: str | None
            Optional error message, stored only when ``status == 'failed'``.

        Raises
        ------
        ValueError
            If *status* is not in :data:`VALID_STATUSES`.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}."
            )
        with _lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error = ?
                WHERE id = ?
                """,
                (status, error, job_id),
            )
            conn.commit()
        logger.debug("JobLedger: job_id=%d → status=%s", job_id, status)

    def finish_job(self, job_id: int) -> None:
        """
        Mark *job_id* as successfully completed (``notion_done``) and
        record the ``finished_at`` timestamp.
        """
        with _lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'notion_done', finished_at = ?, error = NULL
                WHERE id = ?
                """,
                (_utcnow(), job_id),
            )
            conn.commit()
        logger.debug("JobLedger: job_id=%d finished successfully.", job_id)

    # ── Read operations ───────────────────────────────────────────────────────

    def is_already_done(
        self,
        zotero_key: str,
        pdf_sha256: str,
        extraction_version: str,
    ) -> bool:
        """
        Return True if there is already a ``notion_done`` job for the given
        (zotero_key, pdf_sha256, extraction_version) triple.

        This is the primary idempotency guard: if the pipeline crashed and
        was restarted, a previously completed job will not be re-run.
        """
        with _lock, sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                """
                SELECT 1 FROM ingestion_jobs
                WHERE zotero_key = ?
                  AND pdf_sha256 = ?
                  AND extraction_version = ?
                  AND status = 'notion_done'
                LIMIT 1
                """,
                (zotero_key, pdf_sha256, extraction_version),
            ).fetchone()
        return row is not None

    def get_latest_job(self, zotero_key: str) -> dict[str, Any] | None:
        """
        Return the most recent job row for *zotero_key* as a dict, or
        ``None`` if no jobs exist for that key.
        """
        with _lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE zotero_key = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (zotero_key,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

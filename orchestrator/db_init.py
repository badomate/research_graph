"""
orchestrator/db_init.py — Standalone SQLite database initialiser
────────────────────────────────────────────────────────────────
Creates the ``ingestion_jobs`` table (and any supporting indexes) in the
pipeline SQLite database.

Usage
-----
From the orchestrator directory::

    python db_init.py
    python db_init.py --db-path /custom/path/ingestion_jobs.db

Environment variables
---------------------
PIPELINE_DB_PATH
    Overrides the default database path when ``--db-path`` is not supplied.
    Default: ``/tmp/pipeline/ingestion_jobs.db``
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from the orchestrator directory without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from modules.job_ledger import JobLedger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("db_init")


def main() -> None:
    """Parse arguments and initialise the database."""
    parser = argparse.ArgumentParser(
        description="Initialise the pipeline SQLite database (creates tables if absent).",
    )
    parser.add_argument(
        "--db-path",
        metavar="PATH",
        default=None,
        help=(
            "Path to the SQLite database file. "
            "Defaults to PIPELINE_DB_PATH env var or /tmp/pipeline/ingestion_jobs.db."
        ),
    )
    args = parser.parse_args()

    db_path: str | None = args.db_path or os.environ.get("PIPELINE_DB_PATH")

    ledger = JobLedger(db_path=db_path)
    logger.info("Database initialised at: %s", ledger._db_path)
    logger.info("Tables created (or already exist): ingestion_jobs")


if __name__ == "__main__":
    main()

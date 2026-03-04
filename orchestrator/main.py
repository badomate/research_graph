"""
orchestrator/main.py — Central Scheduler
─────────────────────────────────────────
Starts APScheduler with the following jobs:

  ┌─────────────────────────────┬─────────────────────────────────────────────┐
  │ Job                         │ Schedule                                    │
  ├─────────────────────────────┼─────────────────────────────────────────────┤
  │ Ingestion Engine            │ Every 10 minutes (interval)                 │
  │ ArXiv Sniper                │ Daily at 06:00 (cron)                       │
  │ Conflict Detector           │ Every 30 minutes (interval)                 │
  │ LaTeX Skeleton Compiler     │ Every 15 minutes (interval)                 │
  │ Dependency Grapher          │ Every 12 hours (interval)                   │
  └─────────────────────────────┴─────────────────────────────────────────────┘

All credentials are sourced from environment variables (see .env.example).
"""

from __future__ import annotations

import logging
import os
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

# Load .env file if present (useful for local development outside Docker)
load_dotenv()

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("orchestrator")

# ── Deferred imports so that missing env vars give a clear error at startup ───
def _import_modules():
    from modules.ingestion import IngestionEngine
    from modules.arxiv_sniper import ArXivSniper
    from modules.conflict_detector import ConflictDetector
    from modules.latex_compiler import LaTeXCompiler
    from modules.dependency_grapher import DependencyGrapher

    return (
        IngestionEngine,
        ArXivSniper,
        ConflictDetector,
        LaTeXCompiler,
        DependencyGrapher,
    )


# ── Job wrappers (each constructs fresh instance to avoid state leakage) ──────

def run_ingestion() -> None:
    from modules.ingestion import IngestionEngine
    IngestionEngine().run()


def run_arxiv_sniper() -> None:
    from modules.arxiv_sniper import ArXivSniper
    ArXivSniper().run()


def run_conflict_detector() -> None:
    from modules.conflict_detector import ConflictDetector
    ConflictDetector().run()


def run_latex_compiler() -> None:
    from modules.latex_compiler import LaTeXCompiler
    LaTeXCompiler().run()


def run_dependency_grapher() -> None:
    from modules.dependency_grapher import DependencyGrapher
    DependencyGrapher().run()


def run_promotion() -> None:
    from modules.promotion import PromotionEngine
    PromotionEngine().run()


# ── Startup checks ────────────────────────────────────────────────────────────

REQUIRED_ENV_VARS = [
    "NOTION_TOKEN",
    "NOTION_PAPER_TRACKER_DB_ID",
    "NOTION_KNOWLEDGE_INBOX_DB_ID",
    "NOTION_SECOND_BRAIN_DB_ID",
    "NOTION_PROJECTS_DB_ID",
    "OPENAI_API_KEY",
    "KOOFR_USER",
    "KOOFR_APP_PASSWORD",
]


def _check_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        logger.error(
            "Missing required environment variables: %s — check your .env file.",
            ", ".join(missing),
        )
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_env()

    scheduler = BlockingScheduler(timezone="UTC")

    # ── Module 1: Ingestion Engine — every 10 minutes ─────────────────────────
    scheduler.add_job(
        run_ingestion,
        trigger=IntervalTrigger(minutes=10),
        id="ingestion",
        name="Core Ingestion Engine",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    # ── Module 2: ArXiv Sniper — daily at 06:00 UTC ───────────────────────────
    scheduler.add_job(
        run_arxiv_sniper,
        trigger=CronTrigger(hour=6, minute=0, timezone="UTC"),
        id="arxiv_sniper",
        name="ArXiv Sniper",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    # ── Module 3: Conflict Detector — every 30 minutes ────────────────────────
    scheduler.add_job(
        run_conflict_detector,
        trigger=IntervalTrigger(minutes=30),
        id="conflict_detector",
        name="Assumption Conflict Detector",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    # ── Module 4: LaTeX Compiler — every 15 minutes ───────────────────────────
    scheduler.add_job(
        run_latex_compiler,
        trigger=IntervalTrigger(minutes=15),
        id="latex_compiler",
        name="LaTeX Skeleton Compiler",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    # ── Module 5: Dependency Grapher — every 12 hours ─────────────────────────
    scheduler.add_job(
        run_dependency_grapher,
        trigger=IntervalTrigger(hours=12),
        id="dependency_grapher",
        name="Dependency Grapher",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # ── Module 6: Promotion Engine — every 30 minutes ─────────────────────────
    scheduler.add_job(
        run_promotion,
        trigger=IntervalTrigger(minutes=30),
        id="promotion",
        name="Promotion Engine",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    logger.info("Orchestrator starting — %d job(s) scheduled.", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  • [%s] %s", job.id, job.name)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Orchestrator shutting down.")


if __name__ == "__main__":
    main()

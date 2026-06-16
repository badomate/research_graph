"""Central APScheduler entry point for the paper pipeline (SQLite Store backend)."""

from __future__ import annotations

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from modules.arxiv_sniper import ArXivSniper
from modules.config import Config, get_config
from modules.ingestion import IngestionEngine
from modules.promotion import PromotionEngine
from modules.store import Store, make_engine
from modules.vector_index import VectorIndexEngine
from modules.zotero_intake import ZoteroIntake


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("orchestrator")


def _load_config() -> Config:
    try:
        return get_config()
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        sys.exit(1)


def run_arxiv_sniper(config: Config) -> None:
    ArXivSniper(config=config).run()


def run_zotero_intake(config: Config) -> None:
    ZoteroIntake(config=config).run()


def _run_startup_once(config: Config, vector_index: VectorIndexEngine | None) -> None:
    startup_jobs = [
        ("Core Ingestion Engine",
         lambda: IngestionEngine(vector_index=vector_index, config=config).run()),
        ("Promotion Engine",
         lambda: PromotionEngine(vector_index=vector_index, config=config).run()),
    ]
    logger.info("Startup pass: running %d job(s) once now.", len(startup_jobs))
    for name, fn in startup_jobs:
        try:
            logger.info("Startup pass: running %s ...", name)
            fn()
            logger.info("Startup pass: %s complete.", name)
        except Exception as exc:
            logger.exception("Startup pass: %s failed; error_type=%s", name, type(exc).__name__)


def main() -> None:
    config = _load_config()

    # Ensure the shared database exists before any worker touches it.
    Store(make_engine(config.database_url)).create_all()

    vector_index = VectorIndexEngine(config) if config.vector_index_enabled else None

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        lambda: IngestionEngine(vector_index=vector_index, config=config).run(),
        trigger=IntervalTrigger(minutes=10),
        id="ingestion", name="Core Ingestion Engine",
        max_instances=1, coalesce=True, misfire_grace_time=60,
    )
    scheduler.add_job(
        lambda: PromotionEngine(vector_index=vector_index, config=config).run(),
        trigger=IntervalTrigger(minutes=30),
        id="promotion", name="Promotion Engine",
        max_instances=1, coalesce=True, misfire_grace_time=120,
    )
    scheduler.add_job(
        lambda: run_arxiv_sniper(config),
        trigger=CronTrigger(hour=6, minute=0),
        id="arxiv_sniper", name="ArXiv Sniper",
        max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    if config.zotero_poll_enabled:
        scheduler.add_job(
            lambda: run_zotero_intake(config),
            trigger=IntervalTrigger(minutes=config.zotero_poll_minutes),
            id="zotero_intake", name="Zotero Intake Poller",
            max_instances=1, coalesce=True, misfire_grace_time=300,
        )

    logger.info("Orchestrator starting - %d job(s) scheduled.", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  - [%s] %s", job.id, job.name)

    try:
        _run_startup_once(config, vector_index)
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Orchestrator shutting down.")


if __name__ == "__main__":
    main()

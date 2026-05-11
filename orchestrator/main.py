"""Central APScheduler entry point for the paper pipeline."""

from __future__ import annotations

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from modules.config import Config, get_config
from modules.dependency_grapher import DependencyGrapher
from modules.ingestion import IngestionEngine
from modules.promotion import PromotionEngine
from modules.vector_index import VectorIndexEngine


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
    from modules.arxiv_sniper import ArXivSniper

    ArXivSniper(config).run()


def run_dependency_grapher() -> None:
    DependencyGrapher().run()


def main() -> None:
    config = _load_config()
    vector_index = VectorIndexEngine(config) if config.vector_index_enabled else None

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        lambda: IngestionEngine(vector_index=vector_index, config=config).run(),
        trigger=IntervalTrigger(minutes=10),
        id="ingestion",
        name="Core Ingestion Engine",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    scheduler.add_job(
        lambda: run_arxiv_sniper(config),
        trigger=CronTrigger(hour=6, minute=0, timezone="UTC"),
        id="arxiv_sniper",
        name="ArXiv Sniper",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        run_dependency_grapher,
        trigger=IntervalTrigger(hours=12),
        id="dependency_grapher",
        name="Dependency Grapher",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        lambda: PromotionEngine(vector_index=vector_index, config=config).run(),
        trigger=IntervalTrigger(minutes=30),
        id="promotion",
        name="Promotion Engine",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    logger.info("Orchestrator starting - %d job(s) scheduled.", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  - [%s] %s", job.id, job.name)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Orchestrator shutting down.")


if __name__ == "__main__":
    main()

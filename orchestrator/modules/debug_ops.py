"""
modules/debug_ops.py — read-only status + maintenance ops for the debug console.

Used by the web UI's ``/debug`` page. Imports of the heavy pipeline (vector index,
ingestion engine, job workers) are done lazily inside each function so importing
this module stays cheap.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


# ── Status ───────────────────────────────────────────────────────────────────────


def system_stats(store) -> dict:
    """Counts across the SQLite source of truth (papers / concepts / edges / jobs)."""
    papers = store.list_papers()
    concepts = store.list_concepts()
    edges = store.list_edges()

    def _by(items, attr):
        out: dict[str, int] = {}
        for it in items:
            out[getattr(it, attr)] = out.get(getattr(it, attr), 0) + 1
        return out

    return {
        "papers": len(papers),
        "papers_by_status": _by(papers, "status"),
        "concepts": len(concepts),
        "concepts_by_state": _by(concepts, "state"),
        "concepts_by_verification": _by(concepts, "verification_status"),
        "edges": len(edges),
        "edges_by_status": _by(edges, "status"),
        "parse_jobs": _by(store.list_parse_jobs(), "status"),
        "analysis_jobs": _by(store.list_analysis_jobs(), "status"),
        "suggestions": _by(store.list_suggestions(), "status"),
        "math_objects": len(store.list_math_objects()),
    }


def qdrant_status(config) -> dict:
    """Qdrant reachability + per-collection point counts (never raises)."""
    status: dict = {
        "url": config.qdrant_url,
        "enabled": config.vector_index_enabled,
        "backend": config.vector_embedding_backend,
        "available": False,
        "collections": [],
        "error": "",
    }
    try:
        from modules.vector_index import VectorIndexEngine

        vi = VectorIndexEngine(config)
        status["available"] = bool(vi.available)
        if vi.available:
            for name in vi.COLLECTIONS:
                try:
                    cnt = vi._client.count(collection_name=name, exact=True).count
                except Exception:
                    cnt = None
                status["collections"].append({"name": name, "points": cnt})
        else:
            status["error"] = "Qdrant unreachable or collections missing"
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def marker_status(config) -> dict:
    """Ping the Marker proxy's health endpoint."""
    out = {"url": config.marker_api_url, "ok": False, "detail": ""}
    try:
        import requests

        r = requests.get(f"{config.marker_api_url}/", timeout=5)
        out["ok"] = r.status_code == 200
        out["detail"] = r.text[:200]
    except Exception as exc:  # noqa: BLE001
        out["detail"] = f"{type(exc).__name__}: {exc}"
    return out


# ── Maintenance actions ──────────────────────────────────────────────────────────


def rebuild_qdrant_from_store(config, store) -> dict:
    """Drop + recreate the three Qdrant collections and re-index every Second-Brain
    concept from SQLite (the source of truth). Returns a result summary.

    Replaces the stale ``rebuild_index_from_second_brain`` (which still expected a
    Notion client). Use after the index is wiped, the embedding model changes, or
    edges start looking wrong.
    """
    t0 = time.time()
    try:
        from modules.ingestion.engine import IngestionEngine
        from modules.vector_index import VectorIndexEngine
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "indexed": 0, "error": f"import failed: {exc}"}

    vi = VectorIndexEngine(config)
    if not vi.available:
        return {"ok": False, "indexed": 0, "error": "Qdrant not available — check QDRANT_URL / the qdrant service"}

    vi._ensure_collections(force_recreate=True)   # drop + recreate all three
    concepts = store.second_brain_index()
    indexed = 0
    failed = 0
    for c in concepts:
        try:
            mo = IngestionEngine._math_object_from_concept(c)
            vi.index_concept(mo, c.id, verified=True)
            indexed += 1
        except Exception:
            failed += 1
            logger.warning("rebuild: failed to index concept %s", c.id, exc_info=True)
    elapsed = round(time.time() - t0, 1)
    logger.info("Qdrant rebuild complete: %d indexed, %d failed, %.1fs", indexed, failed, elapsed)
    return {"ok": True, "indexed": indexed, "failed": failed,
            "total": len(concepts), "elapsed_s": elapsed, "error": ""}


def drain_job_workers(config, store) -> dict:
    """Run the selective-parse + analysis workers once (drain pending jobs)."""
    try:
        from modules.analysis.analysis_worker import AnalysisWorker
        from modules.parsing.parse_worker import ParseWorker

        n_parse = ParseWorker(store, config=config).run_pending(limit=10)
        n_analysis = AnalysisWorker(store, config=config).run_pending(limit=10)
        return {"ok": True, "parse": n_parse, "analysis": n_analysis, "error": ""}
    except Exception as exc:  # noqa: BLE001
        logger.warning("drain_job_workers failed", exc_info=True)
        return {"ok": False, "parse": 0, "analysis": 0, "error": str(exc)}

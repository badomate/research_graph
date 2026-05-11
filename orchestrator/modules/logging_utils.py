"""
modules/logging_utils.py — Structured logging helper
──────────────────────────────────────────────────────
Provides a single `structured_log()` function that emits consistently
formatted log lines with named key=value fields.

Usage:
    from modules.logging_utils import structured_log
    structured_log(logger, "info", "Stage 1 complete", run_id=run_id, concepts=4)
    # → "[run_id=abc123] [concepts=4] Stage 1 complete"
"""

from __future__ import annotations

import logging
from typing import Any


def structured_log(
    logger: logging.Logger,
    level: str,
    event: str,
    **fields: Any,
) -> None:
    """
    Emit a log line with consistent key=value context fields.

    Fields with None or empty-string values are omitted.
    `run_id` is placed first if present.

    Example:
        structured_log(logger, "info", "Stage 1 complete",
                       run_id="abc123", paper_id="xyz", concepts=4)
        → "[run_id=abc123] [paper_id=xyz] [concepts=4] Stage 1 complete"
    """
    parts: list[str] = []

    run_id = fields.pop("run_id", None)
    if run_id:
        parts.append(f"[run_id={run_id}]")

    for key, value in fields.items():
        if value is not None and value != "":
            parts.append(f"[{key}={value}]")

    prefix = " ".join(parts)
    message = f"{prefix} {event}" if prefix else event

    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message)

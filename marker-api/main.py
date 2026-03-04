"""
marker-api/main.py
------------------
Lightweight FastAPI proxy that accepts the same interface as the local
marker_server (POST /marker  { "filepath": "<path>" }) but forwards the
PDF to the Datalab Marker API instead of processing it locally.

Environment variables:
    DATALAB_API_KEY   Required.  Your https://www.datalab.to API key.
    DATALAB_API_URL   Optional.  Base URL (default: https://www.datalab.to).
    MARKER_MODE       Optional.  Conversion mode: fast | balanced | accurate
                                 (default: balanced).
    MARKER_POLL_INTERVAL  Optional.  Seconds between poll attempts (default: 4).
    MARKER_POLL_TIMEOUT   Optional.  Max seconds to wait for a result (default: 600).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("marker-proxy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATALAB_API_KEY: str = os.environ.get("DATALAB_API_KEY", "")
DATALAB_BASE_URL: str = os.environ.get("DATALAB_API_URL", "https://www.datalab.to").rstrip("/")
MARKER_MODE: str = os.environ.get("MARKER_MODE", "balanced")
POLL_INTERVAL: float = float(os.environ.get("MARKER_POLL_INTERVAL", "4"))
POLL_TIMEOUT: float = float(os.environ.get("MARKER_POLL_TIMEOUT", "600"))

SUBMIT_URL = f"{DATALAB_BASE_URL}/api/v1/marker"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="marker-proxy")


class MarkerRequest(BaseModel):
    filepath: str


# ---------------------------------------------------------------------------
# Healthcheck — matches the probe used by docker-compose
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    if not DATALAB_API_KEY:
        return {"status": "degraded", "reason": "DATALAB_API_KEY not set"}
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.post("/marker")
def convert(req: MarkerRequest):
    pdf_path = Path(req.filepath)
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {pdf_path}")
    if not DATALAB_API_KEY:
        raise HTTPException(status_code=500, detail="DATALAB_API_KEY is not configured")

    headers = {"X-API-Key": DATALAB_API_KEY}

    # -- 1. Submit the PDF to Datalab -----------------------------------------
    logger.info("Submitting %s to Datalab Marker API (mode=%s)", pdf_path.name, MARKER_MODE)
    try:
        with pdf_path.open("rb") as fh:
            submit_resp = requests.post(
                SUBMIT_URL,
                headers=headers,
                files={"file": (pdf_path.name, fh, "application/pdf")},
                data={"output_format": "markdown", "mode": MARKER_MODE},
                timeout=120,
            )
        submit_resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Submission failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Datalab submission error: {exc}") from exc

    submit_data = submit_resp.json()
    request_check_url: str | None = submit_data.get("request_check_url")
    if not request_check_url:
        logger.error("No request_check_url in response: %s", submit_data)
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Datalab response (no request_check_url): {submit_data}",
        )

    logger.info("Submitted OK — polling %s", request_check_url)

    # -- 2. Poll until complete -----------------------------------------------
    deadline = time.monotonic() + POLL_TIMEOUT
    while True:
        time.sleep(POLL_INTERVAL)
        if time.monotonic() > deadline:
            raise HTTPException(
                status_code=504,
                detail=f"Timed out waiting for Datalab result after {POLL_TIMEOUT}s",
            )

        try:
            poll_resp = requests.get(request_check_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Poll request failed (will retry): %s", exc)
            continue

        result = poll_resp.json()
        status: str = result.get("status", "")
        logger.debug("Poll status: %s", status)

        if status == "complete":
            markdown: str = result.get("markdown", "")
            if not markdown:
                raise HTTPException(
                    status_code=502,
                    detail="Datalab returned status=complete but no markdown content",
                )
            logger.info(
                "Conversion complete — %d chars, quality=%.2f",
                len(markdown),
                result.get("parse_quality_score", 0.0),
            )
            return {"markdown": markdown}

        if status == "failed":
            error_msg = result.get("error", "unknown error")
            logger.error("Datalab reported failure: %s", error_msg)
            raise HTTPException(status_code=502, detail=f"Datalab conversion failed: {error_msg}")

        # still processing — loop

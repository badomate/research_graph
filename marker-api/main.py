"""
marker-api/main.py
------------------
FastAPI proxy that preserves the local marker_server interface:

    POST /marker
    { "filepath": "<path-visible-inside-container>" }

…but uses the Datalab Python SDK (datalab-python-sdk) to run conversion in the
hosted Datalab API instead of running Marker locally.

Why SDK:
- Handles the correct upload field naming, polling, retries, and typed responses.

Required env vars:
    DATALAB_API_KEY          Your Datalab API key.

Optional env vars:
    DATALAB_API_URL          Base URL (default: https://www.datalab.to)
    MARKER_MODE              fast | balanced | accurate (default: balanced)
    MARKER_OUTPUT_FORMAT     markdown | html | json | chunks (default: markdown)
    MARKER_PAGINATE          true|false (default: false)
    MARKER_MAX_PAGES         int (optional)
    MARKER_PAGE_RANGE        string like "0-5,10" (optional, 0-indexed)
    MARKER_SKIP_CACHE        true|false (default: false)
    MARKER_DISABLE_IMAGES    true|false (default: true)
    MARKER_DISABLE_CAPTIONS  true|false (default: true)
    MARKER_TOKEN_EFFICIENT_MD true|false (default: true)

Polling config (maps to SDK convert() args):
    MARKER_POLL_INTERVAL     seconds between polls (default: 2)
    MARKER_POLL_TIMEOUT      max seconds total (default: 600)

Test / stub mode (no API calls ever made):
    MARKER_STUB              true|false (default: false) – when true, skip all
                             Datalab calls and return the contents of
                             MARKER_STUB_MD_PATH as the markdown result.
    MARKER_STUB_MD_PATH      path to the .md file to return in stub mode
                             (default: FT2BINAR.md next to this file)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("marker-proxy")

# -----------------------------------------------------------------------------
# Env / Config
# -----------------------------------------------------------------------------
DATALAB_API_KEY: str = os.environ.get("DATALAB_API_KEY", "").strip()
DATALAB_BASE_URL: str = os.environ.get("DATALAB_API_URL", "https://www.datalab.to").strip().rstrip("/")

MARKER_MODE: str = os.environ.get("MARKER_MODE", "balanced").strip()
MARKER_OUTPUT_FORMAT: str = os.environ.get("MARKER_OUTPUT_FORMAT", "markdown").strip()

MARKER_PAGINATE: bool = os.environ.get("MARKER_PAGINATE", "false").strip().lower() in ("1", "true", "yes", "y")
MARKER_MAX_PAGES: Optional[int] = None
if os.environ.get("MARKER_MAX_PAGES", "").strip():
    try:
        MARKER_MAX_PAGES = int(os.environ["MARKER_MAX_PAGES"])
    except Exception:
        MARKER_MAX_PAGES = None

MARKER_PAGE_RANGE: Optional[str] = os.environ.get("MARKER_PAGE_RANGE", "").strip() or None

MARKER_SKIP_CACHE: bool = os.environ.get("MARKER_SKIP_CACHE", "false").strip().lower() in ("1", "true", "yes", "y")
MARKER_DISABLE_IMAGES: bool = os.environ.get("MARKER_DISABLE_IMAGES", "true").strip().lower() in ("1", "true", "yes", "y")
MARKER_DISABLE_CAPTIONS: bool = os.environ.get("MARKER_DISABLE_CAPTIONS", "true").strip().lower() in ("1", "true", "yes", "y")
MARKER_TOKEN_EFFICIENT_MD: bool = os.environ.get("MARKER_TOKEN_EFFICIENT_MD", "true").strip().lower() in ("1", "true", "yes", "y")

POLL_INTERVAL: float = float(os.environ.get("MARKER_POLL_INTERVAL", "2"))
POLL_TIMEOUT: float = float(os.environ.get("MARKER_POLL_TIMEOUT", "600"))

# Stub / test mode – return a static .md file instead of calling the API
MARKER_STUB: bool = os.environ.get("MARKER_STUB", "false").strip().lower() in ("1", "true", "yes", "y")
_DEFAULT_STUB_MD = str(Path(__file__).parent / "FT2BINAR.md")
MARKER_STUB_MD_PATH: str = os.environ.get("MARKER_STUB_MD_PATH", _DEFAULT_STUB_MD).strip()

# convert() accepts max_polls and poll_interval (seconds). We convert timeout seconds to max polls.
def _max_polls() -> int:
    if POLL_INTERVAL <= 0:
        return 300
    return max(1, int(math.ceil(POLL_TIMEOUT / POLL_INTERVAL)))

# -----------------------------------------------------------------------------
# SDK import + client init (lazy)
# -----------------------------------------------------------------------------
_sdk_import_error: Optional[str] = None
_datalab_client = None

def _get_client():
    global _datalab_client, _sdk_import_error
    if _datalab_client is not None:
        return _datalab_client
    try:
        from datalab_sdk import DatalabClient
    except Exception as exc:
        _sdk_import_error = f"{type(exc).__name__}: {exc}"
        raise

    # SDK supports base_url + timeout (seconds)
    _datalab_client = DatalabClient(
        api_key=DATALAB_API_KEY or None,
        base_url=DATALAB_BASE_URL,
        timeout=int(POLL_TIMEOUT) if POLL_TIMEOUT else 300,
    )
    return _datalab_client

def _get_options():
    from datalab_sdk import ConvertOptions

    kwargs: dict[str, Any] = {
        "output_format": MARKER_OUTPUT_FORMAT,
        "mode": MARKER_MODE,
        "paginate": MARKER_PAGINATE,
        "skip_cache": MARKER_SKIP_CACHE,
        "disable_image_extraction": MARKER_DISABLE_IMAGES,
        "disable_image_captions": MARKER_DISABLE_CAPTIONS,
        "token_efficient_markdown": MARKER_TOKEN_EFFICIENT_MD,
    }
    if MARKER_MAX_PAGES is not None:
        kwargs["max_pages"] = MARKER_MAX_PAGES
    if MARKER_PAGE_RANGE:
        kwargs["page_range"] = MARKER_PAGE_RANGE

    return ConvertOptions(**kwargs)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="marker-proxy", version="2.0.0")

class MarkerRequest(BaseModel):
    filepath: str

@app.get("/")
def health() -> dict[str, str]:
    if MARKER_STUB:
        return {"status": "ok", "mode": "stub", "stub_md": MARKER_STUB_MD_PATH}
    if not DATALAB_API_KEY:
        return {"status": "degraded", "reason": "DATALAB_API_KEY not set"}
    if _sdk_import_error:
        return {"status": "degraded", "reason": f"SDK import error: {_sdk_import_error}"}
    return {"status": "ok"}

@app.post("/marker")
def convert(req: MarkerRequest) -> dict[str, Any]:
    # ------------------------------------------------------------------ #
    # STUB MODE: return static .md file, never touch the Datalab API      #
    # ------------------------------------------------------------------ #
    # stub_path = Path(MARKER_STUB_MD_PATH)
    # if not stub_path.exists():
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"MARKER_STUB is enabled but stub file not found: {stub_path}",
    #     )
    # markdown = stub_path.read_text(encoding="utf-8")
    # logger.info("STUB MODE – returning %s (%d chars)", stub_path.name, len(markdown))
    # return {
    #     "markdown": markdown,
    #     "page_count": None,
    #     "parse_quality_score": None,
    #     "cost_breakdown": None,
    #     "metadata": {"stub": True, "stub_file": str(stub_path)},
    # }

    if not DATALAB_API_KEY:
        raise HTTPException(status_code=500, detail="DATALAB_API_KEY is not configured")

    pdf_path = Path(req.filepath)
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {pdf_path}")
    if pdf_path.is_dir():
        raise HTTPException(status_code=400, detail=f"filepath is a directory, expected a PDF file: {pdf_path}")

    try:
        client = _get_client()
        options = _get_options()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Datalab SDK not available. Install datalab-python-sdk. ({type(exc).__name__}: {exc})",
        ) from exc

    logger.info(
        "Datalab SDK convert: file=%s mode=%s format=%s paginate=%s poll_interval=%.2fs max_polls=%d",
        pdf_path.name,
        MARKER_MODE,
        MARKER_OUTPUT_FORMAT,
        MARKER_PAGINATE,
        POLL_INTERVAL,
        _max_polls(),
    )

    try:
        # SDK handles upload + polling internally.
        result = client.convert(
            str(pdf_path),
            options=options,
            poll_interval=POLL_INTERVAL,
            max_polls=_max_polls(),
        )
    except Exception as exc:
        # Surface the error as 502 to match your orchestrator’s expectation
        raise HTTPException(status_code=502, detail=f"Datalab SDK conversion error: {type(exc).__name__}: {exc}") from exc

    # The SDK returns a ConversionResult with fields:
    # success, markdown/html/json/chunks, page_count, parse_quality_score, cost_breakdown, metadata, images, error
    success = bool(getattr(result, "success", False))
    if not success:
        err = getattr(result, "error", "") or "unknown error"
        raise HTTPException(status_code=502, detail=f"Datalab conversion failed: {err}")

    markdown = getattr(result, "markdown", "") or ""
    if not markdown:
        raise HTTPException(
            status_code=502,
            detail=f"Datalab returned success=True but no markdown (output_format={MARKER_OUTPUT_FORMAT})",
        )

    # Keep response compatible with your orchestrator:
    # ingestion reads data.get("markdown") first.
    return {
        "markdown": markdown,
        "page_count": getattr(result, "page_count", None),
        "parse_quality_score": getattr(result, "parse_quality_score", None),
        "cost_breakdown": getattr(result, "cost_breakdown", None),
        "metadata": getattr(result, "metadata", None),
    }
"""
modules/ingestion/pdf_fetcher.py — PDF acquisition and markdown conversion.

Handles Koofr WebDAV, Zotero API, Marker OCR, and the markdown cache.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
import re

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tenacity import RetryError

from ..config import Config
from ..exceptions import KoofrError, MarkerError, ZoteroError
from ..job_ledger import JobLedger
from ..logging_utils import structured_log
from ..store import Store
from ..pdf_resolver import (
    ZOTERO_ATTACH_RE as _ZOTERO_ATTACH_RE,
    ZOTERO_PARENT_RE as _ZOTERO_PARENT_RE,
    build_webdav_client,
    extract_pdf_from_zip,
    sha256_file,
)

logger = logging.getLogger(__name__)

TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))
ZOTERO_API_BASE = "https://api.zotero.org"
EXTRACTION_VERSION: str = os.environ.get("EXTRACTION_VERSION", "v3")

_BOILERPLATE_RE = re.compile(
    r'\n#{1,3}\s*('
    r'References|Bibliography|Works Cited'
    r'|Acknowledgements?|Acknowledgments?'
    r'|Appendix|Appendices|Appendix\s+[A-Z0-9]|[A-Z]\.\s+(?:Proofs?|Appendix)'
    r'|Supplementary\s+Material|Supplemental\s+Material|Supplementary'
    r'|Deferred\s+Proofs?|Proofs?\s+of\s+\w|Technical\s+Lemmas?'
    r'|Funding|Declaration\s+of|Conflicts?\s+of\s+Interest|Author\s+Contributions?'
    r')[^\n]*\n[\s\S]*$',
    re.IGNORECASE,
)


class PdfFetcherService:
    """Fetches PDFs from Koofr/Zotero, converts via Marker, caches markdown."""

    def __init__(
        self,
        store: Store,
        ledger: JobLedger,
        config: Config | None = None,
    ) -> None:
        self.store = store
        self._ledger = ledger
        self._webdav = self._build_webdav_client(config)
        self.marker_url = config.marker_api_url if config is not None else os.environ.get("MARKER_API_URL", "http://marker-api:8080")
        self.koofr_base = config.koofr_pdf_path if config is not None else os.environ.get("KOOFR_PDF_PATH", "/zotero")
        self.zotero_user_id = config.zotero_user_id if config is not None else os.environ["ZOTERO_USER_ID"]
        self.zotero_api_key = config.zotero_api_key if config is not None else os.environ["ZOTERO_API_KEY"]
        self.koofr_markdown_dir = config.koofr_markdown_path if config is not None else os.environ.get("KOOFR_MARKDOWN_PATH", "/zotero_markdown")
        koofr_user = config.koofr_user if config is not None else os.environ.get("KOOFR_USER", "")
        # Only touch Koofr when it's actually configured — uploads/arXiv bypass it.
        if koofr_user:
            self._ensure_koofr_markdown_dir()

    @staticmethod
    def _build_webdav_client(config: Config | None = None):
        try:
            if config is not None:
                return build_webdav_client(
                    koofr_user=config.koofr_user,
                    koofr_app_password=config.koofr_app_password,
                )
            return build_webdav_client()
        except TypeError:
            class _NoopWebDAV:
                def check(self, *_args, **_kwargs):
                    return True

                def mkdir(self, *_args, **_kwargs):
                    return None

            return _NoopWebDAV()

    # -- Zotero ----------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _zotero_children(self, parent_key: str) -> list[dict]:
        url = (
            f"{ZOTERO_API_BASE}/users/{self.zotero_user_id}"
            f"/items/{parent_key}/children"
        )
        resp = requests.get(
            url,
            headers={"Zotero-API-Key": self.zotero_api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_attachment_key(
        self, zotero_uri: str, parent_key: str
    ) -> tuple[str, str] | None:
        attach_match = _ZOTERO_ATTACH_RE.search(zotero_uri)
        if attach_match:
            return parent_key, attach_match.group(1)
        try:
            children = self._zotero_children(parent_key)
        except Exception as exc:
            raise ZoteroError(
                f"Zotero children fetch failed for parent '{parent_key}'"
            ) from exc

        pdf_children: list[tuple[str, dict]] = []
        for child in children:
            data = child.get("data", {})
            link_mode = data.get("linkMode", "")
            content_type = data.get("contentType", "")
            if link_mode in ("imported_file", "imported_url") and "pdf" in content_type:
                attach_key = child.get("key")
                if attach_key:
                    pdf_children.append((attach_key, data))

        if not pdf_children:
            logger.warning("No PDF attachment found for Zotero parent '%s'.", parent_key)
            return None

        for attach_key, data in pdf_children:
            filename = data.get("filename", "")
            if filename.lower().startswith(parent_key.lower()):
                return parent_key, attach_key

        pdf_children.sort(key=lambda x: x[1].get("fileSize", 0), reverse=True)
        return parent_key, pdf_children[0][0]

    def resolve_keys_and_update(
        self, paper_id: str, zotero_uri: str, parent_key: str, run_id: str
    ) -> tuple[str, str] | None:
        resolved = self._resolve_attachment_key(zotero_uri, parent_key)
        if resolved is None:
            return None
        _parent_key, attachment_key = resolved
        try:
            self.store.update_paper(paper_id, attachment_key=attachment_key)
        except Exception:
            logger.warning("[%s] Could not write attachment key to store.", run_id)
        return _parent_key, attachment_key

    # -- Koofr -----------------------------------------------------------------

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def koofr_exists(self, remote_path: str) -> bool:
        try:
            return self._webdav.check(remote_path)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "404" in exc_str or "not found" in exc_str or "no such" in exc_str:
                return False
            raise

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _download_koofr(self, remote_path: str, local_path: Path) -> None:
        self._webdav.download_sync(remote_path=remote_path, local_path=str(local_path))

    def _koofr_download_bytes(self, remote_path: str) -> bytes:
        tmp = TMP_DIR / f"_download_{uuid.uuid4().hex[:8]}.tmp"
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            self._webdav.download_sync(remote_path=remote_path, local_path=str(tmp))
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    def _koofr_upload(self, remote_path: str, data: bytes) -> None:
        tmp = TMP_DIR / f"_upload_{uuid.uuid4().hex[:8]}.tmp"
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            self._webdav.upload_sync(remote_path=remote_path, local_path=str(tmp))
        finally:
            tmp.unlink(missing_ok=True)

    def _ensure_koofr_markdown_dir(self) -> None:
        try:
            if not self._webdav.check(self.koofr_markdown_dir):
                self._webdav.mkdir(self.koofr_markdown_dir)
                logger.info("PdfFetcher: created Koofr markdown dir: %s", self.koofr_markdown_dir)
            else:
                logger.debug("PdfFetcher: Koofr markdown dir exists: %s", self.koofr_markdown_dir)
        except Exception:
            logger.warning(
                "PdfFetcher: could not ensure Koofr markdown dir '%s' — "
                "markdown caching may fail.",
                self.koofr_markdown_dir,
                exc_info=True,
            )

    # -- PDF extraction --------------------------------------------------------

    @staticmethod
    def _extract_pdf_from_zip(
        zip_path: Path, output_path: Path, preferred: str | None = None
    ) -> None:
        extract_pdf_from_zip(zip_path, output_path, preferred=preferred)

    @staticmethod
    def _sha256(path: Path) -> str:
        return sha256_file(path)

    # -- Marker API ------------------------------------------------------------

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _call_marker(self, pdf_path: Path) -> str:
        response = requests.post(
            f"{self.marker_url}/marker",
            json={"filepath": str(pdf_path)},
            timeout=300,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:1000]
            raise MarkerError(
                f"Marker API HTTP {response.status_code} for {pdf_path}: {detail}"
            ) from exc
        data = response.json()
        return (
            data.get("markdown")
            or data.get("output")
            or data.get("text")
            or response.text
        )

    # -- Boilerplate stripping -------------------------------------------------

    def strip_boilerplate(self, text: str) -> str:
        stripped = _BOILERPLATE_RE.sub("", text)
        ratio = len(stripped) / max(len(text), 1)
        if ratio < 0.2:
            logger.warning(
                "Boilerplate strip removed >80%% of document (%.0f%% remaining) — "
                "regex may have matched too early. Returning original.",
                ratio * 100,
            )
            return text
        logger.debug("Boilerplate strip: %.0f%% retained.", ratio * 100)
        return stripped

    # -- Main entry point ------------------------------------------------------

    def markdown_from_local_pdf(
        self, pdf_path: str, run_id: str, paper_id: str
    ) -> tuple[str, int | None]:
        """Convert an uploaded/local PDF via Marker (bypasses Koofr/Zotero)."""
        local_pdf = Path(pdf_path)
        if not local_pdf.exists():
            raise KoofrError(f"[{run_id}] Uploaded PDF not found: {pdf_path}")
        pdf_sha256 = self._sha256(local_pdf)
        self.store.update_paper(paper_id, pdf_sha256=pdf_sha256)
        job_id = self._ledger.start_job(paper_id, pdf_sha256, EXTRACTION_VERSION)
        try:
            markdown_text = self._call_marker(local_pdf)
        except RetryError as exc:
            cause = exc.last_attempt.exception() if exc.last_attempt else exc
            raise MarkerError(f"[{run_id}] Marker OCR failed: {cause}") from exc
        except Exception as exc:
            raise MarkerError(f"[{run_id}] Marker OCR failed: {exc}") from exc
        return self.strip_boilerplate(markdown_text), job_id

    def pdf_to_markdown(
        self,
        attachment_key: str,
        run_id: str,
        zip_remote: str,
        primary_pdf_filename: str | None,
        paper_id: str,
    ) -> tuple[str, int | None] | tuple[None, None]:
        """
        Return (markdown_text, job_id) using a Koofr markdown cache.

        Cache hit: returns (markdown_text, None) — no ledger job started.
        Cache miss: downloads, OCRs, starts ledger job, returns (markdown_text, job_id).
        Returns (None, None) on fatal error.
        """
        md_remote = f"{self.koofr_markdown_dir}/{attachment_key}.md"

        if self.koofr_exists(md_remote):
            logger.info("[%s] Markdown cache hit: %s", run_id, md_remote)
            try:
                raw = self._koofr_download_bytes(md_remote)
                return raw.decode("utf-8"), None
            except Exception:
                logger.warning(
                    "[%s] Markdown cache read failed — falling through to re-conversion.",
                    run_id,
                )

        logger.info("[%s] Markdown cache miss — converting PDF via marker.", run_id)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        local_zip = TMP_DIR / f"{run_id}.zip"
        local_pdf = TMP_DIR / f"{run_id}.pdf"

        try:
            self._download_koofr(zip_remote, local_zip)
            self._extract_pdf_from_zip(
                local_zip, local_pdf, preferred=primary_pdf_filename
            )
        except Exception as exc:
            raise KoofrError(f"[{run_id}] PDF download/extraction failed") from exc
        finally:
            local_zip.unlink(missing_ok=True)

        pdf_sha256 = self._sha256(local_pdf)
        structured_log(logger, "info", "PDF SHA256 computed", run_id=run_id, sha256=pdf_sha256[:16])
        self.store.update_paper(paper_id, pdf_sha256=pdf_sha256)
        job_id = self._ledger.start_job(attachment_key, pdf_sha256, EXTRACTION_VERSION)
        structured_log(logger, "info", "JobLedger job started", run_id=run_id, job_id=job_id)

        try:
            markdown_text = self._call_marker(local_pdf)
        except RetryError as exc:
            cause = exc.last_attempt.exception() if exc.last_attempt else exc
            raise MarkerError(f"[{run_id}] Marker OCR failed: {cause}") from exc
        except Exception as exc:
            raise MarkerError(f"[{run_id}] Marker OCR failed: {exc}") from exc
        finally:
            local_pdf.unlink(missing_ok=True)

        markdown_text = self.strip_boilerplate(markdown_text)
        logger.info(
            "[%s] Markdown ready after boilerplate strip.",
            run_id,
        )

        try:
            self._koofr_upload(md_remote, markdown_text.encode("utf-8"))
            logger.info("[%s] Markdown cached → %s", run_id, md_remote)
        except Exception:
            logger.warning("[%s] Markdown cache upload failed — continuing without cache.", run_id)

        return markdown_text, job_id

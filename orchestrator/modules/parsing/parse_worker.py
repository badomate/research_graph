"""
modules/parsing/parse_worker.py — runs ParseJobs (the Stage-1 selective parser).

Flow for one job:
  1. Resolve the original PDF (uploaded path, else download arXiv).
  2. Subset it to the scope's selected pages with pypdf → ``subset_pdf`` artifact.
  3. Send the subset to the Marker proxy → markdown → ``marker_markdown`` artifact.
  4. Split the markdown into ``paper_chunks`` (the AnalysisScope inputs), keeping
     page/heading provenance.
  5. Record artifacts, chunks, actual page count, and cost on the job.

De-dup: a job's ``input_hash`` = sha256(original PDF) + scope hash. If a SUCCEEDED
job already exists for that hash, we reuse its artifacts/chunks instead of calling
Marker again (requirement 5: never re-Marker an unchanged PDF/scope).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import requests

from ..config import Config, get_config
from ..cost import estimate_marker_cost, estimate_tokens
from ..store import ArtifactKind, JobStatus, ParseBackend, Store
from . import scope_utils

logger = logging.getLogger(__name__)

MARKER_VERSION = "datalab-v2"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scope_hash(scope_json: dict) -> str:
    return hashlib.sha256(json.dumps(scope_json, sort_keys=True).encode()).hexdigest()[:16]


class ParseWorker:
    def __init__(self, store: Store, config: Config | None = None) -> None:
        self.store = store
        self.config = config or get_config()
        self.marker_url = self.config.marker_api_url
        self.uploads_dir = Path(self.config.uploads_dir)

    # -- PDF acquisition -------------------------------------------------------

    def _resolve_original_pdf(self, paper) -> Path:
        if paper.pdf_path and Path(paper.pdf_path).exists():
            return Path(paper.pdf_path)
        if paper.arxiv_id:
            return self._download_arxiv(paper.arxiv_id)
        raise FileNotFoundError(
            f"No local PDF for paper {paper.id}; upload one or set an arXiv id."
        )

    def _download_arxiv(self, arxiv_id: str) -> Path:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = self.uploads_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        if dest.exists():
            return dest
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest

    # -- PDF subsetting --------------------------------------------------------

    @staticmethod
    def _subset_pdf(src: Path, pages_1indexed: list[int], dest: Path) -> int:
        """Write a new PDF containing only ``pages_1indexed``. Returns page count."""
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(src))
        total = len(reader.pages)
        writer = PdfWriter()
        wanted = [p for p in pages_1indexed if 1 <= p <= total]
        for p in wanted:
            writer.add_page(reader.pages[p - 1])
        with dest.open("wb") as fh:
            writer.write(fh)
        return len(wanted)

    @staticmethod
    def _pdf_page_count(src: Path) -> int:
        from pypdf import PdfReader

        return len(PdfReader(str(src)).pages)

    @staticmethod
    def _range_pages(scope_json: dict, total_pages: int) -> list[int]:
        """1-indexed pages from page_ranges only (regions handled separately)."""
        kind = (scope_json or {}).get("kind", "full")
        if kind == "full":
            return list(range(1, total_pages + 1))
        pages: set[int] = set()
        for rng in (scope_json.get("page_ranges") or []):
            if not rng:
                continue
            lo, hi = int(rng[0]), int(rng[-1])
            if lo > hi:
                lo, hi = hi, lo
            pages.update(range(lo, hi + 1))
        return sorted(p for p in pages if 1 <= p <= total_pages)

    def _crop_region(self, src: Path, page_1indexed: int, bbox: list[float], dest: Path) -> None:
        """Write a single-page PDF cropped to ``bbox`` (normalized, top-left origin)."""
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import RectangleObject

        reader = PdfReader(str(src))
        page = reader.pages[page_1indexed - 1]
        mb = page.mediabox
        cropbox = scope_utils.bbox_to_cropbox(
            (float(mb.left), float(mb.bottom), float(mb.right), float(mb.top)), bbox
        )
        writer = PdfWriter()
        writer.add_page(page)
        new_page = writer.pages[0]
        new_page.cropbox = RectangleObject(cropbox)
        new_page.mediabox = RectangleObject(cropbox)
        with dest.open("wb") as fh:
            writer.write(fh)

    # -- Marker ----------------------------------------------------------------

    def _call_marker(self, pdf_path: Path) -> str:
        resp = requests.post(
            f"{self.marker_url}/marker", json={"filepath": str(pdf_path)}, timeout=600
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("markdown") or data.get("output") or data.get("text") or ""

    # -- Chunking --------------------------------------------------------------

    @staticmethod
    def _split_markdown(markdown: str) -> list[dict]:
        """Split markdown into heading-delimited chunks (provenance-light: page
        ranges come from the subset as a whole; per-chunk pages are best-effort)."""
        chunks: list[dict] = []
        heading = ""
        buf: list[str] = []

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                chunks.append({"heading": heading, "text": text})

        for line in markdown.split("\n"):
            if re.match(r"^#{1,3}\s+", line):
                flush()
                heading = line.lstrip("#").strip()
                buf = [line]
            else:
                buf.append(line)
        flush()
        return chunks or ([{"heading": "", "text": markdown.strip()}] if markdown.strip() else [])

    # -- Reuse -----------------------------------------------------------------

    def _reuse_succeeded(self, job, prior) -> None:
        """Clone artifacts/chunks from a prior succeeded job onto this job."""
        for art in self.store.artifacts_for_job(prior.id):
            self.store.add_artifact(
                paper_id=job.paper_id, parse_job_id=job.id, kind=art.kind,
                path=art.path, text=art.text, content_hash=art.content_hash,
                page_from=art.page_from, page_to=art.page_to, bbox=art.bbox, meta=art.meta,
            )
        clones = []
        for ch in self.store.chunks_for_job(prior.id):
            clones.append(dict(
                paper_id=job.paper_id, parse_job_id=job.id, ordinal=ch.ordinal,
                kind=ch.kind, heading=ch.heading, text=ch.text,
                page_from=ch.page_from, page_to=ch.page_to, bbox=ch.bbox,
                token_estimate=ch.token_estimate, content_hash=ch.content_hash,
            ))
        if clones:
            self.store.add_chunks(clones)
        self.store.update_parse_job(
            job.id, status=JobStatus.SUCCEEDED.value, marker_version=prior.marker_version,
            cost_estimate=0.0, cost_actual=0.0, error="(reused cached parse)",
            selected_pages=prior.selected_pages,
        )

    # -- Run one job -----------------------------------------------------------

    def run_job(self, job_id: str) -> None:
        job = self.store.get_parse_job(job_id)
        if job is None:
            return
        paper = self.store.get_paper(job.paper_id)
        if paper is None:
            self.store.update_parse_job(job_id, status=JobStatus.FAILED.value, error="paper not found")
            return

        scope = self.store.get_parse_scope(job.parse_scope_id) if job.parse_scope_id else None
        scope_json = scope.scope_json if scope else {"kind": "full"}

        tmp_files: list[Path] = []
        try:
            src = self._resolve_original_pdf(paper)
            pdf_bytes = src.read_bytes()
            pdf_sha = _sha256_bytes(pdf_bytes)
            input_hash = f"{pdf_sha}:{_scope_hash(scope_json)}"

            prior = self.store.find_parse_job_by_input_hash(input_hash)
            if prior and prior.id != job_id:
                logger.info("[parse %s] reuse cached parse from job %s", job_id, prior.id)
                self.store.update_parse_job(job_id, input_hash=input_hash)
                self._reuse_succeeded(job, prior)
                return

            total_pages = self._pdf_page_count(src)
            range_pages = self._range_pages(scope_json, total_pages)
            regions = scope_json.get("regions") or []
            if not range_pages and not regions:     # nothing explicit → whole document
                range_pages = list(range(1, total_pages + 1))

            # Record the original PDF as an artifact (provenance root).
            self.store.add_artifact(
                paper_id=paper.id, parse_job_id=job.id, kind=ArtifactKind.ORIGINAL_PDF.value,
                path=str(src), content_hash=pdf_sha, page_from=1, page_to=total_pages,
                meta={"total_pages": total_pages},
            )

            chunk_rows: list[dict] = []
            ordinal = 0

            # 1) Page-range subset → one Marker pass over those whole pages.
            if range_pages:
                fd, tmp_name = tempfile.mkstemp(suffix=".pdf", prefix="subset_",
                                                dir=str(self.uploads_dir) if self.uploads_dir.exists() else None)
                os.close(fd)
                tmp_subset = Path(tmp_name)
                tmp_files.append(tmp_subset)
                self._subset_pdf(src, range_pages, tmp_subset)
                self.store.add_artifact(
                    paper_id=paper.id, parse_job_id=job.id, kind=ArtifactKind.SUBSET_PDF.value,
                    path=str(tmp_subset), content_hash=_sha256_bytes(tmp_subset.read_bytes()),
                    page_from=range_pages[0], page_to=range_pages[-1], meta={"pages": range_pages},
                )
                markdown = self._call_marker(tmp_subset)
                if not markdown.strip():
                    raise RuntimeError("Marker returned empty markdown for page subset")
                self.store.add_artifact(
                    paper_id=paper.id, parse_job_id=job.id, kind=ArtifactKind.MARKER_MARKDOWN.value,
                    text=markdown, content_hash=_sha256_bytes(markdown.encode()),
                    page_from=range_pages[0], page_to=range_pages[-1], meta={"pages": range_pages},
                )
                for rc in self._split_markdown(markdown):
                    chunk_rows.append(dict(
                        paper_id=paper.id, parse_job_id=job.id, ordinal=ordinal,
                        kind="heading" if rc["heading"] else "text",
                        heading=rc["heading"], text=rc["text"],
                        page_from=range_pages[0], page_to=range_pages[-1],
                        token_estimate=estimate_tokens(rc["text"]),
                        content_hash=_sha256_bytes(rc["text"].encode()),
                    ))
                    ordinal += 1

            # 2) Region crops → one Marker pass per labeled rectangle, provenance kept.
            for region in regions:
                page = region.get("page")
                bbox = region.get("bbox")
                label = region.get("label", "other")
                if not isinstance(page, int) or not bbox or page < 1 or page > total_pages:
                    continue
                fd, tmp_name = tempfile.mkstemp(suffix=".pdf", prefix="crop_",
                                                dir=str(self.uploads_dir) if self.uploads_dir.exists() else None)
                os.close(fd)
                tmp_crop = Path(tmp_name)
                tmp_files.append(tmp_crop)
                self._crop_region(src, page, bbox, tmp_crop)
                self.store.add_artifact(
                    paper_id=paper.id, parse_job_id=job.id, kind=ArtifactKind.REGION_CROP.value,
                    path=str(tmp_crop), content_hash=_sha256_bytes(tmp_crop.read_bytes()),
                    page_from=page, page_to=page, bbox=bbox, meta={"label": label},
                )
                region_md = self._call_marker(tmp_crop)
                self.store.add_artifact(
                    paper_id=paper.id, parse_job_id=job.id, kind=ArtifactKind.MARKER_MARKDOWN.value,
                    text=region_md, content_hash=_sha256_bytes(region_md.encode()),
                    page_from=page, page_to=page, bbox=bbox, meta={"label": label},
                )
                if region_md.strip():
                    chunk_rows.append(dict(
                        paper_id=paper.id, parse_job_id=job.id, ordinal=ordinal,
                        kind=label, heading=label.replace("_", " ").title(), text=region_md.strip(),
                        page_from=page, page_to=page, bbox=bbox,
                        token_estimate=estimate_tokens(region_md),
                        content_hash=_sha256_bytes(region_md.encode()),
                    ))
                    ordinal += 1

            if not chunk_rows:
                raise RuntimeError("parse produced no content (empty subset and regions)")
            self.store.add_chunks(chunk_rows)

            n_units = len(range_pages) + len(regions)   # regions billed ~1 page each
            est = estimate_marker_cost(n_units, self.config.marker_price_per_page)
            self.store.update_parse_job(
                job_id, status=JobStatus.SUCCEEDED.value, backend=ParseBackend.MARKER_API.value,
                selected_pages=n_units, input_hash=input_hash, marker_version=MARKER_VERSION,
                cost_estimate=est.cost, cost_actual=est.cost, error="",
            )
            logger.info("[parse %s] succeeded: %d page(s) + %d region(s), %d chunks",
                        job_id, len(range_pages), len(regions), len(chunk_rows))
        except Exception as exc:  # noqa: BLE001 — record failure, don't crash the scheduler
            logger.exception("[parse %s] failed", job_id)
            self.store.update_parse_job(job_id, status=JobStatus.FAILED.value, error=str(exc)[:1000])
        finally:
            fresh = self.store.get_parse_job(job_id)
            succeeded = fresh is not None and fresh.status == JobStatus.SUCCEEDED.value
            # Keep artifact PDFs on success (they're referenced); clean up on failure.
            if not succeeded:
                for f in tmp_files:
                    f.unlink(missing_ok=True)

    def run_pending(self, limit: int = 5) -> int:
        """Drain up to ``limit`` pending parse jobs. Returns the number processed."""
        processed = 0
        while processed < limit:
            job = self.store.claim_next_parse_job()
            if job is None:
                break
            self.run_job(job.id)
            processed += 1
        return processed

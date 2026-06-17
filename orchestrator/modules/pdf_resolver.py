"""
Lightweight Zotero/Koofr PDF resolution helpers.

This module is intentionally smaller than ``ingestion.pdf_fetcher``: it
downloads the Koofr attachment zip and extracts a PDF for viewing/caching, but
does not touch Marker, the job ledger, or markdown caches.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ZOTERO_API_BASE = "https://api.zotero.org"

# Accept every Zotero URI form: web profile (zotero.org/<user>/items/KEY),
# API/library (zotero.org/users/<id>/items/KEY), and the "Copy Zotero URI"
# select form (zotero://select/library/items/KEY).
ZOTERO_PARENT_RE = re.compile(r"zotero(?:\.org|://)[^?#]*?/items/([A-Z0-9]{8})(?:/|$)")
ZOTERO_ATTACH_RE = re.compile(
    r"zotero(?:\.org|://)[^?#]*?/items/[A-Z0-9]{8}/attachment/([A-Z0-9]{8})"
)


@dataclass(frozen=True)
class ResolvedPdf:
    path: Path
    attachment_key: str
    filename: str
    sha256: str


def build_webdav_client(
    *, koofr_user: str | None = None, koofr_app_password: str | None = None
) -> Any:
    from webdav3.client import Client as WebDAVClient

    return WebDAVClient(
        {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": koofr_user if koofr_user is not None else os.environ["KOOFR_USER"],
            "webdav_password": (
                koofr_app_password
                if koofr_app_password is not None
                else os.environ["KOOFR_APP_PASSWORD"]
            ),
        }
    )


def zotero_parent_key(zotero_uri: str) -> str | None:
    match = ZOTERO_PARENT_RE.search(zotero_uri or "")
    return match.group(1) if match else None


def zotero_attachment_key(zotero_uri: str) -> str | None:
    match = ZOTERO_ATTACH_RE.search(zotero_uri or "")
    return match.group(1) if match else None


def resolve_zotero_attachment_key(
    *,
    zotero_uri: str,
    parent_key: str,
    zotero_user_id: str,
    zotero_api_key: str,
) -> str | None:
    if attached := zotero_attachment_key(zotero_uri):
        return attached

    url = f"{ZOTERO_API_BASE}/users/{zotero_user_id}/items/{parent_key}/children"
    req = urllib.request.Request(url, headers={"Zotero-API-Key": zotero_api_key})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            children = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Zotero children fetch failed for parent '{parent_key}'") from exc

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
        return None

    for attach_key, data in pdf_children:
        filename = data.get("filename", "")
        if filename.lower().startswith(parent_key.lower()):
            return attach_key

    pdf_children.sort(key=lambda x: x[1].get("fileSize", 0), reverse=True)
    return pdf_children[0][0]


def extract_pdf_from_zip(
    zip_path: Path, output_path: Path, preferred: str | None = None
) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_entries = [e for e in zf.infolist() if e.filename.lower().endswith(".pdf")]
        if not pdf_entries:
            raise FileNotFoundError(f"No PDF found inside {zip_path}")
        selected = None
        if preferred:
            selected = next(
                (e for e in pdf_entries if Path(e.filename).name == preferred), None
            )
        selected = selected or max(pdf_entries, key=lambda e: e.file_size)
        output_path.write_bytes(zf.read(selected.filename))
        return Path(selected.filename).name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_pdf_filename(filename: str, fallback: str) -> str:
    name = Path(filename or fallback).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    if not name:
        name = Path(fallback).name or "paper.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def download_koofr_pdf(
    *,
    webdav: Any,
    koofr_base: str,
    attachment_key: str,
    output_dir: Path,
    preferred_pdf_filename: str | None = None,
    output_stem: str | None = None,
) -> ResolvedPdf:
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_zip = f"{koofr_base.rstrip('/')}/{attachment_key}.zip"
    if not webdav.check(remote_zip):
        raise FileNotFoundError(f"Koofr zip not found: {remote_zip}")

    with tempfile.TemporaryDirectory(prefix="koofr_pdf_") as tmp:
        tmp_dir = Path(tmp)
        local_zip = tmp_dir / f"{attachment_key}.zip"
        local_pdf = tmp_dir / f"{attachment_key}.pdf"
        webdav.download_sync(remote_path=remote_zip, local_path=str(local_zip))
        extracted_name = extract_pdf_from_zip(
            local_zip, local_pdf, preferred=preferred_pdf_filename
        )
        filename = safe_pdf_filename(
            extracted_name,
            fallback=f"{output_stem or attachment_key}.pdf",
        )
        if output_stem:
            filename = safe_pdf_filename(f"{output_stem}_{filename}", filename)
        dest = output_dir / filename
        dest.write_bytes(local_pdf.read_bytes())

    return ResolvedPdf(
        path=dest,
        attachment_key=attachment_key,
        filename=filename,
        sha256=sha256_file(dest),
    )

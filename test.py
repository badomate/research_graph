#!/usr/bin/env python3
"""
Minimal Koofr WebDAV validator.

What it checks:
1) Connects to Koofr WebDAV
2) Verifies base folder exists (e.g., /zotero)
3) For a given attachment key, checks existence of:
   - /zotero/<KEY>.zip
   - /zotero/<KEY>.prop
4) Optionally lists the first N entries in the base folder.

Usage:
  export KOOFR_USER="..."
  export KOOFR_APP_PASSWORD="..."
  export KOOFR_PDF_PATH="/zotero"   # optional
  python koofr_validate.py FT2BINAR --list 20
"""

import argparse
import os
import posixpath
import sys
from time import sleep, time
from typing import Optional
from dotenv import load_dotenv

# Load .env file if present (useful for local development outside Docker)
load_dotenv()
try:
    from webdav3.client import Client as WebDAVClient
    from webdav3.exceptions import ResponseErrorCode
except ImportError:
    print("Missing dependency: webdavclient3. Install: pip install webdavclient3", file=sys.stderr)
    raise




def koofr_join(base: str, name: str) -> str:
    base = "/" + (base or "").strip("/")
    name = (name or "").lstrip("/")
    return posixpath.join(base, name)

def build_client() -> WebDAVClient:
    options = {
        "webdav_hostname": "https://app.koofr.net/dav/Koofr",
        "webdav_login": os.environ["KOOFR_USER"],
        "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
    }
    return WebDAVClient(options)

def check_with_retry(client: WebDAVClient, path: str, attempts: int = 6) -> bool:
    delay = 1.0
    for i in range(1, attempts + 1):
        try:
            return bool(client.check(path))
        except ResponseErrorCode as e:
            # Koofr rate limit
            if getattr(e, "code", None) == 429:
                print(f"[WARN] 429 rate limit on attempt {i}/{attempts} for {path}. Sleeping {delay:.1f}s...")
                sleep(delay)
                delay = min(delay * 2, 20.0)
                continue
            raise
    raise RuntimeError(f"Rate-limited after {attempts} attempts for path={path}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("attachment_key")
    ap.add_argument("--base", default=os.environ.get("KOOFR_PDF_PATH", "/zotero"))
    args = ap.parse_args()

    # Force-correct common misconfig: "/Koofr/zotero" -> "/zotero"
    base = "/" + args.base.strip("/")
    if base.lower().startswith("/koofr/"):
        base = "/" + base.split("/", 2)[-1]  # drop leading "/Koofr"
        print(f"[INFO] Normalized base to: {base}")

    client = build_client()

    print("[INFO] Host:", "https://app.koofr.net/dav/Koofr")
    print("[INFO] Base:", base)

    try:
        if check_with_retry(client, base):
            print("[OK] Base exists:", base)
        else:
            print("[FAIL] Base missing:", base)
            return 2
    except Exception as e:
        print("[ERROR] Base check failed:", type(e).__name__, str(e))
        return 3

    zip_path = koofr_join(base, f"{args.attachment_key}.zip")
    prop_path = koofr_join(base, f"{args.attachment_key}.prop")

    for p, label in [(zip_path, "ZIP"), (prop_path, "PROP")]:
        try:
            exists = check_with_retry(client, p)
            print(f"[{'OK' if exists else 'MISS'}] {label}: {p}")
        except Exception as e:
            print(f"[ERROR] {label} check failed: {p} :: {type(e).__name__}: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
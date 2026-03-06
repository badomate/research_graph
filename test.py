#!/usr/bin/env python3
"""
notion_smoketest.py

A minimal Notion API smoke test that works across *any* database:
- Reads NOTION_TOKEN and NOTION_DATABASE_ID from env
- Retrieves DB schema
- Finds the *actual* title property key (never assumes "Name")
- Creates a page with ONLY the title (guaranteed-valid)
- Appends one paragraph block
- Queries DB to confirm the page exists
"""

import os
import sys
from datetime import datetime, timezone

from notion_client import Client
from notion_client.errors import APIResponseError

from dotenv import load_dotenv


# Load .env file if present (useful for local development outside Docker)
load_dotenv()

def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def find_title_prop_name(db: dict) -> str:
    props = db.get("properties", {})
    for prop_name, prop in props.items():
        if prop.get("type") == "title":
            return prop_name
    die("No title property found in database schema.")


def title_value(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text[:2000]}}]}


def paragraph_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def main() -> None:
    token = os.getenv("NOTION_TOKEN")
    db_id = "3198006826a680358e69d0a09fac9e81"

    if not token:
        die("Missing env NOTION_TOKEN")
    if not db_id:
        die("Missing env NOTION_DATABASE_ID (the database id)")

    client = Client(auth=token)

    try:
        db = client.databases.retrieve(database_id=db_id)
    except APIResponseError as e:
        die(f"Failed to retrieve database. {e}")

    prop_types = {k: v.get("type") for k, v in db.get("properties", {}).items()}
    title_key = find_title_prop_name(db)

    print("=== Database schema ===")
    print(f"DB id: {db_id}")
    print(f"Title property key: {title_key}")
    print("Property types:")
    for k, t in prop_types.items():
        print(f"  - {k}: {t}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    page_title = f"Notion smoke test @ {now}"

    properties = {title_key: title_value(page_title)}

    print("\n=== Creating page ===")
    print(f"Outgoing property keys: {list(properties.keys())}")

    try:
        page = client.pages.create(
            parent={"database_id": db_id},
            properties=properties,
        )
    except APIResponseError as e:
        die(f"Failed to create page. {e}")

    page_id = page["id"]
    page_url = page.get("url", "<no url returned>")
    print(f"Created page_id: {page_id}")
    print(f"URL: {page_url}")

    print("\n=== Appending a paragraph block ===")
    try:
        client.blocks.children.append(
            block_id=page_id,
            children=[paragraph_block("If you can read this, your Notion integration works.")],
        )
    except APIResponseError as e:
        die(f"Failed to append block children. {e}")

    print("Appended block OK.")

    print("\n=== Querying DB to confirm page exists ===")
    try:
        res = client.databases.query(database_id=db_id, page_size=5)
        titles = []
        for r in res.get("results", []):
            # read title safely
            tprop = r["properties"].get(title_key, {})
            parts = tprop.get("title", [])
            txt = "".join(p.get("plain_text", "") for p in parts)
            titles.append(txt)
        print("Latest pages (first 5):")
        for t in titles:
            print(f"  - {t}")
    except APIResponseError as e:
        die(f"Failed to query database. {e}")

    print("\nOK")


if __name__ == "__main__":
    main()
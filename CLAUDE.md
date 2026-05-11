# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`paper_pipeline` is a Python orchestration system that extracts mathematical concepts from academic PDFs and builds a knowledge graph in Notion. Papers flow through a Zotero → Notion → Python pipeline: PDFs are OCR'd via Marker, concepts are extracted with Claude (via `instructor`), semantically indexed in Qdrant, linked across papers, and promoted to a "Second Brain" after human review.

## Running the Project

**Docker (recommended):**
```bash
docker compose --profile marker-api up --build       # cloud Marker API
docker compose --profile marker-local-cpu up --build # local CPU
docker compose --profile marker-local-gpu up --build # local GPU
docker compose logs -f orchestrator
docker compose down
```

**Local dev:**
```bash
cd orchestrator
pip install -r requirements.txt
python db_init.py   # initialize SQLite ledger (run once)
python main.py      # start APScheduler
```

**One-off utility scripts (run from project root):**
```bash
python run_promotion.py
python run_deferred_edges.py
python run_reprocess_paper.py
python run_repromote_edges.py
python run_grapher.py
python run_backfill_ki_props.py
```

**Tests:**
```bash
pytest tests/test_completeness.py
pytest tests/test_edge_quality.py
```

## Architecture

### Three-Stage Ingestion Pipeline

Every paper runs through three sequential stages tracked in the SQLite job ledger (`job_ledger.py`):

| Stage | Module | What it does | LLM call |
|-------|--------|-------------|----------|
| 1 — EXTRACT | `ingestion.py` | PDF → Marker markdown → Claude extracts `MathObject` list | Yes (1 call) |
| 2 — RETRIEVE | `ingestion.py` + `vector_index.py` | Per concept: Qdrant ANN + 4 structural signals → top-K candidates | No |
| 3 — LINK | `ingestion.py` | Per concept + candidates → `CrossPaperLinkResult` edges | Yes (1 call) |

Checkpoints in the ledger: `marker_done → extract_done → retrieve_done → link_done → notion_done`. Crashes restart from the last checkpoint; the pipeline is fully idempotent.

### Schedulers (APScheduler, configured in `main.py`)

- **ingestion** — every 1 min: picks papers with status `s1-processing`, runs 3-stage pipeline
- **promotion** — every 1 min: promotes reviewed KI concepts to Second Brain, resolves deferred edges
- **arxiv_sniper** — daily 06:00 UTC: keyword-based ArXiv auto-ingest
- **dependency_grapher** — every 1 min: builds interactive HTML concept graphs

### Key Modules (`orchestrator/modules/`)

- `ingestion.py` — 3-stage pipeline core
- `promotion.py` — Knowledge Inbox → Second Brain promotion + edge resolution
- `vector_index.py` — Qdrant (3 collections: full/assumptions/conclusions) with OpenAI or local embeddings; falls back to TF-IDF silently if Qdrant is offline
- `notion_client_wrapper.py` — rate-limited (3 req/s) Notion client with exponential backoff
- `job_ledger.py` — SQLite idempotency tracker
- `extraction_schema.py` — Pydantic v2 models for all LLM-structured outputs
- `conflict_detector.py`, `latex_compiler.py` — disabled/experimental

### Paper Status State Machine

Papers in the Notion Paper Tracker DB move through statuses that trigger orchestrator actions:
```
s0-inbox → s1-skim → s1-processing → s2-extracted → s2-read → s3-distilled
                   ↳ s1b-waiting-attachment
                   ↳ blocked-extraction
                              ↳ s2-reextract (loops back)
```

### Notion as the Source of Truth

All application state lives in Notion (5 databases: Paper Tracker, Knowledge Inbox, Second Brain, Edges DB, Projects DB). The SQLite ledger only tracks pipeline idempotency, not application state. See `NOTION_DBS.md` for full schema.

## Configuration

Copy `.env.example` to `.env`. Required variables:

```bash
NOTION_TOKEN
NOTION_PAPER_TRACKER_DB_ID
NOTION_KNOWLEDGE_INBOX_DB_ID
NOTION_SECOND_BRAIN_DB_ID
NOTION_PROJECTS_DB_ID
NOTION_EDGES_DB_ID
ANTHROPIC_API_KEY
KOOFR_USER / KOOFR_APP_PASSWORD   # WebDAV PDF storage
ZOTERO_USER_ID / ZOTERO_API_KEY
```


Key optional settings:
- `CLAUDE_MODEL` (default: `claude-sonnet-4-6`)
- `VECTOR_INDEX_ENABLED` — set to enable Qdrant; otherwise TF-IDF is used
- `VECTOR_EMBEDDING_BACKEND` — `openai` (default) or `local` (allenai/specter2)
- `RETRIEVE_CANDIDATES_K` (default: 30)
- `EDGE_AUTO_CREATE_CONFIDENCE`, `EDGE_REVIEW_FLAG_CONFIDENCE` — confidence thresholds for edge tiers

## LLM Integration

Claude is called via the `instructor` library for structured output against Pydantic v2 schemas defined in `extraction_schema.py`. Do not change schema field names without verifying all downstream Notion property mappings — these are tightly coupled.

Tags for papers are validated against `tags_registry.yaml` by `tag_linter.py`.

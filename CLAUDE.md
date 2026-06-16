# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`paper_pipeline` extracts mathematical concepts from academic PDFs and builds a knowledge graph you browse in a local web app. Papers flow through a Zotero/arXiv → SQLite → Python pipeline: PDFs are OCR'd via Marker, concepts are extracted with Claude (via `instructor`), semantically indexed in Qdrant, linked across papers, and promoted to a "Second Brain" after human review.

**All application state lives in a single SQLite database** (the `Store`); a server-rendered **web UI** (`webapp/`) is the interface for adding papers, reviewing concepts, accepting/rejecting edges, and browsing the graph. (This replaced an earlier Notion-backed design — there is no Notion dependency anymore.)

## Running the Project

**Docker (recommended) — one command brings up everything:**
```bash
docker compose --profile marker-api up --build       # cloud Marker API
docker compose --profile marker-local-cpu up --build # local CPU
docker compose --profile marker-local-gpu up --build # local GPU
# then open the web UI:
open http://localhost:8000
docker compose logs -f orchestrator
docker compose down
```
`orchestrator` (scheduler) and `webapp` (UI) share the `app_data` volume holding the SQLite DB + uploaded PDFs. The schema auto-creates on first start.

**Local dev (two processes sharing one DB):**
```bash
# 1) Web UI
pip install -r webapp/requirements.txt
DATABASE_URL=sqlite:///./app.db uvicorn webapp.main:app --reload   # http://127.0.0.1:8000
python -m webapp.seed     # optional: demo data

# 2) Orchestrator scheduler
cd orchestrator && pip install -r requirements.txt
DATABASE_URL=sqlite:///../app.db python main.py
```

**Tests:**
```bash
cd orchestrator && pytest tests/        # unit + import smoke tests (deps stubbed)
```

## Architecture

### SQLite as the source of truth

All state lives in one SQLite file via `orchestrator/modules/store/` (SQLModel). The unified schema replaces the old Notion databases:

| Table | Replaces | Notes |
|-------|----------|-------|
| `papers` | Paper Tracker | status state machine, intake fields, extraction bookkeeping |
| `concepts` | Knowledge Inbox **+** Second Brain | one table; `state` = `inbox`/`promoted`/`hub` (promotion is a state flip) |
| `edges` | Edges DB **+** Deferred Edges **+** "Edge Suggestions" | first-class rows; `status` = `proposed`/`verified`/`rejected` |
| `ingestion_jobs` | (unchanged) | `job_ledger.py` idempotency tracker |

`Store` (`store/repository.py`) is the only data-access layer; both the pipeline and the web app use it. WAL mode lets the two processes share the file.

### Three-Stage Ingestion Pipeline

Every paper runs three sequential stages (checkpoints in the job ledger: `marker_done → extract_done → retrieve_done → link_done → notion_done`; crashes resume from the last checkpoint):

| Stage | Module | What it does | LLM call |
|-------|--------|-------------|----------|
| 1 — EXTRACT | `ingestion/engine.py` + `extractor.py` | PDF → Marker markdown → Claude extracts `MathObject` list → `concepts` rows | Yes |
| 2 — RETRIEVE | `ingestion/retriever.py` + `vector_index.py` | Per concept: Qdrant ANN + structural signals → top-K candidates | No |
| 3 — LINK | `ingestion/linker.py` | Per concept + candidates → proposed `edges` rows (single calls, or one Batch with `LINK_USE_BATCH_API`) | Yes |

### Schedulers (APScheduler, `orchestrator/main.py`)

- **ingestion** — every 10 min: papers at `s1-skim` / `s2-reextract`, runs the 3-stage pipeline (also once at startup)
- **promotion** — every 30 min: promotes verified concepts to the Second Brain, verifies ready auto-edges (also once at startup)
- **arxiv_sniper** — daily 06:00 UTC: keyword ArXiv auto-ingest → `papers` rows
- **zotero_intake** — every `ZOTERO_POLL_MINUTES` (only if `ZOTERO_POLL_ENABLED`): imports new Zotero items (replaces Notero)

### Paper Status State Machine

```
s0-inbox → s1-skim → s1-processing → s2-extracted → s2-read → s3-distilled
                   ↳ s1b-waiting-attachment
                   ↳ blocked-extraction
                              ↳ s2-reextract (loops back)
```
Human transitions (set status, verify/reject concepts, accept/reject edges) happen in the web UI. Intake: in-app Add Paper (arXiv ID / DOI / PDF upload) and/or the Zotero poller.

### Key Modules

- `orchestrator/modules/store/` — SQLModel models (`models.py`), engine/WAL session (`db.py`), `Store` repository (`repository.py`)
- `ingestion/` — `engine.py` (orchestrator), `concept_writer.py` (concepts + edges), `retriever.py`, `linker.py`, `pdf_fetcher.py`
- `promotion/engine.py` — inbox → Second Brain promotion + edge verification; `zotero_sync.py` (notes → `paper.ai_notes`)
- `metadata.py` — arXiv/Crossref metadata fetch for Add Paper; `zotero_intake.py` — Zotero poller; `arxiv_sniper.py`
- `vector_index.py` — Qdrant (3 collections) with OpenAI or local embeddings; TF-IDF fallback if offline
- `extraction_schema.py` — Pydantic v2 models for all LLM-structured outputs
- `webapp/` — FastAPI + Jinja + HTMX UI (see `webapp/README.md`)

## Configuration

Copy `.env.example` to `.env`. The app **boots with zero config** (SQLite + sensible defaults); set what you use:

```bash
DATABASE_URL          # default sqlite:////data/app.db (Docker volume)
ANTHROPIC_API_KEY     # required for extraction/linking
OPENAI_API_KEY        # embeddings (if VECTOR_EMBEDDING_BACKEND=openai)
KOOFR_USER / KOOFR_APP_PASSWORD     # Zotero-attachment PDF path (optional; uploads bypass it)
ZOTERO_USER_ID / ZOTERO_API_KEY     # Zotero intake + notes (optional)
ZOTERO_POLL_ENABLED / ZOTERO_POLL_MINUTES
```

Key optional settings:
- `CLAUDE_MODEL` (default: `claude-sonnet-4-6`)
- `VECTOR_INDEX_ENABLED`, `VECTOR_EMBEDDING_BACKEND` (`openai` or `local`), `RETRIEVE_CANDIDATES_K`
- `EDGE_AUTO_CREATE_CONFIDENCE`, `EDGE_REVIEW_FLAG_CONFIDENCE`
- `LINK_USE_BATCH_API` — Stage-3 linking via the Anthropic Batches API (50% cheaper; see `linker.py`)

## LLM Integration

Claude is called via the `instructor` library for structured output against Pydantic v2 schemas in `extraction_schema.py`. System prompts use prompt caching. Do not change schema field names without checking the downstream `Store`/`concept_writer` mappings — they are tightly coupled. Tags are validated against `tags_registry.yaml` by `tag_linter.py`.

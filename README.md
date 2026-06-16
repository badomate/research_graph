# paper_pipeline

Extracts mathematical concepts from academic PDFs and builds a reviewable knowledge
graph you browse in a **local web app**. Papers flow Zotero/arXiv → SQLite → Python:
PDFs are OCR'd via Marker, concepts extracted with Anthropic Claude, semantically indexed
in Qdrant, linked across papers, and promoted to a "Second Brain" after human review.

State lives in a single **SQLite** database; the **web UI** (`webapp/`) is how you add
papers, review concepts, accept/reject edges, and explore the graph. *(Earlier versions
used Notion — that dependency has been removed.)*

## Quick start

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY; the rest is optional
docker compose --profile marker-api up --build
# open the UI:
open http://localhost:8000
```

`orchestrator` (the scheduler) and `webapp` (the UI) share the `app_data` volume holding
the SQLite DB + uploaded PDFs. The schema auto-creates on first run. Marker has three
profiles: `marker-api` (Datalab cloud), `marker-local-cpu`, `marker-local-gpu`.

### Local dev (no Docker)

```bash
# Web UI
pip install -r webapp/requirements.txt
DATABASE_URL=sqlite:///./app.db uvicorn webapp.main:app --reload   # http://127.0.0.1:8000
python -m webapp.seed         # optional demo data

# Orchestrator (separate shell, same DB)
cd orchestrator && pip install -r requirements.txt
DATABASE_URL=sqlite:///../app.db python main.py
```

## How it works

1. **Add a paper** in the UI (arXiv ID / DOI / PDF upload) or let the Zotero poller /
   ArXiv sniper add them. It lands at status `s1-skim`.
2. **Ingestion** (every 10 min) runs the 3-stage pipeline — extract concepts (Claude),
   retrieve candidates (Qdrant), propose edges (Claude) — and moves the paper to
   `s2-extracted`.
3. **Review** in the UI: verify/reject concepts, accept/reject proposed edges, edit
   inline. Set the paper to `s2-read`.
4. **Promotion** (every 30 min) flips verified concepts into the Second Brain and
   verifies their auto-edges; the paper reaches `s3-distilled`.

## Docs

- **[CLAUDE.md](./CLAUDE.md)** — architecture, data model, modules, configuration (start here).
- **[webapp/README.md](./webapp/README.md)** — the web UI: pages, routes, keyboard shortcuts.
- **[DATA_MODEL.md](./DATA_MODEL.md)** — tables + the paper status state machine.

## Stack

| Component | Technology |
|-----------|-----------|
| State / source of truth | SQLite (SQLModel) |
| Web UI | FastAPI + Jinja + HTMX (server-rendered) |
| Reference manager (optional) | Zotero (poller replaces the Notero plugin) |
| PDF storage (Zotero path) | Koofr WebDAV |
| OCR / PDF→Markdown | Marker (Datalab cloud, or local CPU/GPU) |
| Concept extraction + linking | Anthropic Claude (via `instructor`) |
| Embeddings | OpenAI `text-embedding-3-small` or `allenai/specter2` (local) |
| Vector retrieval | Qdrant (3 role-specific collections), TF-IDF fallback |
| Orchestration | Python + APScheduler |
| Containerisation | Docker Compose |

## Tests

```bash
cd orchestrator && pytest tests/     # unit + import smoke tests (external deps stubbed)
```

## Security notes

- Never commit `.env` (it's git-ignored). `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` should be
  scoped to a project with spend limits.
- The SQLite DB and uploaded PDFs live in the `app_data` Docker volume (local data only).
- Koofr app passwords are separate from your account password and independently revocable.

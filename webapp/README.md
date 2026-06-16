# webapp — local UI (replaces Notion)

Server-rendered FastAPI + Jinja + HTMX. Reads/writes the same SQLite database the
orchestrator uses (`orchestrator/modules/store`). Research-clean, mobile-friendly,
keyboard-driven review.

## Run locally

```bash
pip install "fastapi" "uvicorn[standard]" "jinja2" "python-multipart"

# optional: load demo data so there's something to click through
DATABASE_URL="sqlite:///./app.db" python -m webapp.seed

# start the UI
DATABASE_URL="sqlite:///./app.db" uvicorn webapp.main:app --reload
# open http://127.0.0.1:8000
```

`DATABASE_URL` defaults to `sqlite:///./app.db` if unset. Delete the `.db` file to reseed.

## Pages

| Route | What |
|---|---|
| `/` | Papers, grouped by status |
| `/papers/new` | Add a paper (arXiv ID / DOI / PDF upload) |
| `/papers/{id}` | Paper detail + status actions |
| `/review`, `/papers/{id}/review` | Concept review — verify/reject, accept/reject edges, inline edit |
| `/brain` | Second Brain (promoted concepts, grouped by hub) + filter |
| `/concepts/{id}` | Concept detail + connections |
| `/graph` | Interactive vis-network graph (`/api/graph.json`) |
| `/search?q=` | Concept search |

## Review keyboard shortcuts

`j`/`k` move · `v` verify · `x` reject · `e` edit. (Ignored while typing in a field.)

## Notes / TODO (next phases)

- Math renders via KaTeX **auto-render** — only delimited math (`$…$`, `$$…$$`) is
  typeset; bare-LaTeX statements show raw. Revisit if extractions omit delimiters.
- Vendors HTMX / KaTeX / vis-network from CDNs — needs internet on first paint.
- Still pending: **Phase 2** (pipeline cutover so the orchestrator writes here instead
  of Notion), **Phase 4** (real arXiv/DOI metadata fetch + Zotero poller + PDF→pipeline),
  **Phase 5** (docker-compose `webapp` service, drop `graph-server`, docs).

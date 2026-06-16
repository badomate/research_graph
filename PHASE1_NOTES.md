# Phase 1 — implementation notes

This delivers the Phase-1 slice of the research-OS spec on the **existing SQLite +
SQLModel** stack (your prompt said MySQL; the codebase is SQLite and we kept it —
single-user, zero-config, WAL-shared between the webapp and orchestrator). The
central rule is enforced in the data layer: **AI output is a suggestion** — it is
quarantined, reviewable, regeneratable, versioned, and traceable back to the exact
paper scope that produced it.

## What's included

| Spec area | Where |
|-----------|-------|
| Projects / collections (nested) / tags, multi-membership, **roles** | `store/models.py`, `store/research_repo.py`; UI `/projects`, `/collections`, `/tags`, paper "Organize…" panel |
| Unified **local search** (metadata + parsed chunks, "why it matched") | `Store.search_papers`; UI `/find` |
| **Parse scopes** + selective page parsing (subset PDF → Marker) | `parsing/scope_utils.py`, `parsing/parse_worker.py`; UI `/papers/{id}/scope` |
| Artifacts + chunks (provenance) | `paper_artifacts`, `paper_chunks` tables |
| **Analysis jobs** (parsed chunks → Claude, never raw PDF) | `analysis/analysis_worker.py`, `analysis/prompts.py` |
| **AI-suggestion quarantine** + accept/reject/edit + promotion to first-class tables | `ai_suggestions`; `Store.accept_suggestion` etc.; UI `/suggestions` |
| **Regenerate** every AI item, with lineage + instruction + version compare | `Store.regenerate_suggestion`, `suggestion_versions`; per-card UI |
| **Cost estimation** (Marker per-page + Claude token band) | `modules/cost.py`; live preview in scope builder + `/api/cost/estimate` |
| Math objects (first-class, promotable) | `math_objects`, `math_object_projects` |
| Project **novelty dashboard** scaffold | `/projects/{id}/novelty` |
| **Migrations** | Alembic (`orchestrator/alembic/`) |

Phase 2/3 items (region drawing + crop parsing, external arXiv/S2/Crossref search,
skip-proof auto-suggester, full novelty aggregation, version-diff UI) are scoped
but **not** in this pass — clean seams are left for them.

## Running

Nothing new is required to boot. New tables auto-create via `create_all` on a
fresh DB. For an **existing** DB, apply the additive migration:

```bash
cd orchestrator
pip install -r requirements.txt              # adds alembic + pypdf
DATABASE_URL=sqlite:///../app.db alembic upgrade head
```

The migration is additive and inspector-guarded (safe whether or not `create_all`
already built the tables; it also adds the new `projects.status/priority/updated_at`
columns).

Web UI + orchestrator as before:

```bash
DATABASE_URL=sqlite:///./app.db uvicorn webapp.main:app --reload
cd orchestrator && DATABASE_URL=sqlite:///../app.db python main.py
```

A new scheduler job, **"Parse + Analysis Job Workers"** (every 1 min), drains
pending `parse_jobs` and `analysis_jobs`. The webapp only **enqueues** jobs (stays
light); the orchestrator does the heavy Marker/Claude work.

## Configuration (new env vars, all optional)

```bash
MARKER_PRICE_PER_PAGE=0.01            # Marker/Datalab per-page rate
CLAUDE_INPUT_PRICE_PER_MTOK=3.0       # $/million input tokens (set to current pricing)
CLAUDE_OUTPUT_PRICE_PER_MTOK=15.0     # $/million output tokens
```

Cost figures are **estimates** with user-set rates — vendor pricing changes, so
they are config, never hard-coded in logic. Set them to the current Anthropic /
Datalab numbers for accurate previews. `ANTHROPIC_API_KEY` is required for
analysis jobs to run (jobs fail with a clear error if unset, never crash).

## Day-to-day flow

1. **Add/organize** a paper → assign to projects/collections/tags with a role
   (core, direct_competitor, baseline, …) from the paper page.
2. **Selective parse** (`Selective parse / analyze`): enter page ranges like
   `1-8, 35-40`; see selected-page count + Marker estimate; queue the parse. The
   worker subsets the PDF (pypdf), sends only those pages to Marker, and stores
   artifacts + chunks. Re-parsing the same PDF+scope is deduped by `input_hash`.
3. **Analyze**: pick an analysis type (triage, theorem extraction, novelty risk,
   …) over the parsed chunks. Output lands in **Suggestions**, never the accepted DB.
4. **Review** suggestions: Accept (promotes to `math_objects` / `concepts` /
   project membership / paper summary), Reject (kept for provenance), or
   **Regenerate** with an optional instruction ("ignore proofs", "be more
   mathematical"). Each regeneration is a new version; the old one is superseded
   only when you accept the new one. Full lineage + input/output hashes are shown.

## Provenance / reproducibility

Every suggestion records `model`, `prompt_version`, `input_hash`, `output_hash`,
the `analysis_job_id` (→ `chunk_ids` → `parse_job_id` → scope → pages → original
PDF artifact), `parent_generation_id`, and—once accepted—`promoted_ref_table` /
`promoted_ref_id`. That chain answers: where did this come from, which pages,
which prompt/model, can I regenerate it, did I accept it, is it superseded.

## Tests

```bash
cd orchestrator && pytest tests/        # 94 passing
```

New: `test_research_store.py` (organization, roles, parse-job dedup, suggestion
accept/promote, **regeneration lineage end-to-end** through the analysis worker
with a stubbed Claude client), `test_cost.py`, `test_scope_utils.py`.

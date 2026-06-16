# Phase 2 — implementation notes

Builds on Phase 1 (same SQLite + SQLModel stack). Adds external discovery,
region-level selective parsing, the skip-proof helper, and semantic search.

## What's included

| Spec area | Where |
|-----------|-------|
| **External search** — arXiv + Semantic Scholar + Crossref, merged/de-duped, never auto-saved | `modules/external_search.py`; UI `/external` with Save / Save+parse / Save+triage, add-to-project/collection, BibTeX helper |
| **Region selection** — draw labeled rectangles on the PDF, normalized bboxes | `webapp/templates/regions.html` (PDF.js); route `/papers/{id}/regions`, `/papers/{id}/scopes/regions` |
| **Crop-region parsing** — each region → cropped single-page PDF → Marker, provenance kept | `parsing/parse_worker.py` (`_crop_region`), `scope_utils.bbox_to_cropbox` |
| **Skip-proof helper** — proposes a scope that keeps intro/assumptions/theorems/results and skips proofs/appendices | `parsing/skip_proof.py`; UI in the scope builder (`/papers/{id}/scope/suggest`) |
| **Semantic search** — vector ranking over parsed chunks, resolves to SQLite rows | `modules/semantic_search.py`; `/find?mode=semantic` |

## How each works

**External search** (`modules/external_search.py`) — stdlib `urllib` only (webapp
stays light). Pure parsers (`parse_arxiv_feed`, `parse_semantic_scholar`,
`parse_crossref_search`) are split from the network calls for offline testing.
`merge_results` de-dups across sources by arXiv id → DOI → normalized title and
fills blank fields. Nothing is saved until you click a Save action. "Save + parse"
queues a full parse; "Save + triage" also queues a triage analysis that **defers**
(re-queues itself) until the parse produces chunks.

**Region selection + crop parsing** — `regions.html` renders pages with PDF.js and
lets you drag rectangles, label them (theorem/definition/assumption/…), and submit.
Bboxes are stored normalized (top-left origin, [0,1]) in `scope_json.regions`. The
parse worker converts each bbox to a PDF cropbox (`scope_utils.bbox_to_cropbox`,
unit-tested), writes a single-page `region_crop` artifact, sends it to Marker, and
creates a chunk tagged with the region's label + page + bbox. A scope can be
`mixed` (page ranges **and** regions); page ranges get one Marker pass, each region
its own. Region parsing needs an **uploaded** PDF (arXiv links open externally).

**Skip-proof helper** — paste section lines from the PDF's contents
(`1-8 Introduction`, `9-30 Proof of Theorem 2`); `propose_scope` classifies each
(skip beats keep, so "Proof of the main theorem" is skipped) and returns selected/
skipped pages with reasons, pre-filling the page box. Leave the box empty to derive
sections from a prior parse's chunk headings. It never executes — you review/accept.

**Semantic search** — `/find` has an `exact` ⇄ `semantic` toggle. Semantic mode
ranks parsed chunks: OpenAI embeddings (cosine, cached by chunk `content_hash`)
when `OPENAI_API_KEY` is set, else a pure-Python **TF-IDF cosine** fallback (works
offline). The vector layer is a *derived* index — every hit resolves back to a
`paper_chunks` row and its paper (one result per paper, best chunk).

## Configuration (all optional)

```bash
OPENAI_API_KEY=...    # enables embedding-based semantic search (else TF-IDF)
```

External search needs no keys (public arXiv/S2/Crossref endpoints). Semantic Scholar
is rate-limited unmasked; add a key in `external_search.py` if you hit limits.

## Tests

```bash
cd orchestrator && pytest tests/        # 115 passing
```

New: `test_external_search.py` (parsers + merge/dedup + BibTeX), `test_skip_proof.py`
(classification + proposal), `test_semantic_search.py` (TF-IDF ranking + resolution
to paper/chunk rows), and cropbox cases in `test_scope_utils.py`.

## Not in this pass (Phase 3)

Google/SerpAPI fallback, version-diff UI for regenerations, the full novelty
aggregation (the dashboard scaffold exists), and the advanced claim/concept graph.
The crop pipeline currently bills regions at ~1 page each for the estimate;
per-region actual cost from Marker's `cost_breakdown` is a later refinement.

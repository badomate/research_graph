# Phase 3 — implementation notes

Completes the spec on the SQLite + SQLModel stack. Phase 3 adds the reviewed-truth
aggregations and browsing on top of the Phase 1/2 data model.

## What's included

| Spec area | Where |
|-----------|-------|
| **Project novelty dashboard** (aggregates *accepted* signals) | `Store.project_novelty`; UI `/projects/{id}/novelty` |
| **Math-object browser** (search/filter, detail, project relevance linking) | UI `/math`, `/math/{id}`; uses existing `list_math_objects` / `link_math_object_project` |
| **Version-comparison UI** for regenerated outputs (accept / restore) | `/suggestions/{id}/compare`; `Store.restore_suggestion_version` |
| **Google/SerpAPI fallback** (key-gated extra source) | `external_search.search_serpapi` + `available_sources()` |
| **Advanced claim/concept graph** (project knowledge graph) | `Store.project_graph`; UI `/projects/{id}/graph` + `/api/projects/{id}/graph.json` |

## How each works

**Novelty dashboard** — `project_novelty(project_id)` gathers the project's papers
grouped by role, all **accepted** AI suggestions for those papers grouped by type
(novelty_risk, baseline_candidate, limitation, citation_use, claim, …), and the
math objects linked to the project with their relevance. Only accepted items appear
— the dashboard reflects reviewed truth, not raw AI output. The page surfaces
"weaken novelty" (direct competitors + novelty-risk), baselines, limitations/
assumptions to mention, citations to add, and relevant theorems.

**Math-object browser** — `/math` lists every promoted math object across papers
with type filter + text search; `/math/{id}` shows the statement/assumptions/
conclusion/source quotes and lets you link the object to a project with a relevance
role (direct_tool, competing_result, missing_assumption, possible_extension, …).
Those links feed the novelty dashboard and the project graph.

**Version comparison** — `/suggestions/{id}/compare` lays out the full regeneration
lineage (oldest → newest) side by side: each version's payload, status, the
instruction that produced it, and the model/prompt/output-hash. "Accept this
version" (newest) and "Restore this version" (older) both call
`restore_suggestion_version`, which makes the chosen version the **sole** accepted
one and supersedes every other version in the chain — so reverting truly reverts,
even past a previously-accepted sibling.

**Google/SerpAPI fallback** — `available_sources()` adds `serpapi` (Google Scholar
via SerpAPI) only when `SERPAPI_KEY` is set; the `/external` page and `external_search`
pick it up automatically. `parse_serpapi_scholar` extracts title/authors/year/pdf
and the merge step de-dups it against arXiv/S2/Crossref hits.

**Project knowledge graph** — `project_graph(project_id)` builds a vis-network graph:
the project at the center, its papers (edges labeled by role), each paper's promoted
concepts and math objects (edges to the paper). `/projects/{id}/graph` renders it;
double-click a node to open the paper / math object / concept.

## Configuration (new, optional)

```bash
SERPAPI_KEY=...    # enables Google Scholar results in external search
```

## Tests

```bash
cd orchestrator && pytest tests/        # 122 passing
```

New `test_phase3.py`: novelty aggregation (accepted-only, roles, linked math
objects), project-graph nodes/edges, version lineage (accept retires others;
restore reverts), SerpAPI parsing, and key-gated source list.

## Status across all phases

All 15 spec areas are now implemented on SQLite/SQLModel. Remaining polish that was
explicitly deferred: per-region *actual* Marker cost (regions are estimated at ~1
page each), and a richer cross-paper concept graph beyond the per-project view. The
central rule holds throughout: AI output is a quarantined, versioned, regeneratable,
traceable suggestion — never written straight into the accepted research tables.

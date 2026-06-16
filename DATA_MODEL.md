# Data Model & Status Flow

All state lives in one SQLite database via `orchestrator/modules/store/` (SQLModel).
`Store` (`repository.py`) is the only data-access layer; both the orchestrator and the
web app use it. (This replaced six Notion databases.)

## Tables

### `papers`
One row per paper. Key fields: `status` (state machine below), `source`
(`zotero`/`arxiv`/`manual`), intake ids (`zotero_key`, `attachment_key`, `arxiv_id`,
`doi`), `pdf_path` (set for uploads — bypasses Koofr/Zotero), `pdf_sha256`, and
extraction bookkeeping (`one_liner`, `active_themes`, `extraction_count`,
`extraction_tokens`, `rejected_concepts`, `reextract_hints`, `extraction_error`,
`ai_notes`).

### `concepts`  (Knowledge Inbox **+** Second Brain, unified)
One row per extracted concept. `state` distinguishes them:

| `state` | meaning |
|---------|---------|
| `inbox` | freshly extracted, under review |
| `promoted` | in the Second Brain (promotion is a state flip, not a copy) |
| `hub` | a knowledge hub node |

Content mirrors the `MathObject` schema: `type`, `title`/`corrected_title`,
`statement_latex`, `assumptions`, `variables`, `conclusion`, `interpretation`,
`proof_idea`, `source_quote`, `source_pages`, `source_anchors`, `suggested_hub`,
`result_category`, `named_tools`, `setting`, `canonical_keywords`, `prereq_keywords`,
`downstream_keywords`, `ai_confidence`. Review fields: `verification_status`
(`unverified`/`verified`/`rejected`), `graph_link_status`, `reviewer_notes`.
`effective_title` = `corrected_title or title`.

### `edges`  (Edges DB **+** Deferred Edges **+** the old "Edge Suggestions" JSON)
First-class rows: `source_concept_id` → `target_concept_id`, `relation_type`
(`depends_on`/`enables`/`generalizes`/`special_case_of`/`related`), `channel`
(`auto`/`suggest`), `status` (`proposed`/`verified`/`rejected`), `ai_confidence`,
`justification`, `falsifiability`, `driving_fields`, `needs_review`.

- Stage 3 writes proposals as `status='proposed'`.
- The review UI flips them to `verified`/`rejected` (one click).
- On promotion, `channel='auto'` edges whose **both** endpoints are promoted are
  auto-verified (`verify_auto_edges_between_promoted()`) — this replaces the old
  title-based deferred-edge resolution.

### `ingestion_jobs`
Unchanged idempotency ledger (`job_ledger.py`). Key: `(zotero_key/paper_id,
pdf_sha256, extraction_version)`. Checkpoints: `marker_done → extract_done →
retrieve_done → link_done → notion_done`.

## Paper status state machine

```
s0-inbox → s1-skim → s1-processing → s2-extracted → s2-read → s3-distilled
                   ↳ s1b-waiting-attachment
                   ↳ blocked-extraction
                              ↳ s2-reextract (loops back to s2-extracted)
```

| Status | Set by | Meaning |
|--------|--------|---------|
| `s0-inbox` | intake (Zotero/arXiv) | arrived, not yet queued |
| `s1-skim` | human / Add Paper | queued for extraction |
| `s1-processing` | pipeline | claimed (race guard) |
| `s1b-waiting-attachment` | pipeline | Koofr zip not found; retries |
| `blocked-extraction` | pipeline | 0 concepts; add hints, reset to `s1-skim` |
| `s2-extracted` | pipeline | concepts + proposed edges written; ready to review |
| `s2-reextract` | human | targeted re-extraction with hints |
| `s2-read` | human | review done; triggers promotion |
| `s3-distilled` | pipeline | promoted to Second Brain |

Human actions (set status, verify/reject concepts, accept/reject edges, add papers)
happen in the web UI; see [webapp/README.md](./webapp/README.md).

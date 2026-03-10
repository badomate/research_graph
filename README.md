# paper_pipeline

A Python orchestration pipeline that ingests academic papers from Zotero/Notero into
Notion, extracts mathematical concepts via Marker OCR and Anthropic Claude, and populates
a structured Knowledge Inbox for human review and promotion to a Second Brain knowledge
graph.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            paper_pipeline                                   │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │ Zotero +     │    │               Notion (cloud)                     │  │
│  │ Notero plugin│───>│  Paper Tracker DB   │   Knowledge Inbox DB       │  │
│  └──────────────┘    │  (status machine)   │   (extracted concepts)     │  │
│                      └──────────┬──────────┴──────────────┬─────────────┘  │
│                                 │ poll s1-skim              │ write          │
│                      ┌──────────▼──────────────────────────▼─────────────┐ │
│                      │          Python Orchestrator (APScheduler)        │ │
│                      │                                                   │ │
│                      │  IngestionEngine  PromotionEngine  ArXivSniper   │ │
│                      │  LaTeXCompiler    DependencyGrapher               │ │
│                      └────┬──────────────────────────┬───────────────────┘ │
│                           │ POST /marker              │ Anthropic API       │
│                      ┌────▼──────────┐          ┌────▼──────────┐         │
│                      │  marker-api   │          │  Claude (LLM) │         │
│                      │  (OCR proxy)  │          │  extraction + │         │
│                      └───────────────┘          │  edge linking │         │
│                                                 └───────────────┘         │
│  ┌───────────────────────────────────┐                                     │
│  │  Qdrant  (vector DB, local)       │  semantic concept retrieval         │
│  │  3 collections (full/assumptions/ │  ANN search + pre-filter scoring    │
│  │  conclusions)                     │  falls back to TF-IDF if offline   │
│  └───────────────────────────────────┘                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Koofr WebDAV  (PDF zips stored as {ZoteroKey}.zip)                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stack

| Component | Technology |
|-----------|-----------|
| Reference manager | Zotero + Notero plugin |
| Knowledge base | Notion |
| PDF storage | Koofr WebDAV |
| OCR / PDF→Markdown | Marker (Datalab cloud API, or local Docker CPU/GPU) |
| Concept extraction + edge linking | Anthropic Claude (via `instructor`) |
| Embeddings | OpenAI `text-embedding-3-small` (default) or `allenai/specter2` (local) |
| Vector retrieval | Qdrant (3 role-specific collections) |
| Fuzzy pre-filter scoring | `rapidfuzz` |
| Orchestration | Python + APScheduler |
| Data validation | Pydantic v2 |
| Idempotency store | SQLite (job ledger) |
| Containerisation | Docker Compose |

---

## Setup

### Prerequisites

- Docker and Docker Compose
- A Notion integration token with access to:
  - Paper Tracker database
  - Knowledge Inbox database
  - Second Brain database
  - Projects database
  - Edges database
  - Deferred Edges database (optional, for edge resolution across promotion batches)
- Koofr account with WebDAV app password
- Anthropic API key (Claude) — used for concept extraction and edge linking
- OpenAI API key — used for embeddings only (or set `VECTOR_EMBEDDING_BACKEND=local`)
- Datalab API key — required when using the default `marker-api` cloud proxy

### Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

#### Core credentials

| Variable | Description |
|----------|-------------|
| `NOTION_TOKEN` | Notion integration token (`secret_…`) |
| `NOTION_PAPER_TRACKER_DB_ID` | 32-char hex database ID |
| `NOTION_KNOWLEDGE_INBOX_DB_ID` | 32-char hex database ID |
| `NOTION_SECOND_BRAIN_DB_ID` | 32-char hex database ID |
| `NOTION_PROJECTS_DB_ID` | 32-char hex database ID |
| `NOTION_EDGES_DB_ID` | 32-char hex database ID for the Edges DB |
| `NOTION_DEFERRED_EDGES_DB_ID` | 32-char hex database ID for the Deferred Edges DB (optional — edges whose targets aren't promoted yet are queued here) |
| `ANTHROPIC_API_KEY` | Anthropic API key — used for Claude extraction and edge linking |
| `OPENAI_API_KEY` | OpenAI API key — used for embeddings only (`text-embedding-3-small`); not required when `VECTOR_EMBEDDING_BACKEND=local` |
| `DATALAB_API_KEY` | Datalab API key — required for the `marker-api` cloud PDF conversion service |
| `KOOFR_USER` | Koofr account email |
| `KOOFR_APP_PASSWORD` | Koofr WebDAV app password |
| `KOOFR_PDF_PATH` | Base path in Koofr where zips live (e.g. `/Papers`) |
| `ZOTERO_USER_ID` | Zotero user ID (numeric) — used to resolve attachment keys via Zotero API |
| `ZOTERO_API_KEY` | Zotero API key — used to fetch item children for PDF attachment resolution |

#### Model and pipeline settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Anthropic model ID used for extraction and edge linking |
| `MARKER_API_URL` | `http://marker-api:8080` | Internal URL of the Marker service (change to `http://marker-local:8080` when using local profiles) |
| `MARKER_MODE` | `balanced` | Conversion quality preset: `fast`, `balanced`, or `accurate` |
| `ARXIV_KEYWORDS` | — | Comma-separated keywords for ArXiv Sniper |
| `ARXIV_RELEVANCE_THRESHOLD` | `8` | Min relevance score (1–10) to auto-add ArXiv papers |
| `GRAPH_SERVER_PORT` | `8000` | Port for the dependency graph server |
| `TAGS_REGISTRY_PATH` | `../tags_registry.yaml` | Path to `tags_registry.yaml` |
| `PIPELINE_DB_PATH` | `/tmp/pipeline/ingestion_jobs.db` | Path to SQLite job ledger |
| `EXTRACTION_VERSION` | `v3` | Schema version string used for idempotency |
| `SB_CONCEPT_LEVEL` | `Concept` | `Note Level` select value used to query Second Brain Concept pages |
| `RETRIEVE_CANDIDATES_K` | `30` | Max Second Brain candidates fetched from Qdrant per concept before pre-filter scoring |

#### Vector index (Qdrant)

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_INDEX_ENABLED` | *(unset)* | Set to any non-empty value to enable semantic retrieval via Qdrant; falls back to TF-IDF when unset |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint (matches Docker Compose service name) |
| `VECTOR_EMBEDDING_BACKEND` | `openai` | Embedding backend: `openai` (requires `OPENAI_API_KEY`) or `local` (requires `sentence-transformers`) |
| `VECTOR_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model when `backend=openai`; or HuggingFace model ID when `backend=local` (e.g. `allenai/specter2`) |
| `VECTOR_EMBEDDING_DIM` | `1536` | Embedding dimension — must match the model: `1536` for `text-embedding-3-small`, `3072` for `text-embedding-3-large`, `768` for `allenai/specter2` |

#### Cross-paper edge pre-filter thresholds

These control the structural pre-filter that scores and prunes Qdrant candidates before they are sent to Claude for edge confirmation. All have sensible defaults and do not need to be changed for typical deployments.

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMED_TOOL_MATCH_THRESHOLD` | `85` | `rapidfuzz token_sort_ratio` threshold (0–100) for named-tool signal |
| `SETTING_CONTAINMENT_THRESHOLD` | `80` | `rapidfuzz partial_ratio` threshold (0–100) for setting-containment signal |
| `ASSUMPTION_OVERLAP_DROP_THRESHOLD` | `0.05` | Minimum assumption-conclusion Jaccard to avoid drop |
| `KEYWORD_JACCARD_DROP_THRESHOLD` | `0.10` | Minimum keyword Jaccard to avoid drop |
| `QDRANT_SIMILARITY_DROP_THRESHOLD` | `0.75` | Minimum Qdrant cosine similarity to avoid drop (only applies when all other signals are zero) |
| `WEIGHT_QDRANT` | `0.40` | Composite score weight for Qdrant similarity |
| `WEIGHT_NAMED_TOOL` | `0.25` | Composite score weight for named-tool signal |
| `WEIGHT_ASSUMPTION_OVERLAP` | `0.20` | Composite score weight for assumption-conclusion overlap |
| `WEIGHT_SETTING_CONTAINMENT` | `0.10` | Composite score weight for setting containment |
| `WEIGHT_KEYWORD_JACCARD` | `0.05` | Composite score weight for keyword Jaccard |
| `EDGE_AUTO_CREATE_CONFIDENCE` | `0.80` | Confidence threshold above which edges with a structural signal are auto-created (`needs_review=False`) |
| `EDGE_REVIEW_FLAG_CONFIDENCE` | `0.65` | Confidence threshold below which edges are NOT written to the Edges DB at all |
| `EDGE_MAX_CANDIDATES_TO_GPT` | `10` | Max candidates passed to Claude after pre-filter reranking |
| `NOTION_HYDRATION_CONCURRENCY` | `5` | Concurrent Notion page fetches when hydrating candidates |

### Required Notion Properties

#### Paper Tracker DB

| Property | Type | Notes |
|----------|------|-------|
| `Status` | **Status** (not Select) | Values: `s0-inbox`, `s1-skim`, `s1-processing`, `s1b-waiting-attachment`, `blocked-extraction`, `s2-extracted`, `s2-reextract`, `s2-read`, `s3-distilled` |
| `Zotero URI` | URL or Rich Text | e.g. `https://www.zotero.org/users/…/items/XXXXXXXX` — parent item key parsed from this |
| `Zotero Attachment Key` | Rich Text | Written by pipeline after resolving PDF attachment via Zotero API |
| `primary_pdf_filename` | Rich Text | Optional — filename to prefer when zip contains multiple PDFs |
| `PDF SHA256` | Rich Text | Written by pipeline after PDF download |
| `Re-extract Hints` | Rich Text | Newline-separated list of concept names/descriptions for targeted re-extraction (human fills this when setting `s2-reextract`) |
| `Extraction Error` | Rich Text | Cleared on success; set to error message on failure |
| `One Liner` | Rich Text | Written by pipeline after extraction |
| `Active Themes` | Multi-select | Written by pipeline after extraction |
| `Extraction Count` | Number | Number of concepts extracted |
| `Extraction Tokens` | Number | Token count of the Markdown fed to the LLM |
| `Extraction Version` | Rich Text | e.g. `v3` — schema version used for this extraction |
| `Processed At` | Date | UTC timestamp of last successful pipeline run |
| `Last Run ID` | Rich Text | 8-char hex run ID for log correlation |
| `Last Error` | Rich Text | Cleared on success; set to error message on failure |
| `Knowledge Items` | Relation → Knowledge Inbox | Optional back-link |

#### Knowledge Inbox DB

| Property | Type | Notes |
|----------|------|-------|
| `Name` | Title | Set to the concept title (without type prefix) |
| `Type` | Select | Values: `Definition`, `Theorem`, `Lemma`, `Algorithm`, `Assumption`, `Proof`, `ProofTechnique` |
| `Status` | Select or Status | Values: `Inbox`, `Reviewed`, `Promoted`, `Dropped` |
| `verification_status` | Select or Status | Values: `unverified`, `verified`, `rejected` |
| `Graph Link Status` | Select or Status | Values: `unlinked` (no edges produced), `linked-ai` (edges written by Stage 3), `promoted` (promoted to Second Brain) |
| `Source Paper` | Relation → Paper Tracker | Back-link to the source paper |
| `Source Pages` | Rich Text | Comma-separated page numbers |
| `Suggested Hub` | Rich Text | Knowledge hub suggested by the LLM |
| `AI Confidence` | Number | Extraction confidence 0–1 |
| `Keywords` | Multi-select | Primary keywords identifying this concept |
| `Prereq Keywords` | Multi-select | Keywords of prerequisite concepts |
| `Downstream Keywords` | Multi-select | Keywords of concepts this enables |
| `Source Anchors` | Rich Text | Section/equation refs, e.g. `Section 3.2; Eq. (12)` |
| `Interpretation` | Rich Text | Plain-English meaning |
| `Proof Idea` | Rich Text | Optional proof sketch |
| `Aliases` | Rich Text | Alternative names for this concept |
| `Named Tools` | Multi-select | Named theorems/lemmas/frameworks invoked, e.g. `Banach fixed-point` |
| `Setting` | Multi-select | Mathematical setting tags, e.g. `finite_state`, `continuous`, `graphon` |
| `Result Category` | Select | `existence`, `uniqueness`, `convergence`, `stability`, `approximation` |
| `Assumptions` | Rich Text | Boundary conditions / hypotheses |
| `Statement LaTeX` | Rich Text | Formal statement in LaTeX |
| `Conclusion` | Rich Text | The conclusion or result established |
| `Source Quote` | Rich Text | Verbatim quote ≤ 25 words |
| `Corrected Title` | Rich Text | Human-corrected title; PromotionEngine uses this instead of `Name` when non-empty |
| `Candidate Matches` | Rich Text | JSON list of Stage 2 candidates (written by pipeline; for debug) |
| `Edge Suggestions` | Rich Text | JSON edge proposals written by Stage 3; read by PromotionEngine |
| `Promotion Target` | Relation → Second Brain | Set by PromotionEngine when concept is promoted |

#### Edges DB (create manually, then set `NOTION_EDGES_DB_ID`)

| Property | Type | Notes |
|----------|------|-------|
| `Name` | Title | Auto: `"[relation_type]: {to_sb_id[:8]}"` |
| `From Concept` | Relation → Second Brain | Source concept |
| `To Concept` | Relation → Second Brain | Target concept |
| `Relation Type` | Select | `depends_on`, `enables`, `generalizes`, `special_case_of`, `related` |
| `Rationale` | Rich Text | Human note or AI-generated rationale |
| `AI Confidence` | Number | Claude confidence score [0, 1] |
| `Source Papers` | Relation → Paper Tracker | Papers that contributed this edge |
| `Created By` | Select | `AI-suggested`, `Human-verified` |
| `Status` | Select | `pending`, `resolved`, `stale` |
| `needs_review` | Checkbox | `True` = auto-created but flagged for human confirmation |
| `driving_fields` | Rich Text | Comma-separated list of concept fields that drove Claude's edge decision |
| `pre_filter_signal` | Select | Dominant pre-filter signal: `named_tool_match`, `assumption_conclusion_overlap`, `setting_containment`, `keyword_jaccard`, or `none` |
| `justification` | Rich Text | Claude's one-sentence justification referencing specific field content |

#### Deferred Edges DB (optional — create manually, set `NOTION_DEFERRED_EDGES_DB_ID`)

Edges whose target concept hasn't been promoted yet are queued here and resolved on the next promotion run.

| Property | Type | Notes |
|----------|------|-------|
| `Name` | Title | Auto: `"[relation_type] → {target_title}"` |
| `From Concept` | Relation → Second Brain | Source concept (already promoted) |
| `Target Title` | Rich Text | Title of the not-yet-promoted target |
| `Relation Type` | Select | Same as Edges DB |
| `Rationale` | Rich Text | |
| `AI Confidence` | Number | |
| `Source Papers` | Relation → Paper Tracker | |
| `Status` | Select | `pending`, `resolved`, `stale` |
| `Resolved To` | Relation → Second Brain | Set when the deferred edge is later resolved |
| `Resolved At` | Date | UTC timestamp of resolution |
| `Created At` | Date | UTC timestamp of deferral |

---

## Running

### Locally (development)

```bash
cd orchestrator
pip install -r requirements.txt
# Initialise the SQLite database
python db_init.py
# Start the scheduler (requires all env vars)
python main.py
```

### Via Docker Compose

The `docker-compose.yml` defines several services. The Marker PDF conversion service has
three modes selected via Docker Compose profiles:

```bash
# Cloud API (default) — forwards PDF conversion to Datalab; requires DATALAB_API_KEY
docker compose --profile marker-api up --build

# Local CPU — runs marker-pdf inside the container; no API key needed
docker compose --profile marker-local-cpu up --build

# Local GPU — requires NVIDIA runtime; fastest local option
docker compose --profile marker-local-gpu up --build

# View orchestrator logs
docker compose logs -f orchestrator

# Stop all services
docker compose down
```

The `qdrant` service is always started regardless of profile and stores its data in a
persistent `qdrant_storage` Docker volume.

The `marker-local` profiles download model weights on first run (~2–4 GB).
These are cached in the `marker_models` Docker volume and survive rebuilds.

---

## Status Pipeline

The pipeline is a state machine driven by the `Status` property of each Paper Tracker
page. The canonical reference is [`NOTION_DBS.md`](./NOTION_DBS.md).

```
s0-inbox         ← Notero syncs paper from Zotero (holding state)
    │
    ▼
s1-skim          ← Human sets this after skimming the abstract
    │
    ▼ pipeline claims paper (race-condition guard)
s1-processing
    │
    ├─► s1b-waiting-attachment   (Koofr zip not found — pipeline retries next tick)
    │
    ├─► blocked-extraction       (LLM returned 0 concepts — human adds hints)
    │
    └─► s2-extracted             (concepts extracted, KI pages created, Qdrant indexed)
            │
            ├─► s2-reextract     (human found missing concepts — targeted re-extraction)
            │        └─► s2-extracted  (pipeline returns here automatically)
            │
            ▼ human sets s2-read when satisfied
        s2-read
            │
            ▼ PromotionEngine
        s3-distilled             (promoted to Second Brain + Zotero notes synced)
```

| Status | Set by | Meaning |
|--------|--------|---------|
| `s0-inbox` | Notero plugin | Paper just arrived from Zotero |
| `s1-skim` | Human | Ready for mathematical extraction |
| `s1-processing` | Pipeline | Pipeline has claimed the paper (race guard) |
| `s1b-waiting-attachment` | Pipeline | Zip not on Koofr; will retry next tick |
| `blocked-extraction` | Pipeline | LLM returned 0 concepts; add `Re-extract Hints` and reset to `s1-skim` |
| `s2-extracted` | Pipeline | All concepts extracted and written to Knowledge Inbox |
| `s2-reextract` | Human | Targeted second-pass extraction requested |
| `s2-read` | Human | Human review complete; trigger promotion |
| `s3-distilled` | Pipeline | Promoted to Second Brain; Zotero notes synced |

---

## Human Verification & Promotion

After the pipeline sets a paper to `s2-extracted`, the human workflow is:

1. Open the **Knowledge Inbox** database and review each concept page for this paper.
   Each page has a 5-item review checklist prepended to the body.
2. For the title: if the AI-generated title is wrong, fill `Corrected Title` — the
   PromotionEngine will use this instead of `Name`.
3. For correct concepts: set `verification_status = verified`.
4. For concepts to discard: set `verification_status = rejected`.
5. Optionally trigger targeted re-extraction: fill `Re-extract Hints` on the Paper
   Tracker page and set `Status = s2-reextract`.
6. When satisfied, set `Status = s2-read` on the Paper Tracker page.
   The **PromotionEngine** polls for `s2-read` on every scheduler tick and handles the rest.

### What the Promotion Engine does

For each paper at `s2-read`:

1. **Pass 1 — Concept nodes:** For each verified KI concept, creates (or patches) a
   Second Brain `Concept` page, transferring `Assumptions`, `Statement LaTeX`,
   `Interpretation`, `Proof Idea`, `Named Tools`, `Aliases`, `Setting`, `Result
   Category`, `Keywords`, and `Sources`. Uses `Corrected Title` when set.
2. **Pass 2 — Edges:** Resolves each entry in the `Edge Suggestions` JSON property on
   the KI page. Targets are matched by title against the Second Brain. Unresolvable
   targets (concept not yet promoted) are written to the **Deferred Edges DB** and
   resolved on the next promotion run.
3. **Qdrant migration:** Migrates Qdrant points from KI page ID → SB page ID and sets
   `verified=True`.
4. **Zotero note sync:** Fetches notes and annotations via the Zotero API and renders
   them as Notion blocks on the Paper Tracker page (idempotent).
5. **Status advance:** Sets `Status = s3-distilled`.

### Edges DB and the three-tier review workflow

The KI page body includes a `## Proposed Cross-Paper Edges` section with three
subsections generated by Stage 3:

| Tier | Icon | Condition | Edge in DB? | Action required |
|------|------|-----------|-------------|-----------------|
| Auto-created | ✅ | `confidence ≥ 0.80` AND structural signal | Yes, `needs_review=False` | None — informational only |
| Flagged | ⚠️ | `0.65 ≤ confidence < 0.80` OR no structural signal | Yes, `needs_review=True` | Review edge page in Edges DB: uncheck `needs_review` to accept, or delete the page to reject |
| Low-confidence hints | 💡 | `confidence < 0.65` | **No** | Create manually in Edges DB if desired |

The checkboxes in the ⚠️ section are **visual prompts only** — checking them in Notion
does not automatically update the Edges DB.

---

## 3-Stage Concept-Graph Pipeline

As of **schema v3**, ingestion runs three distinct stages per paper.

```
Paper Markdown
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 1 — EXTRACT  (LLM: Claude)          │
│  Input : full paper Markdown                │
│  Output: list of MathObject concepts,       │
│           each with title, statement_latex, │
│           assumptions, conclusion,          │
│           named_tools, setting,             │
│           canonical_keywords (5–15),        │
│           prereq_keywords (5–15),           │
│           downstream_keywords (5–15),       │
│           suggested_hub, confidence         │
│  Prompt: EXTRACTION_SYSTEM_PROMPT           │
│  LLM calls: 1 (+ 1 repair if needed)       │
└──────────────────┬──────────────────────────┘
                   │  K concepts
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 2 — RETRIEVE  (Qdrant / TF-IDF)     │
│  Input : one concept + Second Brain index   │
│  Output: top-K candidate concepts           │
│  Primary: Qdrant ANN search across 3        │
│           collections (concept_full,        │
│           concept_assumptions,              │
│           concept_conclusions)              │
│           + cross-paper pre-filter scoring  │
│           (4 structural signals, composite  │
│           score, drop gate)                 │
│  Fallback: TF-IDF token overlap + hub bonus │
│  LLM calls: 0                               │
└──────────────────┬──────────────────────────┘
                   │  ≤10 reranked candidates / concept
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 3 — LINK  (LLM: Claude)             │
│  Input : one concept + its candidates       │
│           (with full field content when     │
│           Qdrant path is active)            │
│  Output: CrossPaperLinkResult (proposals    │
│           with confidence tiers, driving    │
│           fields, review flags)             │
│  Prompt: LINKING_SYSTEM_PROMPT_V2           │
│           (full fields; falls back to V1    │
│           on TF-IDF path)                   │
│  LLM calls: 1 (+ 1 repair if needed)       │
└─────────────────────────────────────────────┘
```

### Design constraints

- **No blind LLM context:** Stage 3 only ever receives the top-K *retrieved* candidates —
  never the entire Second Brain.
- **Full content in prompt:** When Qdrant is active, Claude receives full field content
  (`assumptions`, `conclusion`, `statement_latex`, `setting`, `named_tools`) for both
  the source concept and every candidate. The prompt requires `driving_fields` to be
  populated, serving as a hallucination detector.
- **Intra-paper guard:** Concepts extracted from the same paper bypass the cross-paper
  pre-filter and use the legacy linking path.
- **`graph_link_status`** tracks where each concept sits: `unlinked` → `linked-ai`
  → `promoted` (after PromotionEngine runs).

### Tuning Stage 2 retrieval depth

Control how many candidates are retrieved for each concept via the env var:

```bash
RETRIEVE_CANDIDATES_K=20   # fewer, faster, higher precision
RETRIEVE_CANDIDATES_K=50   # more recall, larger Stage 3 context
```

Default is **30**. Values in the 20–50 range work well for a Second Brain
with up to ~5 000 atomic concepts. After pre-filter scoring and reranking,
at most `EDGE_MAX_CANDIDATES_TO_GPT` (default 10) candidates reach Claude.

### Edge caps

Stage 3 enforces hard limits on the number of edges per type
(defined in `extraction_schema.EDGE_CAPS`):

| Edge type | Max edges |
|-----------|-----------|
| `depends_on` | 3 |
| `enables` | 3 |
| `generalizes` | 2 |
| `special_case_of` | 2 |
| `related` | 5 |

### JobLedger checkpoints (v3)

| Checkpoint | Stage completed |
|------------|-----------------|
| `marker_done` | PDF → Markdown conversion |
| `extract_done` | Stage 1 LLM extraction |
| `retrieve_done` | Stage 2 candidate retrieval |
| `link_done` | Stage 3 LLM linking |
| `notion_done` | Final Notion writes complete |

Each checkpoint is written immediately, so a crash anywhere in the pipeline
results in a clean restart from the last completed checkpoint.

---

## Vector Index (Module 7)

The `VectorIndexEngine` maintains three Qdrant collections with role-specific embeddings:

| Collection | Embeds | Detects |
|-----------|--------|---------|
| `concept_full` | title + conclusion + interpretation + canonical_keywords | `related`, `generalizes`, `special_case_of` |
| `concept_assumptions` | assumptions + prereq_keywords + named_tools | `depends_on` |
| `concept_conclusions` | conclusion + statement_latex (stripped) + downstream_keywords | `enables` |

New KI concepts are indexed immediately after creation (`verified=False`). When a
concept is promoted to the Second Brain, the Qdrant point is migrated from the KI page
ID to the SB page ID and `verified=True` is set.

### Rebuild the index from scratch

```bash
python -c "
from modules.vector_index import VectorIndexEngine
VectorIndexEngine().rebuild()
"
```

This drops and recreates all three collections and re-indexes all Second Brain and
Knowledge Inbox concepts. Use after changing the embedding model or schema.

### Graceful degradation

If Qdrant is unreachable at startup, `VectorIndexEngine._available` is `False` and
every method is a silent no-op. The pipeline falls back to TF-IDF retrieval and the
legacy edge-linking prompt automatically.

---

## Cross-Paper Edge Quality

When Qdrant is active, each Qdrant candidate is scored by four structural signals
before being sent to Claude. This reduces hallucinated edges by grounding the LLM call
in actual mathematical content rather than just titles and summaries.

### Pre-filter signals

| Signal | Method | Weight in composite score |
|--------|--------|--------------------------|
| **Named tool match** | `rapidfuzz.token_sort_ratio` between `C_A.named_tools` elements and `C_B.title` (and vice versa); threshold 85 | 0.25 |
| **Assumption-conclusion overlap** | Token Jaccard between `C_A.assumptions` and `C_B.conclusion` (and vice versa); LaTeX commands stripped | 0.20 |
| **Setting containment** | `rapidfuzz.partial_ratio` between `C_A.setting` and `C_B.setting`; threshold 80 | 0.10 |
| **Keyword Jaccard** | Exact-match Jaccard on normalised `keywords` sets | 0.05 |
| **Qdrant similarity** | Raw cosine similarity from ANN search | 0.40 |

**Drop condition:** a candidate is excluded from the Claude call entirely when all
four structural signals are below threshold AND `qdrant_similarity < 0.75`. Named
tool match always prevents dropping regardless of other scores.

**Reranking:** after scoring, candidates are sorted by composite score descending and
capped at `EDGE_MAX_CANDIDATES_TO_GPT` (default 10) before the LLM call.

### Confidence tiers

| Tier | Condition | Written to Edges DB? | `needs_review` |
|------|-----------|----------------------|----------------|
| Auto-create | `confidence ≥ 0.80` AND ≥1 structural signal | ✅ Yes | `False` |
| Flagged | `0.65 ≤ confidence < 0.80`, OR high confidence but no structural signal | ✅ Yes | `True` |
| Low-confidence hint | `confidence < 0.65` | ❌ No (rendered on KI page only) | — |

---

## Utility Scripts

Several one-off scripts are provided at the repository root:

### `run_promotion.py`

Run the PromotionEngine manually without Docker:

```bash
python run_promotion.py
```

Polls Paper Tracker for `s2-read` papers and promotes all verified KI concepts.

### `run_deferred_edges.py`

Resolve all pending rows in the Deferred Edges DB:

```bash
python run_deferred_edges.py          # live run
python run_deferred_edges.py --dry-run  # preview only
```

Use after a promotion run that left edges unresolved because their target concepts
were promoted later.

### `run_backfill_ki_props.py`

Backfill KI DB properties from page body content:

```bash
python run_backfill_ki_props.py [--dry-run] [--page-id <id> [<id> ...]]
```

Reads `## Assumptions`, `## Statement`, `## Conclusion`, `## Interpretation`, and
`## Proof Idea` sections from the KI page body and writes them to the corresponding
DB properties. Skips properties that already have content (idempotent).

---

## Tag Registry

Tags categorise papers into domains (`d-`), methods (`m-`), and knowledge types (`k-`).

### Format

```yaml
tags:
  - id: d-mfg              # canonical id: prefix-slug
    prefix: d              # d=domain, m=method, k=knowledge-type
    name: Mean Field Games
    definition: >
      The study of strategic decision-making in large populations...
    inclusion_criteria: >
      Papers that explicitly formulate or analyse a MFG equilibrium...
    exclusion_criteria: >
      Papers on single-agent optimal control without population interaction...
    synonyms:
      - MFG
      - mean-field games
    deprecated: false      # optional, default false
```

### Adding a Tag

1. Open `tags_registry.yaml`.
2. Choose the correct prefix: `d-` (domain), `m-` (method), `k-` (knowledge type).
3. Add your entry following the format above.
4. Verify the file is valid:
   ```bash
   python -c "import sys; sys.path.insert(0, 'orchestrator'); from modules.tag_linter import TagRegistry; r = TagRegistry(); print(f'Loaded {len(r)} tags successfully.')"
   ```
5. Set `TAGS_REGISTRY_PATH` in `.env` if using a non-default location.

### Tag Linter

The linter runs automatically during ingestion. It:
- Validates format (`d-/m-/k-` prefix, `[a-z0-9-]` characters).
- Checks registry membership.
- Auto-corrects known synonyms to canonical ids.
- Flags deprecated tags.

If all tags fail validation, the paper is blocked (`blocked-tags`) and
the full lint report is stored in the `tag_lint_report` property.

---

## Idempotency and Job Ledger

Every ingestion run is tracked in a SQLite database at `PIPELINE_DB_PATH`.

The ledger prevents re-processing a paper if:
- The pipeline crashes mid-run and is restarted.
- The same paper is accidentally set back to `s1-skim`.
- The same PDF (same SHA256) is submitted twice.

The idempotency key is the triple `(zotero_key, pdf_sha256, extraction_version)`.

To inspect the ledger:
```bash
sqlite3 /tmp/pipeline/ingestion_jobs.db \
  "SELECT zotero_key, status, started_at, finished_at FROM ingestion_jobs ORDER BY id DESC LIMIT 20;"
```

To reset a stuck job (e.g. after fixing a bug):
```bash
sqlite3 /tmp/pipeline/ingestion_jobs.db \
  "UPDATE ingestion_jobs SET status='failed' WHERE zotero_key='XXXXXXXX' AND status='started';"
```

---

## Troubleshooting

### Missing zip (`s1b-waiting-attachment`)

The pipeline looks for `{attachment_key}.zip` at `KOOFR_PDF_PATH/{attachment_key}.zip`.
The **attachment key** (PDF child item in Zotero) is resolved from the parent key in
`Zotero URI` via the Zotero API.

1. Check that Zotero exported the attachment (`ZOTERO_USER_ID` / `ZOTERO_API_KEY` correct).
2. Check the Koofr sync is up to date and the zip has been uploaded.
3. Verify `Zotero URI` in Notion is a valid URL containing the parent key (8 uppercase alphanumerics).
4. Once the attachment appears on Koofr, reset the paper status to `s1-skim`.

The resolved attachment key is written to `Zotero Attachment Key` in Notion for inspection.

### Multiple PDFs in zip

By default the pipeline selects the **largest** PDF in the zip.
To override, set the `primary_pdf_filename` property in Notion to the exact
filename (e.g. `paper.pdf`) before triggering the pipeline.

### Notion rate limit

The `NotionClientWrapper` enforces a 3 req/s token-bucket limiter and retries
on HTTP 429 with exponential backoff (up to 6 attempts, max 60 s wait).
If you see persistent 429 errors, reduce parallelism or increase the wait.

### Marker API failure

The Marker OCR call is retried up to 4 times with exponential backoff.
Common causes:
- `DATALAB_API_KEY` is missing or invalid (cloud mode).
- The `marker-local` container has not finished downloading model weights (local mode).
- The PDF is corrupted or password-protected.
- The `/tmp/pipeline` shared volume is not mounted correctly.

Check logs: `docker compose logs -f marker-api` (or `marker-local`).

### LLM schema validation failure

The pipeline attempts one automatic repair call if the first extraction fails
Pydantic validation. If the repair also fails, all concepts are flagged with
`confidence=0.0` and the pipeline continues (degraded mode).

Check the `Extraction Error` property on the Paper Tracker page.

### Qdrant unreachable

If `VECTOR_INDEX_ENABLED` is set but Qdrant cannot be reached at startup, the
`VectorIndexEngine` sets `_available=False` and the pipeline falls back silently
to TF-IDF retrieval. The legacy edge-linking prompt (V1) is used instead of the
enriched V2 prompt.

Check logs for: `VectorIndexEngine: Qdrant unreachable`.

---

## Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`.
- The Notion integration token should use **least-privilege** scopes:
  only the databases listed above need to be shared with the integration.
- The pipeline **never writes to the Second Brain DB directly** during extraction — it
  only reads Hub page names/IDs for prompt injection during Stage 1. All writes to the
  Second Brain happen through the PromotionEngine after human review.
- Koofr app passwords are separate from your Koofr account password and can
  be revoked independently.
- Anthropic and OpenAI API keys should be scoped to a project with spend limits set.
- The SQLite job ledger contains only hashes and status strings — no PII or
  paper content is persisted locally.


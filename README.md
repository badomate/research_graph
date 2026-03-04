# paper_pipeline

A Python orchestration pipeline that ingests academic papers from Zotero/Notero into
Notion, extracts mathematical concepts via Marker OCR and OpenAI GPT-4o, and populates
a structured Knowledge Inbox for review.

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
│                                 │ poll s1-process-math     │ write          │
│                      ┌──────────▼──────────────────────────▼─────────────┐ │
│                      │          Python Orchestrator (APScheduler)        │ │
│                      │                                                   │ │
│                      │  IngestionEngine  ArXivSniper  ConflictDetector  │ │
│                      │  LaTeXCompiler    DependencyGrapher               │ │
│                      └────┬──────────────────────────┬───────────────────┘ │
│                           │ POST /marker              │ chat.completions    │
│                      ┌────▼──────────┐          ┌────▼──────────┐         │
│                      │  marker-api   │          │  OpenAI GPT-4o│         │
│                      │  (local OCR)  │          │  (extraction) │         │
│                      └───────────────┘          └───────────────┘         │
│                                                                             │
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
| OCR / PDF→Markdown | marker-pdf (local Docker container) |
| Concept extraction | OpenAI GPT-4o (JSON mode) |
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
- Koofr account with WebDAV app password
- OpenAI API key with GPT-4o access

### Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `NOTION_TOKEN` | Notion integration token (`secret_…`) |
| `NOTION_PAPER_TRACKER_DB_ID` | 32-char hex database ID |
| `NOTION_KNOWLEDGE_INBOX_DB_ID` | 32-char hex database ID |
| `NOTION_SECOND_BRAIN_DB_ID` | 32-char hex database ID |
| `NOTION_PROJECTS_DB_ID` | 32-char hex database ID |
| `OPENAI_API_KEY` | OpenAI API key (`sk-…`) |
| `KOOFR_USER` | Koofr account email |
| `KOOFR_APP_PASSWORD` | Koofr WebDAV app password |
| `KOOFR_PDF_PATH` | Base path in Koofr where zips live (e.g. `/zotero`) |
| `ARXIV_KEYWORDS` | Comma-separated keywords for ArXiv Sniper |
| `ARXIV_RELEVANCE_THRESHOLD` | Min relevance score (1–10) to auto-add ArXiv papers |
| `MARKER_API_URL` | Internal URL of the marker-api container |
| `GRAPH_SERVER_PORT` | Port for the dependency graph server (default `8000`) |
| `TAGS_REGISTRY_PATH` | Path to `tags_registry.yaml` (default `../tags_registry.yaml`) |
| `PIPELINE_DB_PATH` | Path to SQLite job ledger (default `/tmp/pipeline/ingestion_jobs.db`) |
| `EXTRACTION_VERSION` | Schema version string used for idempotency (default `v2`) |

### Required Notion Properties

#### Paper Tracker DB

| Property | Type | Notes |
|----------|------|-------|
| `Status` | Select | Values: `s0-inbox`, `s1-process-math`, `s1b-waiting-attachment`, `blocked-tags`, `s2-extracted`, `s3-verified` |
| `Zotero URI` | Rich Text | e.g. `zotero://select/items/XXXXXXXX` |
| `Tags` | Multi-select | Must match entries in `tags_registry.yaml` |
| `primary_pdf_filename` | Rich Text | Optional — filename to prefer when zip contains multiple PDFs |
| `PDF SHA256` | Rich Text | Written by pipeline after PDF download |
| `tag_lint_report` | Rich Text | Written by pipeline when tags have issues |
| `One Liner` | Rich Text | Written by pipeline after extraction |
| `Active Themes` | Multi-select | Written by pipeline after extraction |
| `AI Status` | Select | Written by pipeline; value: `Unverified-AI` |

#### Knowledge Inbox DB

| Property | Type | Notes |
|----------|------|-------|
| `Name` | Title | Set to `[Type] Title` |
| `Type` | Select | Values: `Definition`, `Theorem`, `Lemma`, `Algorithm`, `Assumption`, `Proof` |
| `Status` | Select | Initial value: `Inbox` |
| `verification_status` | Select | Values: `unverified`, `verified`, `rejected` |
| `Source Paper` | Relation → Paper Tracker | Back-link to the source paper |
| `Source Pages` | Rich Text | Comma-separated page numbers |
| `Hub Suggestions` | Rich Text | JSON string `{"suggested_hub": "…"}` — text only, no live relation |

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

```bash
# Build and start all services
docker-compose up --build

# View orchestrator logs
docker-compose logs -f orchestrator

# Stop all services
docker-compose down
```

The `marker-api` container downloads model weights on first run (~2–4 GB).
These are cached in the `marker_models` Docker volume and survive rebuilds.

---

## Status Pipeline

```
s0-inbox (legacy)
    │
    ▼
s1-process-math   ◄── Trigger: set this manually or via Notero webhook
    │
    ├─► s1b-waiting-attachment   (zip not yet on Koofr — retried next cycle)
    │
    ├─► blocked-tags             (no valid tags — human must fix tags)
    │
    └─► s2-extracted             (pipeline completed successfully)
            │
            ▼
        s3-verified              (human has reviewed the extracted concepts)
```

| Status | Set by | Meaning |
|--------|--------|---------|
| `s0-inbox` | Notero plugin | Paper just arrived from Zotero (legacy trigger) |
| `s1-process-math` | Human / Notero | Ready for mathematical extraction |
| `s1b-waiting-attachment` | Pipeline | Zip file not found on Koofr; will retry |
| `blocked-tags` | Pipeline | No valid tags; human must add tags and reset to `s1-process-math` |
| `s2-extracted` | Pipeline | All concepts extracted and written to Knowledge Inbox |
| `s3-verified` | Human | Extracted concepts have been reviewed |

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
- The same paper is accidentally set back to `s1-process-math`.
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

The pipeline looks for `{ZoteroKey}.zip` at `KOOFR_PDF_PATH/{ZoteroKey}.zip`.

1. Check that Zotero exported the attachment and the Koofr sync is up to date.
2. Verify `Zotero URI` in Notion contains a valid 8-char alphanumeric key.
3. Once the zip appears on Koofr, reset the paper status to `s1-process-math`.

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
- The `marker-api` container has not finished downloading model weights.
- The PDF is corrupted or password-protected.
- The `/tmp/pipeline` shared volume is not mounted correctly.

Check logs: `docker-compose logs -f marker-api`

### OpenAI schema validation failure

The pipeline attempts one automatic repair call if the first extraction fails
Pydantic validation. If the repair also fails, all concepts are flagged with
`confidence=0.0` and the pipeline continues (degraded mode).

Check the `AI Status` property in Notion; degraded extractions will still
appear in the Knowledge Inbox but with zero confidence scores.

---

## Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`.
- The Notion integration token should use **least-privilege** scopes:
  only the databases listed above need to be shared with the integration.
- The pipeline **never writes to the Second Brain DB directly** — it only
  reads Hub page names/IDs for prompt injection. Hub suggestions are stored
  as plain text in the Knowledge Inbox and require human promotion.
- Koofr app passwords are separate from your Koofr account password and can
  be revoked independently.
- OpenAI API keys should be scoped to a project with spend limits set.
- The SQLite job ledger contains only hashes and status strings — no PII or
  paper content is persisted locally.

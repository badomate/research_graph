# Second Brain Ingestion Pipeline — Flow Reference

> **Actors:**
> - 🤖 **PIPELINE** — fully automated, no human action required
> - 🧑 **HUMAN** — you must take an explicit action in Notion before the pipeline continues

---

## Status Flow at a Glance

| Status | Set by | Meaning |
|---|---|---|
| `s0-inbox` | Notero | Paper synced from Zotero — not yet picked up |
| `s1-skim` | Human | You have skimmed the paper and want it processed |
| `s1-processing` | Pipeline | Pipeline claimed the paper — extraction in progress |
| `s1b-waiting-attachment` | Pipeline | Koofr zip not found — waiting for attachment sync |
| `blocked-extraction` | Pipeline | GPT returned 0 concepts — needs human hint |
| `s2-extracted` | Pipeline | Extraction complete — concepts are in Knowledge Inbox |
| `s2-reextract` | Human | You found missing concepts — add hints and set this |
| `s2-read` | Human | You finished reviewing KI concepts — trigger promotion |
| `s3-distilled` | Pipeline | Promoted to Second Brain + Zotero notes synced |

---

## Stage 0 — Paper Arrives

### Step 0 🤖 `s0-inbox` — Notero syncs paper from Zotero

- You add a paper in Zotero. Notero automatically creates a page in the Paper Tracker DB.
- The page contains: title, authors, Zotero URI, abstract (if available), and attachment key.
- No pipeline polling happens yet — this is a holding state.
- Nothing is downloaded. No Koofr check. No OpenAI call.

> 🧑 **Human action required:** Open the paper, skim the abstract. If worth processing, set `Status → s1-skim`.

---

## Stage 1 — Extraction

### Step 1 🧑 `s1-skim` → `s1-processing` — Trigger extraction

- Set `Status = s1-skim` in the Paper Tracker. The pipeline polls for this on every tick.
- Optionally fill `primary_pdf_filename` if the Koofr zip contains multiple PDFs.
- The pipeline immediately sets `s1-processing` as its very first operation — this is a race-condition guard that prevents two scheduler instances from picking up the same paper simultaneously.

### Step 2 🤖 `s1-processing` — Resolve Zotero attachment + check Koofr

- Resolve parent item key and attachment key from the Zotero URI.
- Check Koofr WebDAV: does `/zotero/{attachment_key}.zip` exist?
- If zip **not found**: set `s1b-waiting-attachment` and stop. This happens when Zotero has not yet synced the file to Koofr.
- If zip **found**: proceed to PDF extraction.

### Step 3 🤖 `s1-processing` — PDF → Markdown (cached)

- Check Koofr markdown cache: `/zotero_markdown/{attachment_key}.md`
- **Cache hit:** download the `.md` file directly — no marker call needed.
- **Cache miss:**
  1. Download zip from Koofr
  2. Extract PDF
  3. Compute SHA256, write to Notion
  4. POST to marker-api for conversion
  5. Strip boilerplate (References, Appendix, Acknowledgements, Supplementary, Deferred Proofs, Bibliography)
  6. Upload clean markdown to Koofr cache for future re-extraction runs
- Token count logged. Papers >30k tokens use section-aware chunked extraction.

### Step 4 🤖 `s1-processing` — GPT concept extraction

- **≤30k tokens:** single-shot extraction call.
- **>30k tokens:** chunked extraction — abstract+intro used as shared preamble, sections processed individually, results merged and deduplicated by normalised title.
- **>60k tokens:** proof and appendix sections skipped entirely during chunking.
- Output: list of `MathObject` structs (title, type, `statement_latex`, assumptions, conclusion, interpretation, keywords, setting, named tools, etc.).
- **Zero-concept guard:** if GPT returns 0 concepts → set `blocked-extraction`, write error to `Extraction Error` property. Human must add `Re-extract Hints` and revert to `s1-skim`.

### Step 5 🤖 `s1-processing` — Candidate retrieval (vector + TF-IDF)

- For each extracted concept, retrieve the top-k most similar existing Second Brain concepts.
- **Vector path:** query all three Qdrant collections (`concept_full`, `concept_assumptions`, `concept_conclusions`) with ANN search. Results merged by `notion_page_id`, `edge_type_hint` attached per collection.
- **TF-IDF fallback:** if Qdrant is unavailable, keyword overlap scoring used instead.
- Candidates written to `Edge Suggestions` property on each KI page as JSON.

### Step 6 🤖 `s1-processing` → `s2-extracted` — Edge linking + KI creation

- For each concept, call GPT with the concept + candidate list to confirm/reject proposed edges and assign relation types (`depends_on`, `enables`, `generalizes`, `special_case_of`, `related`).
- Create a Knowledge Inbox page per concept: title, all extracted fields, 5-item review checklist, edge suggestions, `Source Paper` relation.
- Index each concept in Qdrant (`verified=False`).
- Inject `## Extracted Concepts` callout into the Paper Tracker page (idempotent).
- Write `Extraction Count` and `Extraction Tokens` to the paper page.
- Advance paper to `s2-extracted`.

---

## Stage 2 — Human Review (Knowledge Inbox)

The paper is now at `s2-extracted`. All extracted concepts are in the Knowledge Inbox DB as individual pages. No pipeline polling happens during this stage — it is entirely async human work.

### Step 7 🧑 `s2-extracted` — Review each Knowledge Inbox concept

- Open each KI page for the paper. A 5-item review checklist is prepended to the page body.
- Is the title correct? If not, fill `Corrected Title` — promotion will use this instead of `Name`.
- Is the `statement_latex` well-formed? Edit inline if needed.
- Are the proposed edges reasonable? They appear as `Edge Suggestions` on the page.
- Set `verification_status = verified` (accept) or `rejected` (discard) on each concept.
- Optionally add `Reviewer Notes`.

### Step 8 🧑 `s2-extracted` → `s2-reextract` — (Optional) Flag missing concepts

- If you notice important concepts that GPT missed, fill `Re-extract Hints` on the Paper Tracker page as a newline-separated list of concept names or brief descriptions.
- Set `Status = s2-reextract`. Pipeline picks this up on the next tick.
- Pipeline runs a targeted second-pass extraction using your hints + existing KI titles as a dedup guard. The cached markdown is used — no PDF re-download.
- New KI pages created for missed concepts. Paper returns to `s2-extracted` automatically.
- This can be done multiple times until satisfied.

### Step 9 🧑 `s2-extracted` → `s2-read` — Trigger promotion

- When you have reviewed all (or enough) concepts, set `Status = s2-read`.
- Pipeline picks this up on the next tick and begins the promotion run.
- You do not need to review 100% of concepts. Unverified ones are logged but promotion proceeds regardless.

---

## Stage 3 — Promotion to Second Brain

### Step 10 🤖 `s2-read` → `s3-distilled` — Promote verified concepts

- PromotionEngine polls Paper Tracker for `s2-read` papers.
- Fetches all KI pages linked to the paper via `Source Paper` relation.
- Logs stats: total / verified / rejected / unverified counts.
- For each verified concept: creates a Second Brain page (or updates if title already exists), using `Corrected Title` if set.
- **Two-pass edge creation:** pass 1 creates all SB concept pages and builds a `title → page_id` cache; pass 2 creates Edges DB entries — this ensures both endpoints exist before any edge is written.
- Migrates Qdrant points from KI page ID to SB page ID, sets `verified=True`.
- Marks promoted KI pages with a `Promoted` flag. Rejected KI pages are left as-is.

### Step 11 🤖 `s2-read` → `s3-distilled` — Sync Zotero notes + advance status

- Fetches Zotero notes and annotations via the Zotero API for the paper's item key.
- Renders notes as Notion blocks (paragraphs, headings, lists) and annotations as quote blocks with yellow background.
- **Idempotent:** skips papers that already contain a `[zotero:{key}]` callout marker.
- Note sync failure does **not** block promotion — paper advances to `s3-distilled` regardless.
- Sets `Status = s3-distilled`. Paper is fully processed. ✓

---

## Error & Recovery Flows

### `blocked-extraction` — GPT returned 0 concepts

Pipeline sets this when extraction technically succeeds but no concepts are returned. `Extraction Error` property contains the reason.

**Recovery:** Fill `Re-extract Hints` with concept names or descriptions. Set `Status = s1-skim`. Pipeline re-runs full extraction with hints injected into the system prompt.

### `s1b-waiting-attachment` — Koofr zip not found

Zotero has not yet synced the attachment to Koofr, or the attachment key in the Notion page is wrong.

**Recovery:** Wait for Zotero sync to complete, then manually revert `Status = s1-skim`. Or verify the Zotero URI on the page is correct.

### `s1-skim` revert — unexpected exception

Any unhandled exception during extraction reverts the paper to `s1-skim` and writes the error to `Extraction Error`. This prevents papers from getting stuck at `s1-processing` forever.

**Recovery:** Read `Extraction Error`. Fix the underlying issue (bad URI, marker timeout, etc.) and set `Status = s1-skim` to retry.

### Partial promotion — some concepts unverified

If you set `s2-read` before reviewing all concepts, only verified ones are promoted. The pipeline logs a warning with the unverified count. Unverified concepts remain in the KI.

**Recovery:** Go back and verify remaining KI concepts manually, then trigger another promotion run. The two-pass SB title cache handles duplicates gracefully.

---

## Complete Flow (Text Diagram)

```
Zotero → [Notero] → s0-inbox

s0-inbox → [HUMAN: skim paper, set s1-skim]

s1-skim → [Pipeline: race guard] → s1-processing

s1-processing → [Koofr zip check]
  ✗ zip missing → s1b-waiting-attachment → [HUMAN: wait + retry]
  ✓ zip found   → [markdown cache check]
      ✓ cached     → load .md from Koofr
      ✗ not cached → download zip → extract PDF → marker → strip boilerplate → upload .md

→ [GPT extraction]
  ✗ 0 concepts → blocked-extraction → [HUMAN: add hints + revert s1-skim]
  ✓ concepts   → [Stage 2: Qdrant retrieval] → [Stage 3: GPT edge linking]
              → create KI pages + index in Qdrant → s2-extracted

s2-extracted → [HUMAN: review KI pages]
  (optional)  → set s2-reextract + add Re-extract Hints
  s2-reextract → [Pipeline: targeted re-extraction] → new KI pages → s2-extracted
              → [HUMAN: set s2-read when satisfied]

s2-read → [PromotionEngine]
  → promote verified concepts to Second Brain
  → two-pass edge creation in Edges DB
  → migrate Qdrant points KI → SB (verified=True)
  → sync Zotero notes (best-effort)
  → s3-distilled ✓
```
## Overview

This system builds a **verified mathematical concept graph** from papers with **no AI write access to the master graph**. The graph is constructed in stages:

1. **Intake + control** (Paper Tracker)
2. **AI quarantine extraction** (Knowledge Inbox)
3. **Deterministic candidate retrieval** (Second Brain index)
4. **AI link suggestion** (Knowledge Inbox)
5. **Human promotion** into the master graph (Second Brain + optional Edges DB)

The core philosophy is:

* **AI can extract and propose.**
* **Humans verify and promote.**
* **Everything has provenance and traceability.**
* **Graph edges are not “co-occurrence”; they represent reuse-level relationships (depends_on, enables, generalizes, special_case_of, related).**

---

## Database architecture

You have 4 core databases in Notion (plus an optional fifth).

### A) Paper Tracker (control plane)

**Purpose**
One row per paper. It is the orchestration surface: status, keys, tags, processing metadata, and a small amount of paper-level AI output.

**AI writes here only small fields** (one-liner/themes/status/error metadata). No theorem text. No concept graph.

**Properties**

**Identity / provenance**

* **Name** (Title): paper title
* **Zotero URI** (URL): Notero-provided link (typically to parent item)
* **zotero_parent_key** (Text): 8-char key for the bibliographic item
* **zotero_attachment_key** (Text): 8-char key for the PDF attachment item
* **primary_pdf_filename** (Text, optional): disambiguation if multiple PDFs
* **PDF SHA256** (Text): hash of extracted PDF for idempotency

**Pipeline state**

* **Status** (Status/Select): the pipeline stage. Recommended values:

  * `s0-inbox`
  * `blocked-tags`
  * `s1-process-math`
  * `s1b-waiting-attachment`
  * `s2-extracted`
  * `s2b-linked-ai`
  * `s3-verified`
  * `s4-cited`
* **AI Status** (Select): `Unverified-AI | Needs-Fix | Verified`
* **Extraction Version** (Text): e.g. `v3`, `v4`
* **Processed At** (Date)
* **Last Run ID** (Text)
* **Last Error** (Text)
* **attachment_resolution_status** (Select): `ok | error | no_pdf | multiple`
* **attachment_resolution_log** (Text)

**Tagging**

* **Tags (mirror)** (Multi-select): d-/m-/k- tags. (If Zotero is the source of truth, treat this as mirrored.)
* **tag_lint_report** (Text): linter output. Gate on this.

**Paper-level AI outputs**

* **One Liner** (Text)
* **Active Themes** (Multi-select)

**Relations**

* **Knowledge Items** (Relation → Knowledge Inbox) (optional but useful)

---

### B) Knowledge Inbox (AI quarantine zone)

**Purpose**
This is where the AI deposits extracted **Concept Nodes** and later attaches **candidate matches** and **edge suggestions**. Everything here is assumed **unverified** by default.

This database is the staging area for building a concept graph without contaminating the master graph.

**Properties**

**Identity**

* **Name** (Title): recommended format `"[Type] Canonical Concept Name"`
* **Type** (Select): `Definition | Theorem | Lemma | Algorithm | Assumption | ProofTechnique`
* **Status** (Select): `Inbox | Reviewed | Promoted | Dropped`
* **verification_status** (Select): `unverified | verified | needs-fix`

**Provenance**

* **Source Paper** (Relation → Paper Tracker)
* **Source Pages** (Text): e.g. `"12, 13"`
* **Source Anchors** (Text): `"Section 3.2; Eq. (12)"` (optional)
* **Source Quote** (Text) (optional; ≤ 25 words)
* **Confidence** (Number 0–1)

**Hub**

* **Suggested Hub** (Select): must be one of ALLOWED_HUBS + `Uncategorized`
  (Better than storing hub suggestions as JSON; it makes filtering easy.)

**Concept content**

* **Statement LaTeX** (Text)
* **Assumptions** (Text)
* **Variables** (Text)
* **Interpretation** (Text)
* **Conclusion** (Text)
* **Proof Idea** (Text, optional)

**Search + retrieval helpers**

* **Keywords** (Multi-select)
* **Prereq Keywords** (Multi-select)
* **Downstream Keywords** (Multi-select)
* **Named Tools** (Multi-select)
* **Aliases** (Text, optional)

**Graph pipeline payload**

* **Candidate Matches (JSON)** (Text): the deterministic top-K candidates from Second Brain
* **Edge Suggestions (JSON)** (Text): LLM-generated edges (depends_on, enables, …)
* **Graph Link Status** (Select): `unlinked | linked-ai | needs-review | verified-links`

**Optional but high value**

* **Setting** (Multi-select): `finite_state | continuous | graphon | ergodic | common_noise | ...`
* **Result Category** (Select): `existence | uniqueness | convergence | ...`
* **Assumption Atoms (JSON)** (Text)
* **Equations (JSON)** (Text): structured HJB/FP blocks

**Promotion tracking**

* **Promotion Target** (Relation → Second Brain) (filled when promoted)

---

### C) Second Brain (master graph)

**Purpose**
This is the **clean, verified** knowledge graph: hubs + atomic concepts that you trust.

**AI does not write here directly.** Any update is either:

* human-driven, or
* a separate “promotion” action that requires verification.

**Properties**

**Identity**

* **Name** (Title)
* **Note Level** (Select): `Hub | Concept`
* **Type** (Select): same concept types (only for Concept notes)

**Core content**

* **Statement LaTeX** (Text)
* **Assumptions** (Text)
* **Variables** (Text)
* **Interpretation** (Text)
* **Proof Idea** (Text)
* **Keywords** (Multi-select)
* **Named Tools** (Multi-select)
* **Aliases** (Text)

**Hub membership**

* **Hub** (Relation → Second Brain filtered to Note Level=Hub)
  (Or a select; relation is better if you want hub pages to act as maps.)

**Provenance**

* **Sources** (Relation → Paper Tracker): papers that support/introduce this concept
* **Verified** (Checkbox)
* **Last Verified At** (Date)

**Graph structure**
You have two viable designs:

#### Option 1 (simple): relations on the Concept table

* **Depends On** (Relation → Second Brain)
* **Enables** (Relation → Second Brain)
* **Generalizes** (Relation → Second Brain)
* **Special Case Of** (Relation → Second Brain)
* **Related Concepts** (Relation → Second Brain)

This works up to a few hundred concepts but becomes painful to audit.

#### Option 2 (recommended): a separate Edges DB

This is better once you want provenance per edge, confidence per edge, and easy audits.

---

### D) Projects (output layer)

**Purpose**
Writing artifacts (thesis chapters, papers) and what they cite.

**Properties**

* **Name** (Title)
* **Type** (Select): `Thesis | Chapter | Paper | Proposal`
* **Status** (Select)
* **Referenced Papers** (Relation → Paper Tracker)
* **Referenced Concepts** (Relation → Second Brain)
* **Claims / Contributions** (Text)
* **Open Questions** (Text)

---

### E) Optional but strongly recommended: Concept Edges DB

**Purpose**
Represent the concept graph as an auditable edge list, not buried in relation fields.

**Properties**

* **Name** (Title): auto string `A —depends_on→ B`
* **From Concept** (Relation → Second Brain)
* **To Concept** (Relation → Second Brain)
* **Relation Type** (Select): `depends_on | enables | generalizes | special_case_of | related`
* **Rationale** (Text)
* **Confidence** (Number)
* **Source Papers** (Relation → Paper Tracker)
* **Created By** (Select): `AI-suggested | Human-verified`
* **Status** (Select): `suggested | verified | rejected`

This makes your graph traceable and lets you say: “This edge is supported by papers X, Y, Z.”

---

## End-to-end process flow

### Step 0: Intake (Zotero → Notion)

1. You add a paper to Zotero.
2. You apply objective tags in Zotero (d-/m-/k-) or leave empty (but then pipeline blocks later).
3. Notero syncs a record into Paper Tracker, including Zotero URI.

Key point: the Notero URI usually points to the **parent item** (`/items/ID23RPAJ`), not the attachment.

---

### Step 1: Orchestrator polling (Paper Tracker)

The ingestion service periodically queries Paper Tracker where:

* **Status = `s1-process-math`**

It processes papers one-by-one.

---

### Step 2: Preflight gates (fail fast, cheap)

The orchestrator runs a strict series of gates before spending compute/money:

#### Gate 1: Parse Zotero key

Extract parent key from Zotero URI.
If missing/unparseable → set error and stop.

#### Gate 2: Resolve attachment key (parent → child attachment)

Call Zotero Web API:

* `GET /users/{user_id}/items/{parent_key}/children`

Pick the PDF attachment key:

* filter attachments where `contentType == application/pdf` or filename ends with `.pdf`
* if multiple PDFs: use `primary_pdf_filename` or pick largest

Write keys back to Paper Tracker:

* `zotero_parent_key`
* `zotero_attachment_key`
* `attachment_resolution_status`

#### Gate 3: Acquire markdown source

You have two modes:

* **Marker mode** (PDF → Markdown via Marker API)
* **Manual mode** (you pasted `{attachment_key}.md`)

For production, use Marker. For testing, manual mode is fine.

#### Gate 4: SHA and idempotency

Compute `PDF SHA256` (if PDF is involved).
Use JobLedger to skip if already processed under current `EXTRACTION_VERSION`.

#### Gate 5: Tag completeness gate

Run TagLinter on Tags in Paper Tracker.
If invalid/missing → set `Status=blocked-tags` and stop.

Only after these gates do you proceed.

---

## The multi-stage AI pipeline: from markdown to concept graph

### Stage 1 (LLM): Extract high-value concepts from the paper

**Input**

* The paper markdown (possibly truncated)
* ALLOWED_HUBS list
* Strict extraction system prompt
* Output schema = `ExtractionResult` / `MathObject`

**What the LLM does**

* Identifies the **main reusable mathematical contributions**
* Extracts 3–12 **high-value concept nodes**
* Names each concept descriptively (never Theorem 1)
* Produces:

  * statement latex
  * assumptions + boundary conditions
  * variables
  * conclusion/interpretation
  * source pages/anchors
  * hub selection
  * confidence
* Additionally produces:

  * `canonical_keywords`
  * `prereq_keywords`
  * `downstream_keywords`
  * optionally: `named_tools`, `setting`, `result_category`, etc.

**What it must NOT do**

* It must NOT generate a dense internal dependency chain across the paper.
* It must NOT invent hubs.
* It must NOT paraphrase math when exact form is present.
* It must NOT emit low-value micro-lemmas that only serve the main theorem.

**Output handling**

* Validate using Pydantic (`validate_extraction`)
* If schema fails, attempt one repair call.
* Run `latex_sanity_check`; lower confidence if broken.

**Writes to Notion**

* Paper Tracker:

  * Status → `s2-extracted` (or keep it at `s1` until linking is done; your choice)
  * AI Status → `Unverified-AI`
  * One Liner
  * Active Themes
* Knowledge Inbox:

  * Create one page per extracted concept with content blocks and metadata
  * Set verification_status = `unverified`
  * Set Graph Link Status = `unlinked`

At the end of Stage 1 you have **isolated concept nodes** in the quarantine DB.

---

### Stage 2 (Deterministic): Retrieve candidate global concepts

This is the key move that prevents garbage linking.

**Goal**
For each extracted concept, retrieve a **small set of plausible matches** from the existing Second Brain concept library.

**Why this stage exists**
You cannot send “every concept ever” to the LLM for linking. It won’t scale and it will hallucinate edges.

**Input**

* The extracted concept node
* Its keywords and hub
* A cached index of Second Brain concepts (built once per run)

**Second Brain indexing (once per run)**
Fetch all Second Brain concept pages (Note Level=Concept). For each, store:

* `concept_id`
* `title`
* `hub`
* `summary` (if you store one)
* `keywords` (if available)

Build a simple in-memory scoring function:

* token overlap between concept keywords and candidate title/keywords
* hub match bonus
* optional fuzzy match

**Output**
For each extracted concept, produce top-K candidates:

* K = 20–50 (configurable)
* each candidate includes:

  * id
  * title
  * hub
  * score

**Writes to Notion**
Update the corresponding Knowledge Inbox page:

* Candidate Matches (JSON) = `[{"id":"...", "title":"...", "hub":"...", "score":...}, ...]`

Now each extracted concept is paired with a shortlist of global concepts it might link to.

---

### Stage 3 (LLM): Link the extracted concept to the candidate set

This is the only stage where edges are created, and they’re still quarantined.

**Input**

* The single extracted concept (full content)
* Its candidate list from Stage 2 (titles + summaries + ids + hubs)
* Strict linking prompt that enforces:

  * choose links **only** among candidates
  * edge caps
  * rationale requirement
  * do not create within-paper cliques

**What the LLM does**
Outputs `ConceptLinkResult`:

* depends_on (≤3)
* enables (≤3)
* generalizes (≤2)
* special_case_of (≤2)
* related (≤5)

Each edge includes:

* target_concept_id (Second Brain page id)
* target_title
* rationale (1–2 sentences, referencing specific math objects)
* confidence

If no candidates fit: output empty arrays.

**Writes to Notion**
Update Knowledge Inbox concept page:

* Edge Suggestions (JSON)
* Graph Link Status → `linked-ai`

Update Paper Tracker:

* Status → `s2b-linked-ai` (optional but recommended)

At this point, you have a **concept graph proposal**: nodes (Knowledge Inbox) + edges (JSON) pointing to existing master nodes (Second Brain).

---

## Human verification and promotion (the “air lock”)

### Verification workflow (human)

You review Knowledge Inbox concepts:

* check correctness of statement
* check assumptions
* check source pages/anchors
* check edge suggestions

Set:

* verification_status = verified or needs-fix
* graph_link_status = needs-review or verified-links

### Promotion workflow (semi-automated)

A separate “promotion” action (script or manual) does:

**Concept promotion**

* Create or merge into an existing Second Brain concept:

  * if concept already exists, append Sources and improve text
  * if new, create it and set Verified=true only after review

**Edge promotion**
Two approaches:

* If you use relations: write the relation fields in Second Brain.
* If you use Edges DB (recommended): create Edge rows with rationale + source paper + confidence and mark them verified.

This step is the only time the master graph changes.

---

## What you end up with: a concept graph you can trust

After promoting:

* Second Brain contains curated concept nodes and hubs.
* Your graph is queryable:

  * “show concepts that depend on monotonicity”
  * “show all existence theorems under graphon coupling”
  * “show which concepts enable convergence of solver X”
* Each edge can have provenance (if you use Edges DB):

  * which paper(s) justify it
  * why it exists
  * who verified it

---

## How the LLM is used (exactly)

You use the LLM for **two things only**:

1. **Extraction** (paper → concept nodes)
2. **Linking** (concept node + candidate set → edges)

You do NOT use the LLM for:

* retrieval (that’s deterministic)
* writing into Second Brain
* creating relations directly

This keeps hallucinations contained.

---

## Why this design avoids the common failure modes

**Failure mode: paper-local cliques**

* prevented by the separation of linking + retrieval to global candidates

**Failure mode: “send all concepts to link”**

* prevented by top-K retrieval

**Failure mode: AI contaminates master graph**

* prevented by quarantine DB + human promotion gate

**Failure mode: unverifiable extraction**

* reduced by source pages/anchors/quotes + confidence

**Failure mode: duplicate concepts**

* mitigated by canonical naming + aliases + candidate retrieval

---

## Operational notes

### Idempotency and reprocessing

* Store `PDF SHA256` and `EXTRACTION_VERSION`
* JobLedger keys off `(attachment_key, sha256, version)`
* Bump version when:

  * prompt changes
  * schema changes
  * extraction output structure changes

### Costs and scaling

* Stage 1: 1 LLM call per paper
* Stage 3: 1 LLM call per extracted concept (or batched 5–10 at once)
* Stage 2: no LLM cost

If you have many concepts, batch Stage 3 to reduce overhead.

---

## Recommendation: choose Edges DB

If you’re serious about a graph, you want auditable edges. Notion relations alone don’t give you provenance per edge, and they become unmanageable.

Edges DB gives you:

* edge-level provenance
* confidence/rationale per edge
* easy rejection/approval workflow
* clean export to networkx/neo4j later

---

If you tell me whether you will use the **Edges DB** (recommended) or keep relations inside Second Brain, I can write the exact property creation checklist and the expected JSON payloads your orchestrator should send to Notion for each stage.

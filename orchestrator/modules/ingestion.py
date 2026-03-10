"""
modules/ingestion.py - Module 1: Core Ingestion Engine
-------------------------------------------------------
3-stage pipeline for each paper in the 'Paper Tracker' Notion DB.

Status flow
-----------
  s1-skim           → pipeline picks up (set by human after skimming)
  s1-processing     → set FIRST on pickup (race-condition guard, REQ-1)
  s2-extracted      → set after all 3 stages complete successfully
  blocked-extraction→ set when GPT returns 0 concepts (REQ-4)
  s1-skim (revert)  → set on any exception (human can retry, REQ-1)
  s2-reextract      → triggers _reextract_missed_concepts (REQ-5)

  PREFLIGHT GATES
  ---------------
  1. Parse Zotero parent key from "Zotero URI" rich_text / url property.
  1b. Resolve attachment key via Zotero API children endpoint.
  2. Check Koofr zip exists ({attachment_key}.zip); if missing set status "s1b-waiting-attachment".
  3. Download the zip, extract the largest PDF (or "primary_pdf_filename" if set).
  4. Compute pdf_sha256; store in "PDF SHA256" property.

  STAGE 1 - EXTRACT
  -----------------
  5. Convert PDF to Markdown via marker-api (tenacity retry).
  6. Strip boilerplate (appendix/refs/acks) from markdown (REQ-2).
  7. Count tokens; dispatch to chunked extraction if > TOKEN_THRESHOLD_CHUNK (REQ-3).
  8. Extract structured knowledge via GPT (ExtractionResult schema).
  9. Zero-concept guard → blocked-extraction if no concepts returned (REQ-4).
  10. Create Knowledge Inbox pages with review checklist prepended (REQ-8).
  Ledger: extract_done

  STAGE 2 - RETRIEVE
  ------------------
  11. For each concept, retrieve top-RETRIEVE_CANDIDATES_K candidates
      (vector search or TF-IDF fallback).
  Ledger: retrieve_done

  STAGE 3 - LINK
  --------------
  12. For each concept + candidates, call GPT to produce ConceptLinkResult edges.
  13. Write Edge Suggestions JSON + graph_link_status = "linked-ai" to KI page.
  14. Patch paper page body with Extracted Concepts section (REQ-9).
  15. Advance paper to s2-extracted with Extraction Count + Tokens.
  Ledger: link_done -> notion_done

Design constraints:
  - Notion text blocks are hard-capped at 1900 chars (safe margin below 2000).
  - Hub suggestions stored as text only - never set Parent Hub relation automatically.
  - JobLedger tracks milestones for idempotency and restart safety.
  - All Koofr / Marker / OpenAI calls wrapped with tenacity exponential backoff.
  - Zotero parent key (from URI) != attachment key (PDF child item). Koofr stores {attachment_key}.zip.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import traceback
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import anthropic
import instructor
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from webdav3.client import Client as WebDAVClient

from .extraction_schema import (
    EDGE_CAPS,
    EXTRACTION_VERSION,
    LATEX_FORMATTING_RULES,
    REEXTRACT_SYSTEM_PROMPT,
    ConceptLinkResult,
    CrossPaperLinkResult,
    EdgeProposal,
    ExtractionResult,
    MathObject,
    latex_sanity_check,
    validate_extraction,
    validate_link_result,
)
from .job_ledger import JobLedger
from .notion_client_wrapper import NotionClientWrapper
from .tag_linter import TagLinter, lint_report_to_text
from .notion_parser import paragraph_blocks_from_latex, sanitize_statement_latex
from .vector_index import VectorIndexEngine
logger = logging.getLogger(__name__)

# -- Scratch directory inside the Docker volume ---------------------------------
TMP_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))

# -- Claude model ---------------------------------------------------------------
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# -- Notion hard limits ---------------------------------------------------------
NOTION_BLOCK_MAX_CHARS = 1900
NOTION_BLOCKS_PER_REQUEST = 100000

# -- Stage 2 candidate retrieval limit -----------------------------------------
RETRIEVE_CANDIDATES_K: int = int(os.environ.get("RETRIEVE_CANDIDATES_K", "30"))

# -- Cross-paper pre-filter thresholds -----------------------------------------
# rapidfuzz score thresholds (0–100 integer scale).
NAMED_TOOL_MATCH_THRESHOLD: int = int(os.environ.get("NAMED_TOOL_MATCH_THRESHOLD", "85"))
SETTING_CONTAINMENT_THRESHOLD: int = int(os.environ.get("SETTING_CONTAINMENT_THRESHOLD", "80"))
# Float thresholds for Jaccard / overlap signals.
ASSUMPTION_OVERLAP_DROP_THRESHOLD: float = float(
    os.environ.get("ASSUMPTION_OVERLAP_DROP_THRESHOLD", "0.05")
)
KEYWORD_JACCARD_DROP_THRESHOLD: float = float(
    os.environ.get("KEYWORD_JACCARD_DROP_THRESHOLD", "0.10")
)
QDRANT_SIMILARITY_DROP_THRESHOLD: float = float(
    os.environ.get("QDRANT_SIMILARITY_DROP_THRESHOLD", "0.75")
)

# -- Composite score weights (must sum to 1.0) ----------------------------------
WEIGHT_QDRANT: float = float(os.environ.get("WEIGHT_QDRANT", "0.40"))
WEIGHT_NAMED_TOOL: float = float(os.environ.get("WEIGHT_NAMED_TOOL", "0.25"))
WEIGHT_ASSUMPTION_OVERLAP: float = float(os.environ.get("WEIGHT_ASSUMPTION_OVERLAP", "0.20"))
WEIGHT_SETTING_CONTAINMENT: float = float(os.environ.get("WEIGHT_SETTING_CONTAINMENT", "0.10"))
WEIGHT_KEYWORD_JACCARD: float = float(os.environ.get("WEIGHT_KEYWORD_JACCARD", "0.05"))

# -- Edge creation thresholds --------------------------------------------------
EDGE_AUTO_CREATE_CONFIDENCE: float = float(
    os.environ.get("EDGE_AUTO_CREATE_CONFIDENCE", "0.80")
)
EDGE_REVIEW_FLAG_CONFIDENCE: float = float(
    os.environ.get("EDGE_REVIEW_FLAG_CONFIDENCE", "0.65")
)
EDGE_MAX_CANDIDATES_TO_GPT: int = int(os.environ.get("EDGE_MAX_CANDIDATES_TO_GPT", "10"))

# -- Candidate hydration -------------------------------------------------------
NOTION_HYDRATION_CONCURRENCY: int = int(
    os.environ.get("NOTION_HYDRATION_CONCURRENCY", "5")
)

# -- Zotero --------------------------------------------------------------------
ZOTERO_API_BASE = "https://api.zotero.org"
# Matches the *parent* item key in a Zotero URI
_ZOTERO_PARENT_RE = re.compile(r"zotero\.org/[^/]+/items/([A-Z0-9]{8})(?:/|$)")
# Matches an explicit attachment key embedded in a Zotero URI
_ZOTERO_ATTACH_RE = re.compile(
    r"zotero\.org/[^/]+/items/[A-Z0-9]{8}/attachment/([A-Z0-9]{8})"
)

# -- Maximum validation errors shown in repair prompt --------------------------
MAX_REPAIR_ERRORS = 5

# -- Token thresholds for chunked extraction (REQ-3) ---------------------------
TOKEN_THRESHOLD_CHUNK = 30_000   # above this: section-by-section extraction
TOKEN_THRESHOLD_WARN  = 60_000   # above this: log warning, still chunk

# -- Sections to skip during chunked extraction --------------------------------
_SKIP_SECTION_KEYWORDS = (
    "proof of", "proofs of", "deferred", "technical lemma",
)

# -- Boilerplate stripping regex (REQ-2) ---------------------------------------
# Matches appendix / references / acknowledgement headings and everything after.
_BOILERPLATE_RE = re.compile(
    r'\n#{1,3}\s*('
    r'References|Bibliography|Works Cited'
    r'|Acknowledgements?|Acknowledgments?'
    r'|Appendix|Appendices|Appendix\s+[A-Z0-9]|[A-Z]\.\s+(?:Proofs?|Appendix)'
    r'|Supplementary\s+Material|Supplemental\s+Material|Supplementary'
    r'|Deferred\s+Proofs?|Proofs?\s+of\s+\w|Technical\s+Lemmas?'
    r'|Funding|Declaration\s+of|Conflicts?\s+of\s+Interest|Author\s+Contributions?'
    r')[^\n]*\n[\s\S]*$',
    re.IGNORECASE,
)

def _count_tokens(text: str) -> int:
    """Approximate token count. Uses tiktoken when available, falls back to 4 chars/token."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


# -- Cross-paper edge scoring dataclasses --------------------------------------


@dataclass
class ConceptData:
    """
    Fully-hydrated concept data fetched from a Notion page.
    Used by score_candidate_pair and the new edge-confirmation prompt.
    """

    notion_page_id: str
    title: str
    concept_type: str
    statement_latex: str
    assumptions: str
    conclusion: str
    setting: list
    named_tools: list
    keywords: list


@dataclass
class CandidateScore:
    """Structural pre-filter scores for a single (C_A, C_B) candidate pair."""

    candidate_id: str           # Notion page ID of C_B
    qdrant_similarity: float    # raw cosine similarity from Qdrant
    named_tool_match: bool      # Signal 1
    assumption_conclusion_overlap: float   # Signal 2, [0, 1]
    setting_containment: Optional[str]     # Signal 3: "A_in_B" | "B_in_A" | None
    keyword_jaccard: float                 # Signal 4, [0, 1]
    composite_score: float                 # weighted combination
    should_drop: bool                      # True = exclude from GPT call


# -- Cross-paper pre-filter helpers --------------------------------------------

# Strip punctuation, lowercase, collapse whitespace for fuzzy string matching.
_PUNCT_RE = re.compile(r'[^\w\s]')
_WS_RE = re.compile(r'\s+')


def _normalize_for_fuzzy(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(' ', s)
    s = _WS_RE.sub(' ', s).strip()
    return s


# LaTeX command pattern — used to strip \command tokens before overlap scoring.
_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+')
_TOKEN_SEP_RE = re.compile(r'[\s\{\}\[\]\(\)\$,;:\.\|]+')


def _tokenize_for_overlap(text: str) -> set:
    """
    Tokenize text for assumption/conclusion overlap scoring.

    - Lowercases.
    - Removes LaTeX command tokens (\\forall, \\mathbb, etc.) to prevent
      LaTeX boilerplate from inflating scores.
    - Splits on whitespace and punctuation.
    - Removes single-character tokens.
    """
    text = text.lower()
    text = _LATEX_CMD_RE.sub(' ', text)
    tokens = _TOKEN_SEP_RE.split(text)
    return {t for t in tokens if t and len(t) > 1}


def _jaccard(s1: set, s2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


def score_candidate_pair(
    concept_a: ConceptData,
    concept_b: ConceptData,
    qdrant_similarity: float,
) -> CandidateScore:
    """
    Compute four structural signals for a (C_A, C_B) candidate pair and return
    a CandidateScore with composite score and drop flag.

    Signal 1 — Named Tool Match (rapidfuzz token_sort_ratio, threshold 85).
    Signal 2 — Assumption-Conclusion Overlap (token Jaccard, max of both
                directions).
    Signal 3 — Setting Containment (rapidfuzz partial_ratio, threshold 80).
    Signal 4 — Keyword Jaccard (exact match after normalisation).
    """
    try:
        from rapidfuzz import fuzz as _fuzz
    except ImportError:
        # rapidfuzz not installed — return a neutral score so the pipeline
        # degrades gracefully without crashing.
        logger.warning(
            "score_candidate_pair: rapidfuzz not installed — returning neutral score."
        )
        composite = WEIGHT_QDRANT * qdrant_similarity
        return CandidateScore(
            candidate_id=concept_b.notion_page_id,
            qdrant_similarity=qdrant_similarity,
            named_tool_match=False,
            assumption_conclusion_overlap=0.0,
            setting_containment=None,
            keyword_jaccard=0.0,
            composite_score=composite,
            should_drop=(qdrant_similarity < QDRANT_SIMILARITY_DROP_THRESHOLD),
        )

    # ── Signal 1: Named Tool Match ────────────────────────────────────────────
    a_title_norm = _normalize_for_fuzzy(concept_a.title)
    b_title_norm = _normalize_for_fuzzy(concept_b.title)
    named_tool_match = False

    for tool in concept_a.named_tools:
        if _fuzz.token_sort_ratio(
            _normalize_for_fuzzy(tool), b_title_norm
        ) >= NAMED_TOOL_MATCH_THRESHOLD:
            named_tool_match = True
            break

    if not named_tool_match:
        for tool in concept_b.named_tools:
            if _fuzz.token_sort_ratio(
                _normalize_for_fuzzy(tool), a_title_norm
            ) >= NAMED_TOOL_MATCH_THRESHOLD:
                named_tool_match = True
                break

    # ── Signal 2: Assumption-Conclusion Overlap ───────────────────────────────
    a_assumptions = _tokenize_for_overlap(concept_a.assumptions)
    b_conclusion = _tokenize_for_overlap(concept_b.conclusion)
    b_assumptions = _tokenize_for_overlap(concept_b.assumptions)
    a_conclusion = _tokenize_for_overlap(concept_a.conclusion)

    overlap1 = _jaccard(a_assumptions, b_conclusion)
    overlap2 = _jaccard(b_assumptions, a_conclusion)
    assumption_conclusion_overlap = max(overlap1, overlap2)

    # ── Signal 3: Setting Containment ────────────────────────────────────────
    setting_containment: Optional[str] = None
    a_setting_str = " ".join(concept_a.setting) if isinstance(concept_a.setting, list) else str(concept_a.setting or "")
    b_setting_str = " ".join(concept_b.setting) if isinstance(concept_b.setting, list) else str(concept_b.setting or "")

    if a_setting_str.strip() and b_setting_str.strip():
        ratio_a_in_b = _fuzz.partial_ratio(
            a_setting_str.lower(), b_setting_str.lower()
        )
        ratio_b_in_a = _fuzz.partial_ratio(
            b_setting_str.lower(), a_setting_str.lower()
        )
        if ratio_a_in_b >= SETTING_CONTAINMENT_THRESHOLD:
            setting_containment = "A_in_B"
        elif ratio_b_in_a >= SETTING_CONTAINMENT_THRESHOLD:
            setting_containment = "B_in_A"

    # ── Signal 4: Keyword Jaccard ─────────────────────────────────────────────
    kw_a = {k.lower().strip() for k in concept_a.keywords if k}
    kw_b = {k.lower().strip() for k in concept_b.keywords if k}
    keyword_jaccard = _jaccard(kw_a, kw_b)

    # ── Composite score ───────────────────────────────────────────────────────
    composite_score = (
        WEIGHT_QDRANT * qdrant_similarity
        + WEIGHT_NAMED_TOOL * float(named_tool_match)
        + WEIGHT_ASSUMPTION_OVERLAP * assumption_conclusion_overlap
        + WEIGHT_SETTING_CONTAINMENT * float(setting_containment is not None)
        + WEIGHT_KEYWORD_JACCARD * keyword_jaccard
    )

    # ── Drop condition ────────────────────────────────────────────────────────
    # Named tool match prevents dropping regardless of other signals.
    if named_tool_match:
        should_drop = False
    else:
        should_drop = (
            assumption_conclusion_overlap < ASSUMPTION_OVERLAP_DROP_THRESHOLD
            and setting_containment is None
            and keyword_jaccard < KEYWORD_JACCARD_DROP_THRESHOLD
            and qdrant_similarity < QDRANT_SIMILARITY_DROP_THRESHOLD
        )

    return CandidateScore(
        candidate_id=concept_b.notion_page_id,
        qdrant_similarity=qdrant_similarity,
        named_tool_match=named_tool_match,
        assumption_conclusion_overlap=assumption_conclusion_overlap,
        setting_containment=setting_containment,
        keyword_jaccard=keyword_jaccard,
        composite_score=composite_score,
        should_drop=should_drop,
    )


def _dominant_signal(score: CandidateScore) -> str:
    """Return the name of the highest-firing pre-filter signal, or 'none'."""
    if score.named_tool_match:
        return "named_tool_match"
    if score.assumption_conclusion_overlap >= ASSUMPTION_OVERLAP_DROP_THRESHOLD:
        return "assumption_conclusion_overlap"
    if score.setting_containment is not None:
        return "setting_containment"
    if score.keyword_jaccard >= KEYWORD_JACCARD_DROP_THRESHOLD:
        return "keyword_jaccard"
    return "none"


def _assign_review_flag(proposal: EdgeProposal, score: CandidateScore) -> EdgeProposal:
    """
    Determine whether an edge should be auto-created cleanly or flagged for
    human review.  Mutates ``proposal.needs_review`` in place and returns it.

    Routing logic:
    - High confidence (>= 0.80) + structural signal → needs_review = False
    - High confidence (>= 0.80) + no signal         → needs_review = True
    - Medium confidence (0.65–0.80) + structural     → needs_review = True
    - Otherwise (low confidence or no grounding)     → needs_review = True
    """
    has_structural_signal = (
        score.named_tool_match
        or score.assumption_conclusion_overlap >= 0.10
        or score.setting_containment is not None
    )
    fields_are_grounded = len(proposal.driving_fields) >= 1

    if not fields_are_grounded:
        proposal.needs_review = True
        return proposal

    high_confidence = proposal.confidence >= EDGE_AUTO_CREATE_CONFIDENCE
    medium_confidence = EDGE_REVIEW_FLAG_CONFIDENCE <= proposal.confidence < EDGE_AUTO_CREATE_CONFIDENCE

    if high_confidence and has_structural_signal:
        proposal.needs_review = False
    elif high_confidence and not has_structural_signal:
        proposal.needs_review = True
    elif medium_confidence and has_structural_signal:
        proposal.needs_review = True
    else:
        proposal.needs_review = True

    return proposal

# -- Stage 3 linking system prompt ---------------------------------------------
LINKING_SYSTEM_PROMPT_V1 = """\
You are a concept-graph linker.

TASK
Given ONE extracted concept and a list of CANDIDATE existing concepts (from a clean knowledge base),
propose directed edges from the extracted concept to candidates.

ABSOLUTE CONSTRAINTS
- You may ONLY link to the provided candidates.
- Use the candidate's exact id and title.
- If no candidate fits, output empty lists for all edge types.
- Precision > recall. False positives are worse than omissions.

EDGE TYPES (DIRECTED)
- depends_on: prerequisites required to understand/prove/apply the extracted concept
- enables: results/methods that become possible because of the extracted concept
- generalizes: the extracted concept is a generalization of the target
- special_case_of: the extracted concept is a special case of the target
- related: meaningful relatedness (shared objects/assumptions/techniques), NOT mere topical similarity

RATIONALE (CRITICAL)
Each edge MUST have a 1–2 sentence rationale referencing specific mathematical objects:
- equation types (HJB/FP/master), operator classes, monotonicity/convexity/Lipschitz, fixed point, contraction, etc.
Do NOT write generic rationales like "they are related".

CAPS (STRICT)
- depends_on ≤ 3
- enables ≤ 3
- generalizes ≤ 2
- special_case_of ≤ 2
- related ≤ 5

CONFIDENCE
- confidence ∈ [0,1]
- Use 0.9 only when the link is very clearly justified by the concept content and candidate description.

OUTPUT FORMAT (STRICT)
Return ONLY valid JSON matching EXACTLY this schema (all keys required, lists may be empty):
{
  "depends_on": [
    {"target_concept_id": "string", "target_title": "string", "rationale": "string", "confidence": number}
  ],
  "enables": [
    {"target_concept_id": "string", "target_title": "string", "rationale": "string", "confidence": number}
  ],
  "generalizes": [
    {"target_concept_id": "string", "target_title": "string", "rationale": "string", "confidence": number}
  ],
  "special_case_of": [
    {"target_concept_id": "string", "target_title": "string", "rationale": "string", "confidence": number}
  ],
  "related": [
    {"target_concept_id": "string", "target_title": "string", "rationale": "string", "confidence": number}
  ]
}
"""

# -- Stage 3 cross-paper edge confirmation system prompt (v2) ------------------
# Used when Qdrant vector search is active (the "new" enriched prompt path).
LINKING_SYSTEM_PROMPT_V2 = """\
You are a mathematical concept relationship analyst. Your job is to determine
whether a directed logical relationship exists between pairs of mathematical
concepts.

You will be given:
- One SOURCE concept (C_A): a concept freshly extracted from a paper.
- A list of TARGET concepts (C_B, C_C, ...): existing concepts in a mathematical
  knowledge base.

For each target concept, you must decide:
1. Does a meaningful logical relationship exist between C_A and this target?
2. If yes: what is the relation type and direction?
3. What is your confidence, and which specific fields drove your decision?

CRITICAL RULES:
- Base your decision ONLY on the mathematical content of the fields provided.
  Do not use the titles alone to infer relationships.
- The `justification` field must reference actual content from the concept
  fields (e.g., specific assumptions, conclusions, or tool names), not just
  topic labels.
- The `driving_fields` list must contain the names of the fields from either
  concept that were the primary evidence. If you cannot identify specific
  fields as evidence, do not propose the edge.
- Do not propose an edge of type `related` unless you can identify at least
  one shared structural element (shared assumption, shared tool, overlapping
  setting). Topic similarity alone does not justify `related`.
- For `generalizes` / `special_case_of`: the settings or assumption sets must
  have a clear containment relationship. State which is more general.
- For `depends_on` / `enables`: one concept's conclusion must appear
  (exactly or approximately) in the other's assumptions, OR one concept's
  named_tools must reference the other.
- Return an empty proposals list if no relationships meet these criteria. Do not
  fabricate relationships to be helpful.

DIRECTION CONVENTION:
- "A_to_B" means the edge goes FROM C_A TO C_B.
  Example: if C_A depends_on C_B, direction is "A_to_B".
  Example: if C_A generalizes C_B, direction is "A_to_B".
- "B_to_A" means the edge goes FROM C_B TO C_A.
  Example: if C_B depends_on C_A, direction is "B_to_A".
"""

# -- System prompt template -----------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """\
You are a mathematical extraction engine for applied mathematics papers (MFG/PDE/probability/optimization).
You extract a SMALL set of reusable mathematical concept nodes from ONE paper, from Markdown input.

LATEX FORMATTING RULES (STRICTLY ENFORCED — violations will break rendering)
═════════════════════════════════════════════════════════════════════════════

1. DELIMITERS — every LaTeX expression must be wrapped. No exceptions.
   - Inline math:  $...$       → for symbols, variables, short expressions
   - Display math: \\[...\\]   → for full statements, multi-line equations
   - NEVER use $$...$$ (double-dollar) — use \\[...\\] for display math
   - NEVER write bare LaTeX commands outside a delimiter:
       WRONG:  \\partial_\\alpha f(\\alpha^*)=0
       CORRECT: $\\partial_\\alpha f(\\alpha^*) = 0$

2. ENVIRONMENTS — must always be nested inside \\[...\\]
   - CORRECT:  \\[\\begin{aligned} f(x) &= 0 \\\\ g(x) &= 1 \\end{aligned}\\]
   - WRONG:    \\begin{aligned} f(x) &= 0 \\\\ g(x) &= 1 \\end{aligned}
   - Multi-line equations: use \\\\ for line breaks inside \\begin{aligned}
   - NEVER use \\begin{equation} — use \\[...\\] directly
   - NEVER write literal newlines between \\[ and \\] without \\begin{aligned}

3. \\text{} — only valid inside a math environment
   - WRONG:  \\text{If condition holds, then } \\alpha \\in (0,1)
   - CORRECT: "If condition holds, then $\\alpha \\in (0,1)$"
   - To mix prose and math: write prose outside delimiters, math inside

4. FORBIDDEN IN ALL FIELDS
   - \\tag{N}       — equation numbers from the source paper are meaningless here
   - \\label{...}   — no cross-referencing
   - \\ref{...}     — no cross-referencing
   - \\nonumber     — irrelevant outside a document
   - \\begin{equation} / \\end{equation}

5. NOTATION — always use the canonical form
   - Fractions:    \\frac{a}{b}         NEVER a/b in display math
   - Norms:        \\|x\\|              NEVER ||x||
   - Inner product: \\langle x,y \\rangle  NEVER <x,y>
   - Real numbers: \\mathbb{R}          NEVER just R
   - Expectation:  \\mathbb{E}          NEVER E[...]
   - Probability:  \\mathbb{P}          NEVER P(...)
   - Implies:      \\Rightarrow         NEVER =>
   - Iff:          \\Leftrightarrow     NEVER <=>

6. FIELD-SPECIFIC RULES

   statement_latex:
     - Must be exactly ONE \\[...\\] block containing the complete formal statement
     - If the statement has multiple equations, use \\begin{aligned}...\\end{aligned}
       inside the \\[...\\]
     - Must be self-contained and parseable by KaTeX
     - Example:
         \\[\\begin{aligned}
           \\partial_\\alpha f_\\ell(\\alpha^*) &= 0, \\quad \\forall \\ell \\in \\{1,\\dots,L\\} \\\\
           \\partial_{\\alpha\\alpha}^2 f_\\ell(\\alpha^*) &\\geq 0
         \\end{aligned}\\]

   assumptions:
     - Plain English prose only
     - Mathematical objects in inline $...$ only — NO display math blocks
     - Example: "Finite state/action spaces; $\\alpha \\in [0,1]$; Lipschitz
       graphon $W \\in L^p$"

   variables:
     - One variable per line as: $<symbol>$ (<plain English description>)
     - Example: "$\\alpha \\in [0,1]$ (node index), $m^\\alpha$ (initial mean)"

   conclusion:
     - Plain English — avoid LaTeX unless unavoidable
     - If LaTeX is needed, inline $...$ only — no display math

   interpretation:
     - Plain English — same rule as conclusion

   proof_idea:
     - May use inline $...$ freely
     - No display math blocks

GOAL
Produce high-fidelity, reusable mathematical "Concept Nodes" suitable for a long-term concept graph.
This is NOT summarization. Do not invent. Do not add general background material that is not in the paper.

OUTPUT BIAS
Prefer FEWER, HIGHER-VALUE concepts rather than many low-value fragments.
A false concept is worse than a missed concept.

HUBS
You MUST assign exactly one hub per concept:
- The hub MUST be one of ALLOWED_HUBS (provided below) or "Uncategorized".
- Never invent hubs.

CONCEPT GRANULARITY RULE (CRITICAL)
Extract only concepts that are useful beyond this single paper.

Include:
- Main definitions that introduce new objects / equilibrium notions / operators.
- Main theorems (existence/uniqueness/stability/convergence/characterization).
- Algorithms/procedures that can be reused (not just “we compute this”).
- Key assumptions ONLY if they are used as reusable conditions (e.g., monotonicity, convexity, Lipschitz).

Exclude by default:
- “Lemma A used only to prove Theorem B” (do NOT include A unless independently reusable).
- Intermediate inequalities, technical estimates, proof bookkeeping.
- Restatements of known textbook facts unless the paper uses them as a named condition central to the contribution.
- Numbered titles like "Theorem 1" or "Lemma 3.2".

EXCEPTION (INTERNAL LEMMA RULE)
If the paper proves a big result using a smaller lemma that is clearly a standard reusable tool
(e.g., a contraction estimate, monotonicity lemma, stability inequality) AND it is stated cleanly as a general-purpose statement,
then you MAY extract that lemma as its own concept. Otherwise omit it.

NAMING RULE (CRITICAL)
Every concept title must be a descriptive canonical name that stands alone.
Do NOT use numbering.
The title MUST be straight to the point, very dense. Few words, but still identifiable. 
Bad: "Theorem 1", "Lemma 2.3", "Equation (5)".
Good: "Existence of Mean Field Game Equilibrium under Lasry–Lions Monotonicity", "Convergence of Policy Iteration for Finite-State MFG".

MATHEMATICAL FIDELITY RULES
- If the statement is present in the Markdown: reproduce it as exactly as possible in LaTeX.
- Do NOT paraphrase equations into different symbols.
- If the exact statement is not available (e.g., badly extracted), you may give a best-effort reconstruction, but then reduce confidence.

ASSUMPTIONS / BOUNDARY CONDITIONS
- Explicitly list all assumptions needed for the statement.
- Include boundary/terminal conditions if the result is PDE-based.
- If none are explicitly stated: write "None explicitly stated."

VARIABLES FIELD
Give a comma-separated list of variable descriptions, e.g.:
"x∈Ω (state), t∈[0,T] (time), m_t (population distribution), V(t,x) (value function), H(x,p,m) (Hamiltonian)"

CONCLUSION FIELD
Explain the result in plain English.
No marketing language.

KEYWORDS (FOR GRAPH RETRIEVAL)
You MUST produce three keyword lists per concept:
- canonical_keywords maximum 15 terms describing what the concept IS
- prereq_keywords: maximum 15 terms describing what the concept REQUIRES
- downstream_keywords: maximum 15 terms describing what the concept ENABLES

Keyword format rules:
- lowercase
- hyphen-separated
- 1–4 words per keyword
- examples: "lasry-lions-monotonicity", "fixed-point-existence", "viscosity-solution", "graphon-coupling"

OPTIONAL FIELDS (include only if supported by the text)
- interpretation: plain-English intuition (≤ 3 sentences)
- proof_idea: high-level reusable technique (≤ 3 sentences), NOT a full proof
- source_anchors: section/equation refs like "Section 3.2; Eq. (12); Theorem 4.1"
- named_tools: named theorems/techniques explicitly referenced (e.g., Schauder, Kakutani, Gronwall)
- setting: list of setting tags such as finite_state, continuous, graphon, ergodic, common_noise, etc...
- result_category: one of {existence, uniqueness, convergence, stability, approximation, etc...}
- aliases: short list of alternative names for the concept (strings)

TRACEABILITY
- source_pages must be a list of integers (pages where the statement appears).
- source_quotes: optional short quote ≤ 25 words (verbatim) or null.

CONFIDENCE
Return confidence ∈ [0,1]:
- 0.9-1.0: statement clearly present and clean
- 0.6-0.8: mostly clear but minor reconstruction
- 0.0-0.5: extraction uncertain / noisy

OUTPUT FORMAT (STRICT)
Return ONLY valid JSON matching this schema exactly (no extra keys):
{
  "one_liner": string,
  "active_themes": [string],
  "extracted_concepts": [
    {
      "type": "Definition"|"Theorem"|"Lemma"|"Algorithm"|"Assumption"|"ProofTechnique",
      "title": string,
      "statement_latex": string,
      "assumptions": string,
      "variables": string,
      "conclusion": string,
      "source_pages": [int],
      "source_quotes": string|null,
      "confidence": number,
      "suggested_hub": string,
      "canonical_keywords": [string],
      "prereq_keywords": [string],
      "downstream_keywords": [string],
      "interpretation": string (optional),
      "proof_idea": string (optional),
      "source_anchors": string (optional),
      "named_tools": [string] (optional),
      "setting": [string] (optional),
      "result_category": string (optional),
      "aliases": [string] (optional)
    }
  ]
}

ALLOWED_HUBS:
[INJECT_DYNAMIC_HUBS_HERE]
"""


class IngestionEngine:
    """Module 1: Core Ingestion Engine (Notion -> WebDAV -> Marker -> OpenAI -> Notion)."""

    def __init__(self, vector_index: Optional[VectorIndexEngine]) -> None:
        self.notion = NotionClientWrapper()
        _anthropic = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.claude_client = instructor.from_anthropic(_anthropic)
        self.anthropic_raw = _anthropic
        self._webdav = self._build_webdav_client()
        self.marker_url = os.environ.get("MARKER_API_URL", "http://marker-api:8080")
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.koofr_base = os.environ.get("KOOFR_PDF_PATH", "/zotero")
        self._ledger = JobLedger()
        self._tag_linter = TagLinter()
        self.zotero_user_id = os.environ["ZOTERO_USER_ID"]
        self.zotero_api_key = os.environ["ZOTERO_API_KEY"]
        # Lazy-cached {prop_name: prop_type} for the Knowledge Inbox DB.
        # Populated on first use by _get_ki_schema().
        self._ki_schema: dict[str, str] | None = None
        # Module 7: VectorIndexEngine — only active when VECTOR_INDEX_ENABLED is set.
        # Falls back to TF-IDF silently if Qdrant is unreachable.
        self._vector_index: VectorIndexEngine | None = vector_index if vector_index else None
        self.koofr_markdown_dir = os.environ.get("KOOFR_MARKDOWN_PATH", "/zotero_markdown")
        self._ensure_koofr_markdown_dir()
    # -- WebDAV client factory --------------------------------------------------

    @staticmethod
    def _build_webdav_client() -> WebDAVClient:
        options = {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": os.environ["KOOFR_USER"],
            "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
        }
        return WebDAVClient(options)

    # -- Entry point ------------------------------------------------------------

    def run(self) -> None:
        """
        Poll 'Paper Tracker' for papers to process.

        Queries two status values:
          s1-skim       — primary extraction queue (REQ-3)
          s2-reextract  — targeted re-extraction of missed concepts (REQ-5)

        Hubs and the Second Brain concept index are fetched once per run()
        invocation so that every paper in the same batch uses a consistent
        snapshot.
        """
        logger.info("Ingestion: polling for s1-skim and s2-reextract papers ...")
        pages = self.notion.query_database(
            self.paper_tracker_db,
            filter={
                "property": "Status",
                "status": {"equals": "s1-skim"},
            },
        )
        pages_to_reextract = self.notion.query_database(
            self.paper_tracker_db,
            filter={
                "property": "Status",
                "status": {"equals": "s2-reextract"},
            },
        )
        logger.info(
            "Ingestion: found %d paper(s) to extract, %d paper(s) to re-extract.",
            len(pages),
            len(pages_to_reextract),
        )

        if not pages and not pages_to_reextract:
            return

        hubs: dict[str, str] = self._fetch_allowed_hubs()
        sb_index: list[dict] = self._build_second_brain_index()
        logger.info(
            "Ingestion: loaded %d hub(s), %d Second Brain concept(s).",
            len(hubs),
            len(sb_index),
        )

        for page in pages:
            try:
                self._process_paper(page, hubs, sb_index)
            except Exception:
                logger.exception("Failed to process page %s", page["id"])

        for page in pages_to_reextract:
            try:
                self._reextract_missed_concepts(page, hubs, sb_index)
            except Exception:
                logger.exception("Failed re-extraction for page %s", page["id"])

    # -- Hub fetching -----------------------------------------------------------

    def _fetch_allowed_hubs(self) -> dict[str, str]:
        """
        Query the Second Brain DB for Hub pages.

        Returns
        -------
        dict mapping hub name -> Notion page ID.
        """
        logger.debug("Ingestion: fetching Hub pages from Second Brain ...")
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Note Level",
                "select": {"equals": "Hub"},
            },
        )
        hubs: dict[str, str] = {}
        for page in pages:
            name = self._get_page_title(page)
            if name:
                hubs[name] = page["id"]
        return hubs

    # -- Second Brain index ----------------------------------------------------

    def _build_second_brain_index(self) -> list[dict]:
        """
        Query the Second Brain DB for Concept-level pages and return a flat
        list of concept records used by Stage 2 candidate retrieval.

        Each record has keys:
            id, title, hub, summary, tags, keywords_bag (set[str])

        The Note Level filter is controlled by SB_CONCEPT_LEVEL (default: "Concept").
        """
        concept_level = os.environ.get("SB_CONCEPT_LEVEL", "Concept")
        logger.debug(
            "Ingestion: building Second Brain index (Note Level='%s') ...",
            concept_level,
        )
        pages = self.notion.query_database(
            self.second_brain_db,
            filter={
                "property": "Note Level",
                "select": {"equals": concept_level},
            },
        )
        records: list[dict] = []
        for page in pages:
            title = self._get_page_title(page)
            if not title:
                continue
            props = page.get("properties", {})
            hub = self._get_text_prop(props, "Suggested Hub")
            summary = self._get_text_prop(props, "One Liner")
            tags = self._get_multi_select_prop(props, "Tags")
            keywords = self._get_multi_select_prop(props, "Keywords")

            def _tokenise(s: str) -> set:
                return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

            bag: set = set()
            bag |= _tokenise(title)
            for kw in keywords:
                bag |= _tokenise(kw)
            for tag in tags:
                bag |= _tokenise(tag)

            records.append(
                {
                    "id": page["id"],
                    "title": title,
                    "hub": hub,
                    "summary": summary,
                    "tags": tags,
                    "keywords_bag": bag,
                }
            )
        logger.debug(
            "Ingestion: Second Brain index built -- %d concept(s).", len(records)
        )
        return records

    # -- Per-paper pipeline ----------------------------------------------------

    def _process_paper(
        self,
        page: dict,
        hubs: dict[str, str],
        sb_index: list[dict],
    ) -> None:
        """
        Full ingestion pipeline for a single paper: preflight gates -> Stage 1
        (EXTRACT) -> Stage 2 (RETRIEVE) -> Stage 3 (LINK).
        """
        page_id = page["id"]
        props = page["properties"]

        # REQ-1: Set s1-processing FIRST — race condition guard.
        # Must be the very first operation so a second scheduler tick skips this paper.
        self.notion.update_page(
            page_id=page_id,
            properties={"Status": self.notion.status_prop("s1-processing")},
        )

        run_id = uuid.uuid4().hex[:8]
        local_pdf: Path | None = None
        job_id: int | None = None
        cleaned_tokens: int = 0

        try:
            # -- Preflight gate 1: Parse Zotero parent key --------------------------
            zotero_uri = self._get_text_prop(props, "Zotero URI")
            parent_match = _ZOTERO_PARENT_RE.search(zotero_uri)
            if not parent_match:
                logger.warning(
                    "[%s] Missing or invalid Zotero URI: '%s' — reverting to s1-skim.",
                    page_id,
                    zotero_uri,
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1-skim"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            "Missing or invalid Zotero URI."
                        )},
                    },
                )
                return
            parent_key = parent_match.group(1)

            # -- Preflight gate 1b: Resolve attachment key ----------------------
            resolved = self._resolve_keys_and_update_notion(
                page_id, zotero_uri, parent_key, run_id
            )
            if resolved is None:
                logger.warning(
                    "[%s] Cannot resolve attachment key for parent '%s' "
                    "-- setting s1b-waiting-attachment.",
                    run_id,
                    parent_key,
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1b-waiting-attachment")
                    },
                )
                return
            
            parent_key, attachment_key = resolved
            # -- Preflight gate 2: Check Koofr zip exists ----------------------
            zip_remote = f"{self.koofr_base}/{attachment_key}.zip"
            logger.info("[%s] Checking Koofr zip: %s", run_id, zip_remote)
            if not self._koofr_exists(zip_remote):
                logger.warning(
                    "[%s] Zip not found -- setting s1b-waiting-attachment.", run_id
                )
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1b-waiting-attachment")
                    },
                )
                return

            # -- STAGE 1 / Step 1: Convert PDF to Markdown ---------------------
            logger.info("[%s] Stage 1: converting PDF to Markdown ...", run_id)
            markdown_text, job_id = self._pdf_to_markdown(
                attachment_key=attachment_key,
                run_id=run_id,
                # PDF extraction args — only used on cache miss
                zip_remote=zip_remote,
                primary_pdf_filename=self._get_text_prop(props, "primary_pdf_filename") or None,
                page_id=page_id,
                props=props,
            )
            if markdown_text is None:
                # _pdf_to_markdown already updated status to s1b-waiting-attachment
                # or logged the error — just bail.
                return

            self._ledger.update_status(job_id, "marker_done")


            # REQ-2: Strip boilerplate BEFORE token counting or any GPT call.
            markdown_text = self._strip_boilerplate(markdown_text)
            # REQ-3: Count tokens after stripping; used to decide chunking.
            cleaned_tokens = _count_tokens(markdown_text)

            # -- STAGE 1 / Step 2: Extract via OpenAI --------------------------
            logger.info(
                "[%s] Stage 1: extracting knowledge via OpenAI (%d tokens) ...",
                run_id, cleaned_tokens,
            )
            extraction = self._run_extraction(markdown_text, cleaned_tokens, hubs, run_id)
            self._ledger.update_status(job_id, "openai_done")

            # -- STAGE 1 / Step 3: Patch Paper Tracker metadata ----------------
            logger.info("[%s] Stage 1: patching Notion paper row ...", run_id)
            # REQ-9: Set Thesis Relevance = supporting as default if unset.
            self._patch_notion_page(
                page_id, extraction, run_id,
            )

            # -- STAGE 1 / Step 4: Create Knowledge Inbox entries --------------
            concepts = extraction.extracted_concepts

            # REQ-4: Zero concept guard — block paper rather than silently advancing.
            if len(concepts) == 0:
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("blocked-extraction"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            "GPT returned 0 concepts. Check markdown quality or "
                            "add Re-extract Hints and set status back to s1-skim."
                        )},
                        "Extraction Count": {"number": 0},
                    },
                )
                logger.warning(
                    "Ingestion: zero concepts extracted for paper %s — "
                    "set to blocked-extraction.",
                    page_id,
                )
                return

            logger.info(
                "[%s] Stage 1: creating %d Knowledge Inbox page(s) ...",
                run_id,
                len(concepts),
            )
            ki_pages: list[tuple[MathObject, str]] = []
            for concept in concepts:
                try:
                    ki_page_id = self._create_knowledge_item(page_id, concept, hubs)
                    ki_pages.append((concept, ki_page_id))
                    # Module 7: index in Qdrant immediately after KI creation.
                    if self._vector_index and self._vector_index.available:
                        try:
                            self._vector_index.index_concept(
                                concept, ki_page_id, verified=False
                            )
                        except Exception:
                            logger.warning(
                                "[%s] VectorIndex: failed to index '%s' — continuing.",
                                run_id, concept.title,
                            )
                except Exception:
                    logger.exception(
                        "[%s] Failed to create knowledge item '%s'",
                        run_id,
                        concept.title,
                    )
            logger.info("[%s] Created %d Knowledge Inbox page(s).", run_id, len(ki_pages))
            self._ledger.update_status(job_id, "extract_done")

            # Inject newly created concepts into sb_index so Stage 2/3 can link to them.
            self._inject_ki_pages_into_index(ki_pages, sb_index)

            # -- STAGE 2: Retrieve candidates ----------------------------------
            logger.info("[%s] Stage 2: retrieving candidates from Second Brain ...", run_id)
            # Build the set of same-paper KI IDs so the pre-filter can skip them.
            all_ki_ids = {ki_id for _, ki_id in ki_pages}
            concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
            for concept, ki_page_id in ki_pages:
                same_paper_ids = all_ki_ids - {ki_page_id}
                candidates = self._retrieve_candidates_for_concept(
                    concept, sb_index,
                    current_page_id=ki_page_id,
                    same_paper_ids=same_paper_ids,
                )
                self._update_knowledge_item_candidates(ki_page_id, candidates)
                concept_candidates.append((concept, ki_page_id, candidates))
                logger.info(
                    "[%s] '%s': %d candidate(s) retrieved.",
                    run_id,
                    concept.title,
                    len(candidates),
                )
            self._ledger.update_status(job_id, "retrieve_done")

            # -- STAGE 3: LLM linking ------------------------------------------
            logger.info("[%s] Stage 3: LLM linking ...", run_id)
            for concept, ki_page_id, candidates in concept_candidates:
                try:
                    link_result = self._run_stage_link(concept, candidates, run_id)
                    self._update_knowledge_item_graph_data(ki_page_id, link_result)
                except Exception:
                    logger.exception(
                        "[%s] Link stage failed for concept '%s'",
                        run_id,
                        concept.title,
                    )
            self._ledger.update_status(job_id, "link_done")

            # -- Finalise ------------------------------------------------------
            self._patch_notion_paper_post_linking(page_id, run_id)
            self._ledger.update_status(job_id, "notion_done")
            self._ledger.finish_job(job_id)

            # REQ-9: Patch paper page body (idempotent — checks for existing heading).
            self._patch_paper_page(page_id, [ki_id for _, ki_id in ki_pages])

            # REQ-1: Advance to s2-extracted with extraction counts.
            self.notion.update_page(
                page_id=page_id,
                properties={
                    "Status": self.notion.status_prop("s2-extracted"),
                    "Extraction Count": {"number": len(ki_pages)},
                    "Extraction Tokens": {"number": cleaned_tokens},
                },
            )
            logger.info("[%s] Done.", run_id)

        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", run_id, exc)
            if job_id is not None:
                self._ledger.update_status(job_id, "failed", error=str(exc))
            try:
                self.notion.update_page(
                    page_id=page_id,
                    properties={
                        "Status": self.notion.status_prop("s1-skim"),
                        "Extraction Error": {"rich_text": self.notion.rich_text(
                            traceback.format_exc()[:2000]
                        )},
                        "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
                    },
                )
            except Exception:
                logger.warning(
                    "[%s] Could not write error context to Notion Paper Tracker.", run_id
                )
            raise

        finally:
            if local_pdf is not None and local_pdf.exists():
                local_pdf.unlink()

    def _inject_ki_pages_into_index(
        self,
        ki_pages: list[tuple[MathObject, str]],
        sb_index: list[dict],
    ) -> None:
        """
        Append freshly created Knowledge Inbox concepts into the live sb_index
        so that Stage 2/3 candidate retrieval can link to them.

        This handles both intra-paper linking (concepts within the same paper)
        and inter-paper linking (concepts from earlier papers in the same batch).
        """
        def _toks(s: str) -> set:
            return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

        for concept, ki_page_id in ki_pages:
            bag: set = set()
            for kw_list in (
                concept.canonical_keywords,
                concept.prereq_keywords,
                concept.downstream_keywords,
            ):
                for kw in kw_list:
                    bag |= _toks(kw)
            bag |= _toks(concept.title)

            sb_index.append({
                "id": ki_page_id,
                "title": concept.title,
                "hub": concept.suggested_hub or "",
                "summary": concept.conclusion or "",
                "tags": concept.setting or [],
                "keywords_bag": bag,
            })

        logger.debug(
            "Injected %d new concept(s) into sb_index (total now: %d).",
            len(ki_pages),
            len(sb_index),
        )
        
    # -- Zotero helpers --------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _zotero_children(self, parent_key: str) -> list[dict]:
        """Fetch children of a Zotero item via the REST API."""
        url = (
            f"{ZOTERO_API_BASE}/users/{self.zotero_user_id}"
            f"/items/{parent_key}/children"
        )
        resp = requests.get(
            url,
            headers={"Zotero-API-Key": self.zotero_api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_attachment_key(
        self, zotero_uri: str, parent_key: str
    ) -> tuple[str, str] | None:
        """
        Resolve the attachment key (PDF child item) for a Zotero parent.

        Resolution order:
        1. If the URI itself contains an explicit attachment key, use it.
        2. Query Zotero API children; pick the first PDF attachment.

        Returns (parent_key, attachment_key) or None if not found.
        """
        attach_match = _ZOTERO_ATTACH_RE.search(zotero_uri)
        if attach_match:
            return parent_key, attach_match.group(1)

        try:
            children = self._zotero_children(parent_key)
        except Exception:
            logger.warning(
                "Could not fetch Zotero children for parent '%s'.", parent_key
            )
            return None

        pdf_children: list[tuple[str, dict]] = []
        for child in children:
            data = child.get("data", {})
            link_mode = data.get("linkMode", "")
            content_type = data.get("contentType", "")
            if link_mode in ("imported_file", "imported_url") and "pdf" in content_type:
                attach_key = child.get("key")
                if attach_key:
                    pdf_children.append((attach_key, data))

        if not pdf_children:
            logger.warning("No PDF attachment found for Zotero parent '%s'.", parent_key)
            return None

        # Prefer attachment whose filename starts with parent_key (Zotero convention)
        for attach_key, data in pdf_children:
            filename = data.get("filename", "")
            if filename.lower().startswith(parent_key.lower()):
                return parent_key, attach_key

        # Fallback: pick attachment with largest fileSize
        pdf_children.sort(key=lambda x: x[1].get("fileSize", 0), reverse=True)
        return parent_key, pdf_children[0][0]

    def _resolve_keys_and_update_notion(
        self,
        page_id: str,
        zotero_uri: str,
        parent_key: str,
        run_id: str,
    ) -> tuple[str, str] | None:
        """
        Resolve (parent_key, attachment_key) and write the attachment key to
        the Paper Tracker Notion page for auditability.

        Returns (parent_key, attachment_key) or None on failure.
        """
        resolved = self._resolve_attachment_key(zotero_uri, parent_key)
        if resolved is None:
            return None
        _parent_key, attachment_key = resolved
        try:
            self.notion.update_page(
                page_id=page_id,
                properties={
                    "Zotero Attachment Key": {
                        "rich_text": self.notion.rich_text(attachment_key)
                    }
                },
            )
        except Exception:
            logger.warning(
                "[%s] Could not write Zotero Attachment Key to Notion.", run_id
            )
        return _parent_key, attachment_key

    # -- Koofr helpers ---------------------------------------------------------

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _koofr_exists(self, remote_path: str) -> bool:
        """Return True if remote_path exists on Koofr."""
        try:
            return self._webdav.check(remote_path)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "404" in exc_str or "not found" in exc_str or "no such" in exc_str:
                return False
            raise

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _download_koofr(self, remote_path: str, local_path: Path) -> None:
        """Download remote_path from Koofr to local_path."""
        self._webdav.download_sync(remote_path=remote_path, local_path=str(local_path))

    @staticmethod
    def _extract_pdf_from_zip(
        zip_path: Path, output_path: Path, preferred: str | None = None
    ) -> None:
        """
        Extract a PDF from zip_path to output_path.

        If preferred is set and found in the archive, use that file.
        Otherwise, extract the largest PDF in the archive.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            pdf_entries = [
                e for e in zf.infolist() if e.filename.lower().endswith(".pdf")
            ]
            if not pdf_entries:
                raise FileNotFoundError(f"No PDF found inside {zip_path}")

            if preferred:
                match = next(
                    (e for e in pdf_entries if Path(e.filename).name == preferred),
                    None,
                )
                if match:
                    data = zf.read(match.filename)
                    output_path.write_bytes(data)
                    return
                logger.warning(
                    "primary_pdf_filename '%s' not found in zip; using largest PDF.",
                    preferred,
                )

            largest = max(pdf_entries, key=lambda e: e.file_size)
            data = zf.read(largest.filename)
            output_path.write_bytes(data)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute the hex SHA-256 digest of a file."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- Marker API ------------------------------------------------------------


    def _pdf_to_markdown(
        self,
        attachment_key: str,
        run_id: str,
        zip_remote: str,
        primary_pdf_filename: str | None,
        page_id: str,
        props: dict,
    ) -> tuple[str, int] | tuple[None, None]:
        """
        Return the Markdown text for a paper, using a Koofr cache.

        Cache hit  (fast path):
            /zotero_markdown/{attachment_key}.md exists on Koofr
            → download and return immediately, no marker call needed.

        Cache miss (slow path):
            1. Download zip from Koofr
            2. Extract PDF
            3. Compute SHA256, write to Notion
            4. Start job ledger
            5. POST to marker-api
            6. Strip boilerplate
            7. Upload .md to Koofr cache
            8. Return markdown

        Returns None if a fatal error occurs (status already updated by caller).
        """
        md_remote = f"{self.koofr_markdown_dir}/{attachment_key}.md"

        # ── Cache hit ─────────────────────────────────────────────────────────
        if self._koofr_exists(md_remote):
            logger.info(
                "[%s] Markdown cache hit: %s — skipping marker conversion.",
                run_id, md_remote,
            )
            try:
                raw = self._koofr_download_bytes(md_remote)
                return raw.decode("utf-8")
            except Exception:
                logger.warning(
                    "[%s] Markdown cache read failed — falling through to re-conversion.",
                    run_id,
                )
                # Fall through to slow path below.

        # ── Cache miss — full pipeline ────────────────────────────────────────
        logger.info("[%s] Markdown cache miss — converting PDF via marker.", run_id)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        local_zip = TMP_DIR / f"{run_id}.zip"
        local_pdf = TMP_DIR / f"{run_id}.pdf"

        try:
            self._download_koofr(zip_remote, local_zip)
            self._extract_pdf_from_zip(
                local_zip, local_pdf, preferred=primary_pdf_filename
            )
        finally:
            local_zip.unlink(missing_ok=True)

        # SHA256 + ledger (only on cache miss — PDF was just extracted)
        pdf_sha256 = self._sha256(local_pdf)
        logger.info("[%s] PDF SHA256: %s", run_id, pdf_sha256)
        self.notion.update_page(
            page_id=page_id,
            properties={"PDF SHA256": {"rich_text": self.notion.rich_text(pdf_sha256)}},
        )
        job_id = self._ledger.start_job(attachment_key, pdf_sha256, EXTRACTION_VERSION)
        logger.info("[%s] JobLedger job_id=%d", run_id, job_id)

        # Marker conversion
        markdown_text = self._call_marker(local_pdf)
        local_pdf.unlink(missing_ok=True)

        # Strip boilerplate before caching — cache stores the clean version
        markdown_text = self._strip_boilerplate(markdown_text)
        token_count = _count_tokens(markdown_text)
        logger.info(
            "[%s] Markdown ready: %d tokens after boilerplate strip.", run_id, token_count
        )

        # Upload to Koofr cache
        try:
            self._koofr_upload(md_remote, markdown_text.encode("utf-8"))
            logger.info("[%s] Markdown cached → %s", run_id, md_remote)
        except Exception:
            logger.warning(
                "[%s] Markdown cache upload failed — continuing without cache.",
                run_id,
            )

        return markdown_text, job_id
    
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _call_marker(self, pdf_path: Path) -> str:
        """POST PDF path to the marker-api container and return raw Markdown."""
        response = requests.post(
            f"{self.marker_url}/marker",
            json={"filepath": str(pdf_path)},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("markdown")
            or data.get("output")
            or data.get("text")
            or response.text
        )

    def _koofr_upload(self, remote_path: str, data: bytes) -> None:
        """Upload bytes to Koofr via WebDAV. Writes to a temp file first."""
        tmp = TMP_DIR / f"_upload_{uuid.uuid4().hex[:8]}.tmp"
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            self._webdav.upload_sync(
                remote_path=remote_path,
                local_path=str(tmp),
            )
        finally:
            tmp.unlink(missing_ok=True)

    def _koofr_download_bytes(self, remote_path: str) -> bytes:
        """Download a file from Koofr and return raw bytes."""
        tmp = TMP_DIR / f"_download_{uuid.uuid4().hex[:8]}.tmp"
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            self._webdav.download_sync(
                remote_path=remote_path,
                local_path=str(tmp),
            )
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    
    # -- Boilerplate stripping (REQ-2) -----------------------------------------

    def _strip_boilerplate(self, text: str) -> str:
        stripped = _BOILERPLATE_RE.sub("", text)
        ratio = len(stripped) / max(len(text), 1)
        if ratio < 0.2:
            logger.warning(
                "Boilerplate strip removed >80%% of document (%.0f%% remaining) — "
                "regex may have matched too early. Returning original.",
                ratio * 100,
            )
            return text
        logger.debug("Boilerplate strip: %.0f%% retained.", ratio * 100)
        return stripped

    # -- Extraction dispatcher (REQ-3) -----------------------------------------

    def _run_extraction(
        self,
        markdown: str,
        token_count: int,
        hubs: dict[str, str],
        run_id: str,
    ) -> ExtractionResult:
        """
        Dispatch to single-shot or section-by-section extraction depending on
        token count vs TOKEN_THRESHOLD_CHUNK.
        """
        if token_count > TOKEN_THRESHOLD_WARN:
            logger.warning(
                "[%s] Paper is very long (%d tokens) — may risk output truncation.",
                run_id, token_count,
            )
        if token_count <= TOKEN_THRESHOLD_CHUNK:
            return self._extract_and_validate(markdown, hubs, run_id)
        logger.info(
            "[%s] Paper exceeds %d tokens — using section-by-section extraction.",
            run_id, TOKEN_THRESHOLD_CHUNK,
        )
        return self._chunked_extract(markdown, hubs, run_id, token_count)

    def _extract_preamble(self, markdown: str, max_tokens: int = 3000) -> str:
        """
        Extract the abstract + introduction + notation as shared context for
        all section-level extraction calls. Truncated to max_tokens.
        """
        lines = markdown.split("\n")
        preamble_lines: list[str] = []
        in_intro = True
        for line in lines:
            # Stop collecting once we hit a top-level section heading
            # that is NOT introduction-related.
            if line.startswith("## ") or line.startswith("# "):
                heading_lower = line.lstrip("#").strip().lower()
                intro_keywords = ("abstract", "introduction", "notation", "preliminaries", "setup")
                if not any(kw in heading_lower for kw in intro_keywords):
                    # Allow the intro section heading itself but stop at the next.
                    if not in_intro:
                        break
                    in_intro = False
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines)
        # Truncate to max_tokens.
        while _count_tokens(preamble) > max_tokens and "\n" in preamble:
            preamble = preamble[:preamble.rfind("\n")]
        return preamble.strip()

    def _split_by_sections(self, markdown: str) -> list[tuple[str, str]]:
        """
        Split markdown on ``##`` headings.

        Returns list of (heading, content) tuples. Sections matching
        _SKIP_SECTION_KEYWORDS are excluded.
        """
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in markdown.split("\n"):
            if line.startswith("## ") or (line.startswith("# ") and not line.startswith("## ")):
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                current_heading = line.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        # Filter skip sections.
        filtered = [
            (h, c) for h, c in sections
            if not any(kw in h.lower() for kw in _SKIP_SECTION_KEYWORDS)
        ]
        return filtered

    @staticmethod
    def _normalize_concept_title(title: str) -> str:
        """Lowercase, strip LaTeX delimiters and punctuation for deduplication."""
        t = title.lower()
        t = re.sub(r'\$[^$]*\$', '', t)         # strip inline math
        t = re.sub(r'\\\[.*?\\\]', '', t, flags=re.DOTALL)  # strip display math
        t = re.sub(r'[^\w\s]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    def _chunked_extract(
        self,
        markdown: str,
        hubs: dict[str, str],
        run_id: str,
        token_count: int,
    ) -> ExtractionResult:
        """
        Section-by-section extraction for papers over TOKEN_THRESHOLD_CHUNK tokens.

        Each section is extracted with the paper preamble prepended as shared
        context. Results are merged and deduplicated by normalised title.
        """
        preamble = self._extract_preamble(markdown)
        sections = self._split_by_sections(markdown)
        logger.info(
            "[%s] Chunked extraction: %d sections, preamble %d tokens.",
            run_id, len(sections), _count_tokens(preamble),
        )

        all_concepts: list[MathObject] = []
        seen_titles: set[str] = set()
        merged_one_liner = ""
        merged_themes: list[str] = []

        for heading, content in sections:
            section_tokens = _count_tokens(content)
            if section_tokens < 100:
                continue  # Skip nearly-empty sections.
            chunk = f"{preamble}\n\n{content}" if preamble else content
            try:
                result = self._extract_and_validate(chunk, hubs, run_id)
            except Exception:
                logger.warning(
                    "[%s] Chunked extraction failed for section '%s' — skipping.",
                    run_id, heading,
                )
                continue

            if not merged_one_liner and result.one_liner:
                merged_one_liner = result.one_liner
            for theme in result.active_themes:
                if theme not in merged_themes:
                    merged_themes.append(theme)

            for concept in result.extracted_concepts:
                norm = self._normalize_concept_title(concept.title)
                if norm and norm not in seen_titles:
                    seen_titles.add(norm)
                    all_concepts.append(concept)
                else:
                    logger.debug(
                        "[%s] Dedup: skipping duplicate concept '%s'.",
                        run_id, concept.title,
                    )

        logger.info(
            "[%s] Chunked extraction complete: %d unique concept(s).",
            run_id, len(all_concepts),
        )
        return ExtractionResult(
            one_liner=merged_one_liner,
            active_themes=merged_themes,
            extracted_concepts=all_concepts,
        )
    

    def _ensure_koofr_markdown_dir(self) -> None:
        """Create the markdown cache directory on Koofr if it does not exist."""
        try:
            if not self._webdav.check(self.koofr_markdown_dir):
                self._webdav.mkdir(self.koofr_markdown_dir)
                logger.info(
                    "Ingestion: created Koofr markdown dir: %s",
                    self.koofr_markdown_dir,
                )
            else:
                logger.debug(
                    "Ingestion: Koofr markdown dir exists: %s",
                    self.koofr_markdown_dir,
                )
        except Exception:
            logger.warning(
                "Ingestion: could not ensure Koofr markdown dir '%s' — "
                "markdown caching may fail.",
                self.koofr_markdown_dir,
                exc_info=True,
            )

    # -- OpenAI extraction (existing, untouched) --------------------------------

    def _extract_and_validate(
        self, markdown: str, hubs: dict[str, str], run_id: str
    ) -> ExtractionResult:
        """
        Call OpenAI, validate the response, attempt a repair pass if needed,
        and run latex_sanity_check on each concept.
        """
        result = self._call_openai(markdown, hubs)

        # if errors:
        #     logger.warning(
        #         "[%s] Pydantic validation failed (%d error(s)) -- attempting repair.",
        #         run_id,
        #         len(errors),
        #     )
        #     total_errors = len(errors)
        #     error_summary = "; ".join(errors[:MAX_REPAIR_ERRORS])
        #     if total_errors > MAX_REPAIR_ERRORS:
        #         error_summary += (
        #             f" ... (showing {MAX_REPAIR_ERRORS} of {total_errors} errors)"
        #         )
        #     raw2 = self._call_openai_repair(raw, error_summary)
        #     result, errors2 = validate_extraction(raw2)
        #     if errors2:
        #         logger.error(
        #             "[%s] Repair also failed -- flagging concepts with confidence=0.",
        #             run_id,
        #         )
        #         for concept in result.extracted_concepts:
        #             concept.confidence = 0.0

        for concept in result.extracted_concepts:
            issues = latex_sanity_check(concept.statement_latex)
            if issues:
                logger.warning(
                    "[%s] LaTeX issues in concept '%s': %s",
                    run_id,
                    concept.title,
                    issues,
                )
                concept.confidence = min(concept.confidence, 0.5)

        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _call_openai(self, markdown: str, hubs: dict[str, str]) -> ExtractionResult:
        """Send the paper Markdown to GPT and return the parsed JSON dict."""
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = EXTRACTION_SYSTEM_PROMPT.replace(
            "[INJECT_DYNAMIC_HUBS_HERE]", hub_names_str
        )
        # response = self.openai_client.chat.completions.create(
        #     model=OPENAI_MODEL,
        #     max_tokens=4096,
        #     response_format={"type": "json_object"},
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {
        #             "role": "user",
        #             "content": (
        #                 "Extract structured knowledge from the following "
        #                 " "
        #                 "INSTRUCTIONS"
        #                 "- Follow the schema strictly."
        #                 "- Prefer 3–12 high-value concepts."
        #                 "- Do not output theorem/lemma numbers as titles."
        #                 "- Do not include proof-only microlemmas."
        #                 " "
        #                 "PAPER MARKDOWN:\n\n"
        #                 f"{markdown[:100_000]}"
        #             ),
        #         },
        #     ],
        # )
        result = self.claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract structured knowledge from the following "
                        " "
                        "INSTRUCTIONS"
                        "- Follow the schema strictly."
                        "- Prefer 3–12 high-value concepts."
                        "- Do not output theorem/lemma numbers as titles."
                        "- Do not include proof-only microlemmas."
                        " "
                        "PAPER MARKDOWN:\n\n"
                        f"{markdown[:100_000]}"
                    ),
                },
            ],
            response_model=ExtractionResult,
        )
        logger.info("Claude extraction response received.")
        return result

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _call_openai_repair(
        self, invalid_output: dict[str, Any], error_summary: str
    ) -> dict[str, Any]:
        """Send the invalid output back to Claude with a repair instruction."""
        response = self.anthropic_raw.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=(
                "You are a JSON repair assistant. Fix the following JSON to match "
                "the required schema. Return only valid JSON, no explanation."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"The following JSON failed validation with these errors:\n"
                        f"{error_summary}\n\n"
                        f"Invalid JSON:\n{json.dumps(invalid_output, indent=2)}\n\n"
                        "Return the corrected JSON."
                    ),
                },
            ],
        )
        return json.loads(response.content[0].text)

    # -- Patch Paper Tracker row -----------------------------------------------

    def _patch_notion_page(
        self, page_id: str, result: ExtractionResult, run_id: str,
        set_thesis_relevance: bool = False,
    ) -> None:
        """
        Update the Paper Tracker page metadata after a successful extraction.

        Note: Status is NOT updated here; it is managed in _process_paper so
        that the full pipeline (Stages 1-3) completes before advancing.
        """
        properties: dict = {
            "AI Status": self.notion.select_prop("Unverified-AI"),
            "One Liner": {"rich_text": self.notion.rich_text(result.one_liner)},
            "Active Themes": self.notion.multi_select_prop(result.active_themes),
            "Extraction Version": {
                "rich_text": self.notion.rich_text(EXTRACTION_VERSION)
            },
            "Processed At": {
                "date": {"start": datetime.now(tz=timezone.utc).isoformat()}
            },
            "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
            "Last Error": {"rich_text": self.notion.rich_text("")},
        }

        self.notion.update_page(page_id=page_id, properties=properties)

    def _patch_notion_paper_post_linking(self, page_id: str, run_id: str) -> None:
        """
        Update the Paper Tracker page after the LINK stage completes.

        Status is NOT updated here; _process_paper sets s2-extracted after
        all stages complete successfully.
        """
        self.notion.update_page(
            page_id=page_id,
            properties={
                "Last Run ID": {"rich_text": self.notion.rich_text(run_id)},
            },
        )

    # -- Create Knowledge Inbox entry ------------------------------------------

    def _get_ki_schema(self) -> dict[str, str]:
        """
        Return a ``{property_name: property_type}`` mapping for the Knowledge
        Inbox DB, fetched once and cached for the lifetime of this engine
        instance.
        """
        if self._ki_schema is None:
            db = self.notion.get_database(self.knowledge_inbox_db)
            self._ki_schema = {
                k: v["type"] for k, v in db.get("properties", {}).items()
            }
            logger.debug("KI DB schema: %s", self._ki_schema)
        return self._ki_schema

    def _ki_prop(self, key: str, value: str) -> dict:
        """
        Build a Notion property value for a Knowledge Inbox field whose type
        may be either ``select`` or ``status`` depending on the live schema.

        Using ``select_prop`` for a ``status``-typed field (or vice-versa)
        causes Notion to return a 400 validation error, so we always consult
        the cached schema rather than hardcoding the type.
        """
        prop_type = self._get_ki_schema().get(key, "select")
        if prop_type == "status":
            return self.notion.status_prop(value)
        return self.notion.select_prop(value)

    def _create_knowledge_item(
        self,
        paper_page_id: str,
        concept: MathObject,
        hubs: dict[str, str],
    ) -> str:
        """
        Materialise a single MathObject as a Knowledge Inbox Notion page.

        Sets graph_link_status = "unlinked" at creation.
        Stage 3 (_update_knowledge_item_graph_data) later writes edges and
        promotes to "linked-ai".

        Returns the Notion page ID of the created page.
        """
        kind = concept.type
        title = concept.title

        source_pages_str = (
            ", ".join(str(p) for p in concept.source_pages)
            if concept.source_pages
            else ""
        )
        title_key = self.notion.get_title_property_name(self.knowledge_inbox_db)
        properties: dict = {
            title_key: self.notion.title_prop(f"{title}"),
            "Type": self.notion.select_prop(kind),
            "Status": self._ki_prop("Status", "Inbox"),
            "verification_status": self._ki_prop("verification_status", "unverified"),
            "Graph Link Status": self._ki_prop("Graph Link Status", "unlinked"),
            "Source Paper": self.notion.relation_prop([paper_page_id]),
        }

        if source_pages_str:
            properties["Source Pages"] = {
                "rich_text": self.notion.rich_text(source_pages_str)
            }
        if concept.suggested_hub:
             properties["Suggested Hub"] = {"rich_text": self.notion.rich_text(concept.suggested_hub)}

        properties["AI Confidence"] = {"number": concept.confidence}

        if concept.canonical_keywords:
            properties["Keywords"] = self.notion.multi_select_prop(
                concept.canonical_keywords
            )
        if concept.prereq_keywords:
            properties["Prereq Keywords"] = self.notion.multi_select_prop(
                concept.prereq_keywords
            )
        if concept.downstream_keywords:
            properties["Downstream Keywords"] = self.notion.multi_select_prop(
                concept.downstream_keywords
            )
        if concept.source_anchors:
            properties["Source Anchors"] = {
                "rich_text": self.notion.rich_text(concept.source_anchors)
            }
        if concept.interpretation:
            properties["Interpretation"] = {
                "rich_text": self.notion.rich_text(concept.interpretation)
            }
        if concept.proof_idea:
            properties["Proof Idea"] = {
                "rich_text": self.notion.rich_text(concept.proof_idea)
            }
        if concept.aliases:
            properties["Aliases"] = {
                "rich_text": self.notion.rich_text(concept.aliases)
            }
        if concept.assumptions:
            properties["Assumptions"] = {
                "rich_text": self.notion.rich_text(concept.assumptions[:2000])
            }
        if concept.statement_latex:
            properties["Statement LaTeX"] = {
                "rich_text": self.notion.rich_text(concept.statement_latex[:2000])
            }
        if concept.source_quotes:
            properties["Source Quote"] = {
                "rich_text": self.notion.rich_text(concept.source_quotes)
            }
        if concept.named_tools:
            properties["Named Tools"] = self.notion.multi_select_prop(concept.named_tools)
        if concept.setting:
            properties["Setting"] = self.notion.multi_select_prop(concept.setting)
        if concept.result_category:
            properties["Result Category"] = self.notion.select_prop(
                concept.result_category
            )

        new_page = self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties=properties,
        )
        new_page_id: str = new_page["id"]
        logger.info(
            "Created Knowledge Inbox page %s for concept '%s'.", new_page_id, title
        )

        body_blocks: list[dict] = []
        # REQ-8: Prepend review checklist so human sees guided review flow first.
        body_blocks.extend(self._review_checklist_blocks())
        body_blocks.append(self._heading_block("Assumptions"))
        body_blocks.extend(paragraph_blocks_from_latex(concept.assumptions))
        body_blocks.append(self._heading_block("Statement"))
        body_blocks.extend(paragraph_blocks_from_latex(sanitize_statement_latex(concept.statement_latex)))
        if concept.variables:
            body_blocks.append(self._heading_block("Variables"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.variables))
        if concept.conclusion:
            body_blocks.append(self._heading_block("Conclusion"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.conclusion))
        if concept.source_quotes:
            body_blocks.append(self._heading_block("Source Quote"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.source_quotes))
        if concept.interpretation:
            body_blocks.append(self._heading_block("Interpretation"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.interpretation))
        if concept.proof_idea:
            body_blocks.append(self._heading_block("Proof Idea"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.proof_idea))

        self._append_blocks_in_batches(new_page_id, body_blocks)
        return new_page_id

    # -- Stage 3: write edge data to Knowledge Inbox page ----------------------

    def _update_knowledge_item_graph_data(
        self,
        ki_page_id: str,
        link_result: ConceptLinkResult | CrossPaperLinkResult,
    ) -> None:
        """
        Write edge results to the KI page property and append the 3-tier
        cross-paper edge section to the page body.

        Handles both the legacy ConceptLinkResult (intra-paper / TF-IDF path)
        and the new CrossPaperLinkResult (cross-paper Qdrant path).

        Sets graph_link_status = "linked-ai" only when at least one edge is
        produced.
        """
        if isinstance(link_result, CrossPaperLinkResult):
            self._update_ki_cross_paper(ki_page_id, link_result)
        else:
            self._update_ki_legacy(ki_page_id, link_result)

    def _update_ki_legacy(
        self, ki_page_id: str, link_result: ConceptLinkResult
    ) -> None:
        """Write legacy ConceptLinkResult edges (old format) to KI page."""
        edge_dict = link_result.model_dump(exclude_none=True)
        edge_dict = {k: v for k, v in edge_dict.items() if v}
        if not edge_dict:
            logger.debug(
                "KI page %s: no edges produced -- remaining 'unlinked'.", ki_page_id
            )
            return
        payload = edge_dict
        s = json.dumps(payload, ensure_ascii=False)
        if len(s) > NOTION_BLOCK_MAX_CHARS:
            for rel in ["related", "enables", "depends_on", "generalizes", "special_case_of"]:
                while payload.get(rel) and len(json.dumps(payload, ensure_ascii=False)) > NOTION_BLOCK_MAX_CHARS:
                    payload[rel].pop()
        edge_json = json.dumps(payload, ensure_ascii=False)
        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "Edge Suggestions": {
                    "rich_text": self.notion.rich_text(edge_json)
                },
                "Graph Link Status": self._ki_prop("Graph Link Status", "linked-ai"),
            },
        )

    def _update_ki_cross_paper(
        self, ki_page_id: str, link_result: CrossPaperLinkResult
    ) -> None:
        """
        Write CrossPaperLinkResult to the KI page:

        1. Serialise proposals (confidence >= EDGE_REVIEW_FLAG_CONFIDENCE) as
           JSON in the 'Edge Suggestions' property for PromotionEngine.
        2. Append the 3-tier '## Proposed Cross-Paper Edges' section to the
           page body.
        """
        all_proposals = link_result.proposals
        low_conf = link_result.low_confidence_suggestions

        if not all_proposals and not low_conf:
            logger.debug(
                "KI page %s: no cross-paper edges produced -- remaining 'unlinked'.",
                ki_page_id,
            )
            return

        # -- Write Edge Suggestions property (for PromotionEngine) ------------
        payload = {
            "proposals": [p.model_dump() for p in all_proposals],
        }
        edge_json = json.dumps(payload, ensure_ascii=False)
        # Truncate if necessary (Notion 2000-char limit per rich_text segment).
        if len(edge_json) > NOTION_BLOCK_MAX_CHARS:
            while all_proposals and len(
                json.dumps({"proposals": [p.model_dump() for p in all_proposals]},
                           ensure_ascii=False)
            ) > NOTION_BLOCK_MAX_CHARS:
                all_proposals.pop()
            edge_json = json.dumps(
                {"proposals": [p.model_dump() for p in all_proposals]},
                ensure_ascii=False,
            )

        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "Edge Suggestions": {
                    "rich_text": self.notion.rich_text(edge_json)
                },
                "Graph Link Status": self._ki_prop("Graph Link Status", "linked-ai"),
            },
        )

        # -- Append 3-tier edge section to page body --------------------------
        edge_blocks = self._render_cross_paper_edges_blocks(
            proposals=all_proposals,
            low_confidence_suggestions=low_conf,
        )
        if edge_blocks:
            try:
                self._append_blocks_in_batches(ki_page_id, edge_blocks)
            except Exception:
                logger.warning(
                    "KI page %s: failed to append cross-paper edge blocks — "
                    "edge JSON is still stored in Edge Suggestions property.",
                    ki_page_id,
                )

    # -- Stage 2: candidate retrieval (dispatcher) ----------------------------

    def hydrate_candidates(
        self,
        candidate_ids: list[str],
    ) -> Dict[str, ConceptData]:
        """
        Fetch full Notion page data for a list of concept page IDs concurrently.

        Uses a ThreadPoolExecutor with a semaphore of NOTION_HYDRATION_CONCURRENCY
        to avoid rate-limiting the Notion API.

        Returns a dict mapping notion_page_id → ConceptData.  Any page that
        cannot be fetched or parsed is silently omitted (missing fields default
        to empty string / empty list, not an exception).
        """
        if not candidate_ids:
            return {}

        def _fetch_one(page_id: str) -> tuple[str, ConceptData | None]:
            try:
                page = self.notion.get_page(page_id)
                props = page.get("properties", {})

                def _text(key: str) -> str:
                    try:
                        segs = props[key]["rich_text"]
                        return "".join(s.get("plain_text", "") for s in segs)
                    except (KeyError, TypeError):
                        return ""

                def _select(key: str) -> str:
                    try:
                        return props[key]["select"]["name"] or ""
                    except (KeyError, TypeError):
                        return ""

                def _multi(key: str) -> list[str]:
                    try:
                        return [o["name"] for o in props[key]["multi_select"]]
                    except (KeyError, TypeError):
                        return []

                # Title — try "Name" (SB) then the first title-type property (KI).
                title = ""
                try:
                    title = props["Name"]["title"][0]["plain_text"] or ""
                except (KeyError, IndexError, TypeError):
                    pass
                if not title:
                    for v in props.values():
                        if v.get("type") == "title":
                            try:
                                title = v["title"][0]["plain_text"] or ""
                                break
                            except (KeyError, IndexError, TypeError):
                                pass

                import re as _re
                title = _re.sub(r"^\[[^\]]+\]\s*", "", title).strip()

                concept_data = ConceptData(
                    notion_page_id=page_id,
                    title=title or "(unknown)",
                    concept_type=_select("Type") or _select("Concept Type") or "Definition",
                    statement_latex=_text("Statement LaTeX"),
                    assumptions=_text("Assumptions"),
                    conclusion=_text("Conclusion") or _text("Interpretation"),
                    setting=_multi("Setting"),
                    named_tools=_multi("Named Tools"),
                    keywords=_multi("Keywords"),
                )
                return page_id, concept_data
            except Exception:
                logger.debug(
                    "hydrate_candidates: failed to fetch page %s — skipping.",
                    page_id,
                    exc_info=True,
                )
                return page_id, None

        results: Dict[str, ConceptData] = {}
        max_workers = min(NOTION_HYDRATION_CONCURRENCY, len(candidate_ids))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, pid): pid for pid in candidate_ids}
            for future in concurrent.futures.as_completed(futures):
                pid, data = future.result()
                if data is not None:
                    results[pid] = data

        return results

    def _retrieve_candidates_for_concept(
        self,
        concept: MathObject,
        sb_index: list[dict],
        k: int = RETRIEVE_CANDIDATES_K,
        current_page_id: str | None = None,
        same_paper_ids: set | None = None,
    ) -> list[dict]:
        """
        Return top-k candidate concepts for linking.

        When VECTOR_INDEX_ENABLED is set and Qdrant is reachable, delegates to
        VectorIndexEngine.retrieve_candidates (semantic ANN search) and then
        applies cross-paper pre-filter scoring and reranking.

        Pre-filter scoring is ONLY applied to cross-paper candidates (concepts
        whose Notion page ID is not in ``same_paper_ids``).  Same-paper
        candidates bypass the filter and pass through unchanged.

        Falls back to TF-IDF token-overlap scoring when Qdrant is unavailable.

        ``current_page_id`` — the concept's own KI page ID; excluded from
        results to prevent self-links.
        ``same_paper_ids``  — KI page IDs of other concepts from the same paper;
        these bypass the pre-filter and use existing logic.
        """
        if not (self._vector_index and self._vector_index.available):
            return self._tfidf_retrieve(concept, sb_index, k)

        # ── Qdrant path ───────────────────────────────────────────────────────
        hints = self._vector_index.retrieve_candidates(concept, verified_only=False)
        # Exclude self.
        if current_page_id:
            hints = [h for h in hints if h.notion_page_id != current_page_id]

        same_paper_ids = same_paper_ids or set()

        # Separate same-paper and cross-paper candidates.
        same_paper_hints = [h for h in hints if h.notion_page_id in same_paper_ids]
        cross_paper_hints = [h for h in hints if h.notion_page_id not in same_paper_ids]

        # ── Apply pre-filter scoring to cross-paper candidates ────────────────
        cross_paper_dicts: list[dict] = []
        if cross_paper_hints:
            cross_ids = [h.notion_page_id for h in cross_paper_hints]
            logger.debug(
                "Pre-filter: hydrating %d cross-paper candidate(s) for '%s'.",
                len(cross_ids), concept.title,
            )
            hydrated = self.hydrate_candidates(cross_ids)

            # Build ConceptData for C_A from the MathObject.
            concept_a_data = ConceptData(
                notion_page_id=current_page_id or "",
                title=concept.title,
                concept_type=concept.type,
                statement_latex=concept.statement_latex,
                assumptions=concept.assumptions or "",
                conclusion=concept.conclusion or "",
                setting=list(concept.setting) if concept.setting else [],
                named_tools=list(concept.named_tools) if concept.named_tools else [],
                keywords=list(concept.canonical_keywords) if concept.canonical_keywords else [],
            )

            scored: list[tuple[float, dict]] = []
            n_before = len(cross_paper_hints)
            dropped = 0

            for hint in cross_paper_hints:
                concept_b_data = hydrated.get(hint.notion_page_id)
                if concept_b_data is None:
                    # Could not hydrate — include with raw Qdrant similarity.
                    d = hint.to_dict()
                    d["_pre_filter_signal"] = "none"
                    scored.append((hint.score, d))
                    continue

                score = score_candidate_pair(concept_a_data, concept_b_data, hint.score)

                if score.should_drop:
                    dropped += 1
                    continue

                d = hint.to_dict()
                # Attach scoring metadata for use in the GPT prompt.
                d["_concept_data"] = concept_b_data
                d["_pre_filter_signal"] = _dominant_signal(score)
                d["_score_obj"] = score
                scored.append((score.composite_score, d))

            logger.debug(
                "Pre-filter '%s': %d → %d candidate(s) (%d dropped).",
                concept.title, n_before, len(scored), dropped,
            )

            # Sort descending by composite score, cap at EDGE_MAX_CANDIDATES_TO_GPT.
            scored.sort(key=lambda x: x[0], reverse=True)
            cross_paper_dicts = [d for _, d in scored[:EDGE_MAX_CANDIDATES_TO_GPT]]

        # ── Combine same-paper and (reranked) cross-paper candidates ──────────
        same_paper_dicts = [h.to_dict() for h in same_paper_hints]
        return same_paper_dicts + cross_paper_dicts

    def _tfidf_retrieve(
        self,
        concept: MathObject,
        sb_index: list[dict],
        k: int = RETRIEVE_CANDIDATES_K,
    ) -> list[dict]:
        """
        Fallback Stage 2: TF-IDF-style token overlap with hub-affinity bonus.

        Score = |concept_tokens ∩ r.keywords_bag| / log(1 + |bag|)
                + 0.2 if r.hub == concept.suggested_hub
        """
        def _toks(s: str) -> set:
            return {t.lower().strip() for t in re.split(r"[\s\-,;]+", s) if t.strip()}

        concept_tokens: set = set()
        for kw_list in (
            concept.canonical_keywords,
            concept.prereq_keywords,
            concept.downstream_keywords,
        ):
            for kw in kw_list:
                concept_tokens |= _toks(kw)
        concept_tokens |= _toks(concept.title)

        scored: list[tuple[float, dict]] = []
        for record in sb_index:
            bag = record.get("keywords_bag", set())
            overlap = len(concept_tokens & bag)
            score = overlap / math.log(1.0 + len(bag))
            if record.get("hub") and record["hub"] == concept.suggested_hub:
                score += 0.2
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, r in scored[:k]:
            out.append({
                "id": r["id"],
                "title": r["title"],
                "hub": r.get("hub", ""),
                "summary": r.get("summary", ""),
                "score": round(float(score), 4),
            })
        return out

    def _update_knowledge_item_candidates(
        self, ki_page_id: str, candidates: list[dict]
    ) -> None:
        """Write the Stage 2 candidate list to the KI 'Candidate Matches' property.

        Private keys (prefixed with '_') attached by the pre-filter scorer are
        stripped before serialisation — they hold non-JSON-serialisable objects.
        """
        # Strip private metadata (ConceptData objects, CandidateScore objects).
        slim = [
            {k: v for k, v in c.items() if not k.startswith("_")}
            for c in candidates
        ]
        s = json.dumps(slim, ensure_ascii=False)
        while len(s) > NOTION_BLOCK_MAX_CHARS and len(slim) > 1:
            slim.pop()
            s = json.dumps(slim, ensure_ascii=False)
        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "Candidate Matches": {"rich_text": self.notion.rich_text(s)}
            },
        )

    # -- Stage 3: LLM linking --------------------------------------------------

    def _run_stage_link(
        self,
        concept: MathObject,
        candidates: list[dict],
        run_id: str,
    ) -> ConceptLinkResult | CrossPaperLinkResult:
        """
        Call the LLM linking prompt and return edge proposals.

        When candidates include pre-filter metadata (Qdrant path), the new
        enriched prompt (v2) is used and a CrossPaperLinkResult is returned.
        Otherwise the legacy prompt is used and a ConceptLinkResult is returned.

        Returns an empty ConceptLinkResult on failure or when there are no
        candidates.
        """
        if not candidates:
            return ConceptLinkResult()

        # Check whether any cross-paper candidates have been hydrated.
        has_cross_paper = any(c.get("_concept_data") is not None for c in candidates)

        try:
            if has_cross_paper:
                result = self._call_claude_link_v2(concept, candidates)
            else:
                result = self._call_openai_link(concept, candidates)
        except Exception:
            logger.warning(
                "[%s] LLM linking failed for '%s'.", run_id, concept.title
            )
            return ConceptLinkResult()
        return result

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=10, max=120))
    def _call_openai_link(
        self, concept: MathObject, candidates: list[dict]
    ) -> ConceptLinkResult:
        """Invoke the Stage 3 linking prompt (legacy v1) via Claude."""
        candidate_lines = "\n".join(
            f"{i + 1}. [id:{r.get('id', '')}] {r['title']}"
            + (f" [{r['hub']}]" if r.get("hub") else "")
            + (f" (suggested relation: {r['edge_type_hint']})" if r.get("edge_type_hint") else "")
            + (f" — {r['summary']}" if r.get("summary") else "")
            for i, r in enumerate(candidates)
        )
        concept_summary = (
            f"Title: {concept.title}\n"
            f"Type: {concept.type}\n"
            f"Conclusion: {concept.conclusion or '(none)'}\n"
            f"Keywords: {', '.join(concept.canonical_keywords)}\n"
            f"Prereq keywords: {', '.join(concept.prereq_keywords)}\n"
            f"Downstream keywords: {', '.join(concept.downstream_keywords)}"
        )
        user_message = (
            f"CONCEPT:\n{concept_summary}\n\n"
            f"CANDIDATES:\n{candidate_lines}\n\n"
            "Identify relationships. Return JSON only."
        )
        result = self.claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=LINKING_SYSTEM_PROMPT_V1,
            messages=[
                {"role": "user", "content": user_message},
            ],
            response_model=ConceptLinkResult,
        )
        return result

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=10, max=120))
    def _call_claude_link_v2(
        self, concept: MathObject, candidates: list[dict]
    ) -> CrossPaperLinkResult:
        """
        Invoke the enriched Stage 3 cross-paper edge confirmation prompt (v2).

        Builds a detailed user message with full concept fields for C_A and
        every candidate, including pre-filter signal annotations.  Parses the
        response into CrossPaperLinkResult, applying the review-flag routing
        logic to each EdgeProposal.

        On parse failure, logs the raw response and returns an empty
        CrossPaperLinkResult (pipeline continues in degraded mode).
        """
        # ── Build user message ────────────────────────────────────────────────
        _MAX_LATEX = 800

        def _fmt_latex(s: str) -> str:
            if not s:
                return "(not recorded)"
            if len(s) > _MAX_LATEX:
                return s[:_MAX_LATEX] + " [truncated]"
            return s

        def _fmt_field(s: str) -> str:
            return s if s else "(not recorded)"

        # Source concept (C_A)
        src_lines = [
            "## Source Concept (C_A)",
            "",
            f"Title: {concept.title}",
            f"Type: {concept.type}",
            f"Setting: {_fmt_field(', '.join(concept.setting) if concept.setting else '')}",
            f"Statement (LaTeX): {_fmt_latex(concept.statement_latex)}",
            f"Assumptions: {_fmt_field(concept.assumptions)}",
            f"Conclusion: {_fmt_field(concept.conclusion)}",
            f"Named Tools: {', '.join(concept.named_tools) if concept.named_tools else 'none'}",
            f"Keywords: {', '.join(concept.canonical_keywords)}",
            "",
            "---",
            "",
            "## Target Concepts",
            "",
        ]

        for i, cand in enumerate(candidates, start=1):
            cd: ConceptData | None = cand.get("_concept_data")
            sig = cand.get("_pre_filter_signal", "")
            score_obj: CandidateScore | None = cand.get("_score_obj")

            if cd is not None:
                # Cross-paper candidate — full context available.
                setting_str = (
                    ", ".join(cd.setting)
                    if isinstance(cd.setting, list)
                    else str(cd.setting or "")
                )
                named_tools_str = (
                    ", ".join(cd.named_tools) if cd.named_tools else "none"
                )
                keywords_str = (
                    ", ".join(cd.keywords) if cd.keywords else "none"
                )
                signal_desc = ""
                if score_obj is not None:
                    parts = []
                    if score_obj.named_tool_match:
                        parts.append("named_tool_match=True")
                    if score_obj.assumption_conclusion_overlap > 0:
                        parts.append(
                            f"assumption_conclusion_overlap="
                            f"{score_obj.assumption_conclusion_overlap:.2f}"
                        )
                    if score_obj.setting_containment:
                        parts.append(f"setting_containment={score_obj.setting_containment}")
                    if score_obj.keyword_jaccard > 0:
                        parts.append(f"keyword_jaccard={score_obj.keyword_jaccard:.2f}")
                    signal_desc = ", ".join(parts) if parts else "no signals fired"

                target_lines = [
                    f"### Target {i}: {cd.title}",
                    f"Notion Page ID: {cd.notion_page_id}",
                    f"Type: {cd.concept_type}",
                    f"Setting: {_fmt_field(setting_str)}",
                    f"Statement (LaTeX): {_fmt_latex(cd.statement_latex)}",
                    f"Assumptions: {_fmt_field(cd.assumptions)}",
                    f"Conclusion: {_fmt_field(cd.conclusion)}",
                    f"Named Tools: {named_tools_str}",
                    f"Keywords: {keywords_str}",
                ]
                if signal_desc:
                    target_lines.append(f"[Pre-filter signals: {signal_desc}]")
            else:
                # Same-paper candidate or un-hydrated candidate — basic info.
                target_lines = [
                    f"### Target {i}: {cand.get('title', '(unknown)')}",
                    f"Notion Page ID: {cand.get('id', '')}",
                    f"Type: {cand.get('concept_type', '')}",
                    f"Setting: (not recorded)",
                    f"Statement (LaTeX): (not recorded)",
                    f"Assumptions: (not recorded)",
                    f"Conclusion: (not recorded)",
                    f"Named Tools: none",
                    f"Keywords: {cand.get('edge_type_hint', 'none')}",
                ]

            src_lines.extend(target_lines)
            src_lines.append("")

        src_lines += [
            "---",
            "",
            "Return a JSON object with a 'proposals' key containing a list of "
            "EdgeProposal objects. Return {\"proposals\": []} if no relationships "
            "are warranted. Do not include any text outside the JSON.",
        ]

        user_message = "\n".join(src_lines)

        # ── Claude call with structured output ────────────────────────────────
        try:
            result: CrossPaperLinkResult = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=LINKING_SYSTEM_PROMPT_V2,
                messages=[{"role": "user", "content": user_message}],
                response_model=CrossPaperLinkResult,
            )
        except Exception:
            logger.warning(
                "_call_claude_link_v2: Claude call failed for '%s'.", concept.title,
                exc_info=True,
            )
            return CrossPaperLinkResult()

        # ── Build a lookup from candidate notion_page_id → CandidateScore ────
        scores_by_id: Dict[str, CandidateScore] = {}
        for cand in candidates:
            score_obj = cand.get("_score_obj")
            if score_obj is not None:
                scores_by_id[cand.get("id", "")] = score_obj

        # ── Apply post-call routing: confidence tiers, pre_filter_signal ──────
        proposals: list[EdgeProposal] = []
        low_confidence: list[EdgeProposal] = []

        for proposal in result.proposals:
            pid = proposal.target_notion_page_id

            # Look up the dominant signal for this candidate.
            # Fall back to scanning candidates by title if ID not in scores_by_id.
            score_obj = scores_by_id.get(pid)
            if score_obj is None:
                for cand in candidates:
                    if cand.get("_concept_data") and cand["_concept_data"].notion_page_id == pid:
                        score_obj = cand.get("_score_obj")
                        break

            if score_obj is not None:
                proposal.pre_filter_signal = _dominant_signal(score_obj)
            else:
                # No score available (same-paper or un-hydrated).
                proposal.pre_filter_signal = "none"
                # Provide a neutral CandidateScore for review-flag logic.
                score_obj = CandidateScore(
                    candidate_id=pid,
                    qdrant_similarity=0.0,
                    named_tool_match=False,
                    assumption_conclusion_overlap=0.0,
                    setting_containment=None,
                    keyword_jaccard=0.0,
                    composite_score=0.0,
                    should_drop=False,
                )

            if proposal.confidence < EDGE_REVIEW_FLAG_CONFIDENCE:
                # Below creation threshold — informational hint only.
                low_confidence.append(proposal)
            else:
                _assign_review_flag(proposal, score_obj)
                proposals.append(proposal)

        return CrossPaperLinkResult(
            proposals=proposals,
            low_confidence_suggestions=low_confidence,
        )

    # -- Re-extraction flow (REQ-5) --------------------------------------------

    def _reextract_missed_concepts(
        self,
        page: dict,
        hubs: dict[str, str],
        sb_index: list[dict],
    ) -> None:
        """
        Targeted second-pass extraction for concepts flagged as missing by the
        human reviewer via the 'Re-extract Hints' paper property.

        Steps:
        1. Read Re-extract Hints from paper page.
        2. Fetch existing KI concept titles for this paper (dedup guard).
        3. Run GPT with REEXTRACT_SYSTEM_PROMPT.
        4. Create KI pages for new concepts only.
        5. Run Stage 2 + Stage 3 on new concepts.
        6. Advance paper back to s2-extracted.
        """
        page_id = page["id"]
        props = page["properties"]
        run_id = uuid.uuid4().hex[:8]

        hints = self._get_text_prop(props, "Re-extract Hints").strip()
        if not hints:
            logger.warning(
                "[%s] s2-reextract: 'Re-extract Hints' is empty for page %s — "
                "reverting to s2-extracted.",
                run_id, page_id,
            )
            self.notion.update_page(
                page_id=page_id,
                properties={"Status": self.notion.status_prop("s2-extracted")},
            )
            return

        # Fetch existing KI concept titles to avoid duplicates.
        existing_ki = self.notion.query_database(
            self.knowledge_inbox_db,
            filter={
                "property": "Source Paper",
                "relation": {"contains": page_id},
            },
        )
        existing_titles = [self._get_page_title(p) for p in existing_ki if self._get_page_title(p)]
        existing_titles_str = "\n".join(f"- {t}" for t in existing_titles) or "(none)"

        # Build the targeted extraction prompt.
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = (
            REEXTRACT_SYSTEM_PROMPT
            .replace("{hints}", hints)
            .replace("{existing_titles}", existing_titles_str)
            .replace("{latex_formatting_rules}", LATEX_FORMATTING_RULES)
        )
        # Append hub list so GPT can assign suggested_hub.
        system_prompt += f"\n\nALLOWED_HUBS:\n[{hub_names_str}]"

        # Fetch the paper's markdown from the Notion page body as context.
        # Fall back to using hints alone if markdown is unavailable.
        markdown_context = hints  # minimal context fallback

        logger.info(
            "[%s] Re-extraction: %d hint(s), %d existing concept(s).",
            run_id, len(hints.split("\n")), len(existing_titles),
        )

        try:
            reextraction = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract ONLY the missing concepts described above.\n\n"
                            f"PAPER CONTEXT:\n{markdown_context[:20_000]}"
                        ),
                    },
                ],
                response_model=ExtractionResult,
            )
        except Exception:
            logger.exception("[%s] Re-extraction Claude call failed.", run_id)
            return

        new_concepts = [
            c for c in reextraction.extracted_concepts
            if self._normalize_concept_title(c.title)
            not in {self._normalize_concept_title(t) for t in existing_titles}
        ]

        logger.info(
            "[%s] Re-extraction: %d new concept(s) after dedup.",
            run_id, len(new_concepts),
        )

        if not new_concepts:
            logger.info("[%s] Re-extraction: nothing new — advancing to s2-extracted.", run_id)
            self.notion.update_page(
                page_id=page_id,
                properties={"Status": self.notion.status_prop("s2-extracted")},
            )
            return

        # Create KI pages for new concepts.
        ki_pages: list[tuple[MathObject, str]] = []
        for concept in new_concepts:
            try:
                ki_page_id = self._create_knowledge_item(page_id, concept, hubs)
                ki_pages.append((concept, ki_page_id))
                if self._vector_index and self._vector_index.available:
                    try:
                        self._vector_index.index_concept(concept, ki_page_id, verified=False)
                    except Exception:
                        logger.warning(
                            "[%s] VectorIndex: failed to index '%s'.", run_id, concept.title
                        )
            except Exception:
                logger.exception(
                    "[%s] Re-extraction: failed to create KI item '%s'",
                    run_id, concept.title,
                )

        self._inject_ki_pages_into_index(ki_pages, sb_index)

        # Stage 2 + 3 on new concepts.
        concept_candidates: list[tuple[MathObject, str, list[dict]]] = []
        for concept, ki_page_id in ki_pages:
            candidates = self._retrieve_candidates_for_concept(
                concept, sb_index, current_page_id=ki_page_id
            )
            self._update_knowledge_item_candidates(ki_page_id, candidates)
            concept_candidates.append((concept, ki_page_id, candidates))

        for concept, ki_page_id, candidates in concept_candidates:
            try:
                link_result = self._run_stage_link(concept, candidates, run_id)
                self._update_knowledge_item_graph_data(ki_page_id, link_result)
            except Exception:
                logger.exception(
                    "[%s] Re-extraction: link stage failed for '%s'",
                    run_id, concept.title,
                )

        # Return paper to s2-extracted.
        self.notion.update_page(
            page_id=page_id,
            properties={"Status": self.notion.status_prop("s2-extracted")},
        )
        logger.info(
            "[%s] Re-extraction complete: %d new concept(s) created.",
            run_id, len(ki_pages),
        )

    # -- Text chunking ---------------------------------------------------------

    def _chunk_text(self, text: str, max_len: int = NOTION_BLOCK_MAX_CHARS) -> list[str]:
        """Split text into chunks of at most max_len chars, preferring newlines."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break
            split_pos = remaining.rfind("\n", 0, max_len)
            if split_pos <= 0:
                split_pos = max_len
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos:]
            else:
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos + 1:]
        return [c for c in chunks if c.strip()]

    # -- Notion block builders -------------------------------------------------

    def _paragraph_blocks(self, text: str) -> list[dict]:
        """Convert a long string into a list of Notion paragraph block dicts."""
        return [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": chunk}}],
                },
            }
            for chunk in self._chunk_text(text)
        ]

    @staticmethod
    def _heading_block(text: str) -> dict:
        """Build a Notion heading_2 block for a section title."""
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
            },
        }

    @staticmethod
    def _todo_block(text: str) -> dict:
        """Build a Notion to_do block (checkbox item)."""
        return {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}],
                "checked": False,
            },
        }

    @staticmethod
    def _divider_block() -> dict:
        """Build a Notion divider block."""
        return {"object": "block", "type": "divider", "divider": {}}

    # -- Cross-paper edge rendering (Part 3) -----------------------------------
    #
    # Human review workflow for flagged edges:
    #
    #   ✅ Auto-Created (needs_review=False):
    #      The edge already exists in Edges DB.  No action required.
    #      The callout is informational — it shows WHY the edge was created.
    #
    #   ⚠️  Flagged — Requires Your Decision (needs_review=True):
    #      The edge EXISTS in Edges DB but is flagged for human confirmation.
    #      - Accept: open the edge page in Edges DB, uncheck 'needs_review'.
    #      - Reject: delete the edge page from Edges DB.
    #      The checkbox in Notion is a VISUAL PROMPT only — checking it does
    #      NOT automatically update the Edges DB.
    #
    #   💡 Low-Confidence Suggestions (confidence < EDGE_REVIEW_FLAG_CONFIDENCE):
    #      These edges were NOT written to Edges DB.  They are similarity hints
    #      only.  If you want to create one, do so manually in the Edges DB.

    def _render_cross_paper_edges_blocks(
        self,
        proposals: list[EdgeProposal],
        low_confidence_suggestions: list[EdgeProposal],
    ) -> list[dict]:
        """
        Build Notion block children for the '## Proposed Cross-Paper Edges'
        section appended to every KI page that has cross-paper edge proposals.

        Three subsections:
          1. Auto-Created Edges (needs_review=False)  — green callout blocks
          2. Flagged Edges (needs_review=True)         — to-do blocks
          3. Low-Confidence Suggestions               — bulleted list (gray)
        """
        if not proposals and not low_confidence_suggestions:
            return []

        blocks: list[dict] = [
            self._divider_block(),
            self._heading_block("Proposed Cross-Paper Edges"),
        ]

        # ── Sub-section 1: Auto-Created ────────────────────────────────────────
        auto_created = [p for p in proposals if not p.needs_review]
        if auto_created:
            blocks.append(self._heading_block("Auto-Created Edges"))
            for p in auto_created:
                text = (
                    f"✅ {p.relation_type} → {p.target_concept_title}\n"
                    f"   Confidence: {p.confidence:.0%}\n"
                    f"   Why: {p.justification}\n"
                    f"   Fields: {', '.join(p.driving_fields)}\n"
                    f"   Signal: {p.pre_filter_signal or 'none'}"
                )
                blocks.append({
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [
                            {"type": "text", "text": {"content": text[:2000]}}
                        ],
                        "icon": {"type": "emoji", "emoji": "✅"},
                        "color": "green_background",
                    },
                })

        # ── Sub-section 2: Flagged — Requires Decision ─────────────────────────
        flagged = [p for p in proposals if p.needs_review]
        if flagged:
            blocks.append(self._heading_block("Flagged Edges — Requires Your Decision"))
            for p in flagged:
                text = (
                    f"⚠️ {p.relation_type} → {p.target_concept_title}  "
                    f"Confidence: {p.confidence:.0%}\n"
                    f"   Why: {p.justification}\n"
                    f"   Fields: {', '.join(p.driving_fields)}\n"
                    f"   Signal: {p.pre_filter_signal or 'none'}\n"
                    f"   → Accept: open edge in Edges DB, uncheck needs_review. "
                    f"Reject: delete edge from Edges DB."
                )
                blocks.append(self._todo_block(text))

        # ── Sub-section 3: Low-Confidence Suggestions ─────────────────────────
        if low_confidence_suggestions:
            blocks.append(self._heading_block("Low-Confidence Suggestions"))
            for p in low_confidence_suggestions:
                text = (
                    f"💡 {p.relation_type} → {p.target_concept_title}  "
                    f"Confidence: {p.confidence:.0%}  [NOT auto-created — similarity hint only]\n"
                    f"   Why: {p.justification}"
                )
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": text[:2000]},
                                "annotations": {"color": "gray"},
                            }
                        ]
                    },
                })

        return blocks

    # -- Review checklist (REQ-8) ----------------------------------------------

    def _review_checklist_blocks(self) -> list[dict]:
        """
        Return the guided review checklist blocks prepended to every KI page.
        """
        return [
            self._heading_block("Review"),
            self._todo_block(
                "1. Is the title correct? "
                "(edit Name, or fill Corrected Title property)"
            ),
            self._todo_block(
                "2. Is the formal statement correct? Check the Statement block below."
            ),
            self._todo_block(
                "3. Are the assumptions and variables correct?"
            ),
            self._todo_block(
                "4. Review proposed edges in Edge Suggestions property."
            ),
            self._todo_block(
                "5. Set verification_status → verified or rejected"
            ),
            self._divider_block(),
        ]

    # -- Paper page body patching (REQ-9) --------------------------------------

    def _patch_paper_page(self, paper_page_id: str, ki_page_ids: list[str]) -> None:
        """
        Append an '## Extracted Concepts' heading and callout to the paper page.

        Idempotent: checks whether the heading already exists before appending.
        Also skips if ki_page_ids is empty.
        """
        if not ki_page_ids:
            return
        try:
            existing_blocks = self.notion.get_block_children(paper_page_id)
            for block in existing_blocks:
                if block.get("type") == "heading_2":
                    rt = block.get("heading_2", {}).get("rich_text", [])
                    text = "".join(seg.get("plain_text", "") for seg in rt)
                    if "Extracted Concepts" in text:
                        logger.debug(
                            "PaperPage %s: 'Extracted Concepts' heading already exists — skipping patch.",
                            paper_page_id,
                        )
                        return

            count = len(ki_page_ids)
            callout_text = (
                f"{count} concept(s) extracted into Knowledge Inbox. "
                "Filter KI by Source Paper to review."
            )
            blocks: list[dict] = [
                self._heading_block("Extracted Concepts"),
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [
                            {"type": "text", "text": {"content": callout_text[:2000]}}
                        ],
                        "icon": {"type": "emoji", "emoji": "📚"},
                        "color": "blue_background",
                    },
                },
            ]
            self._append_blocks_in_batches(paper_page_id, blocks)
            logger.info(
                "PaperPage %s: appended Extracted Concepts section (%d concept(s)).",
                paper_page_id, count,
            )
        except Exception:
            logger.warning(
                "PaperPage %s: could not patch paper page body — continuing.",
                paper_page_id,
            )

    def _append_blocks_in_batches(self, page_id: str, blocks: list[dict]) -> None:
        """Append blocks in batches of 100 to respect the Notion API limit."""
        for i in range(0, len(blocks), NOTION_BLOCKS_PER_REQUEST):
            batch = blocks[i : i + NOTION_BLOCKS_PER_REQUEST]
            self.notion.append_block_children(block_id=page_id, children=batch)

    # -- Property / page title helpers -----------------------------------------

    @staticmethod
    def _get_page_title(page: dict) -> str:
        """Extract the plain-text title from a raw Notion page object."""
        for value in page.get("properties", {}).values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return ""
        return ""

    @staticmethod
    def _get_text_prop(props: dict, key: str) -> str:
        """
        Extract plain text from a Notion property.

        Handles both rich_text and url property types.
        """
        prop = props.get(key, {})
        if prop.get("type") == "url":
            return prop.get("url") or ""
        try:
            return prop["rich_text"][0]["plain_text"]
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _get_title_prop(props: dict) -> str:
        """Extract plain text from a Notion title property."""
        for value in props.values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return "unknown"
        return "unknown"

    @staticmethod
    def _get_multi_select_prop(props: dict, key: str) -> list[str]:
        """Extract option names from a Notion multi_select property."""
        try:
            return [opt["name"] for opt in props[key]["multi_select"]]
        except (KeyError, TypeError):
            return []

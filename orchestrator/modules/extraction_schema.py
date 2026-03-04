"""
modules/extraction_schema.py — Pydantic models for OpenAI extraction output
────────────────────────────────────────────────────────────────────────────
Defines the canonical schema that the OpenAI extraction prompt must produce,
plus link-stage models and validation helpers.

EXTRACTION_VERSION should be bumped whenever the prompt or schema changes
so that the job ledger can detect stale extractions and re-run them.

Changelog:
  v1 — original schema: type, name, content, assumptions, suggested_hub
  v2 — hardened schema: type, title, statement_latex, assumptions, variables,
        conclusion, source_pages, source_quotes, confidence; hub_suggestions
        stored as text only; verification_status added to Knowledge Inbox.
  v3 — 3-stage pipeline: canonical_keywords, prereq_keywords,
        downstream_keywords added to MathObject; LinkEdge / ConceptLinkResult /
        validate_link_result added for Stage 3 graph-linking;
        ProofTechnique type added.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Version string ─────────────────────────────────────────────────────────────
EXTRACTION_VERSION: str = "v3"

# ── Allowed concept types ─────────────────────────────────────────────────────
ALLOWED_CONCEPT_TYPES: frozenset[str] = frozenset(
    {
        "Definition",
        "Theorem",
        "Lemma",
        "Algorithm",
        "Assumption",
        "Proof",
        "ProofTechnique",
    }
)

# ── Allowed edge relation types ────────────────────────────────────────────────
ALLOWED_EDGE_TYPES: frozenset[str] = frozenset(
    {"depends_on", "enables", "generalizes", "special_case_of", "related"}
)

# ── Edge caps per concept ──────────────────────────────────────────────────────
EDGE_CAPS: dict[str, int] = {
    "depends_on": 3,
    "enables": 3,
    "generalizes": 2,
    "special_case_of": 2,
    "related": 5,
}
# Bump this whenever the extraction schema or system prompt changes.
# Can be overridden via the EXTRACTION_VERSION environment variable.
# Changelog:
#   v1 — original schema: type, name, content, assumptions, suggested_hub
#   v2 — hardened schema: type, title, statement_latex, assumptions, variables,
#         conclusion, source_pages, source_quotes, confidence; hub_suggestions
#         stored as text only; verification_status added to Knowledge Inbox.
#   v3 — extended schema: suggested_hub (proper field), interpretation, proof_idea,
#         source_anchors, named_tools, setting, result_category,
#         canonical_keywords, prereq_keywords, downstream_keywords, aliases.
EXTRACTION_VERSION: str = os.environ.get("EXTRACTION_VERSION", "v3")


# ── Stage 1 sub-models ─────────────────────────────────────────────────────────


class MathObject(BaseModel):
    """
    A single extracted mathematical concept from a paper (Stage 1 output).

    Fields mirror the OpenAI system prompt schema so that JSON deserialisation
    is a direct, lossless mapping.
    """

    type: str = Field(
        ...,
        description=(
            "One of: Definition, Theorem, Lemma, Algorithm, "
            "Assumption, Proof, ProofTechnique"
        ),
    )
    title: str = Field(
        ...,
        description="Descriptive canonical concept name (never 'Theorem 1').",
    )
    statement_latex: str = Field(
        ...,
        description="Exact mathematical statement in valid LaTeX.",
    )
    assumptions: str = Field(
        default="None explicitly stated.",
        description="Boundary conditions or assumptions required for the statement.",
    )
    variables: str = Field(
        default="",
        description="Comma-separated list of variables with brief descriptions.",
    )
    conclusion: str = Field(
        default="",
        description="The conclusion or result established, in plain English.",
    )
    source_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers in the source PDF where this concept appears.",
    )
    source_quotes: Optional[str] = Field(
        default=None,
        description="Optional verbatim quote from the paper (max 25 words).",
        max_length=200,
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score in [0, 1].",
    )
    suggested_hub: str = Field(
        default="Uncategorized",
        description="Hub name from ALLOWED_HUBS.",
    )
    canonical_keywords: list[str] = Field(
        default_factory=list,
        description="5–15 canonical keywords describing what this concept IS.",
    )
    prereq_keywords: list[str] = Field(
        default_factory=list,
        description="5–15 keywords for concepts this result REQUIRES / builds on.",
    )
    downstream_keywords: list[str] = Field(
        default_factory=list,
        description="5–15 keywords for concepts this result ENABLES or supports.",
    )

    # ── Extended fields (v3) ───────────────────────────────────────────────────

    suggested_hub: str = Field(
        default="",
        description="Suggested knowledge hub from ALLOWED_HUBS.",
    )
    interpretation: str = Field(
        default="",
        description="Plain-English meaning of this mathematical object.",
    )
    proof_idea: str = Field(
        default="",
        description="Brief sketch of the proof technique (if applicable).",
    )
    source_anchors: str = Field(
        default="",
        description="Section/equation references, e.g. 'Section 3.2; Eq. (12)'.",
    )
    named_tools: list[str] = Field(
        default_factory=list,
        description="Named mathematical tools used, e.g. 'Banach fixed-point'.",
    )
    setting: list[str] = Field(
        default_factory=list,
        description="Mathematical setting tags, e.g. 'finite_state', 'continuous'.",
    )
    result_category: str = Field(
        default="",
        description=(
            "One of: existence, uniqueness, convergence, stability, approximation."
        ),
    )
    canonical_keywords: list[str] = Field(
        default_factory=list,
        description="Primary keywords identifying this concept.",
    )
    prereq_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords of prerequisite concepts this object builds on.",
    )
    downstream_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords of concepts this object enables or is used by.",
    )
    aliases: str = Field(
        default="",
        description="Alternative names for this concept.",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ALLOWED_CONCEPT_TYPES:
            raise ValueError(
                f"type must be one of {sorted(ALLOWED_CONCEPT_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("source_quotes")
    @classmethod
    def validate_quote_length(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        words = v.split()
        if len(words) > 25:
            # Truncate rather than reject — model may overshoot by one word
            return " ".join(words[:25])
        return v


class ExtractionResult(BaseModel):
    """Top-level extraction result returned by Stage 1 (OpenAI extraction call)."""

    one_liner: str = Field(
        ...,
        description="One-sentence summary of the paper's main contribution.",
    )
    active_themes: list[str] = Field(
        default_factory=list,
        description="High-level thematic tags for the paper.",
    )
    extracted_concepts: list[MathObject] = Field(
        default_factory=list,
        description="All extracted mathematical objects.",
    )

    @model_validator(mode="after")
    def at_least_one_concept(self) -> "ExtractionResult":
        # Log a warning but do not reject — paper may genuinely have no math.
        if not self.extracted_concepts:
            import logging
            logging.getLogger(__name__).warning(
                "ExtractionResult: no extracted_concepts — paper may lack formal math."
            )
        return self


# ── Validation helpers ─────────────────────────────────────────────────────────


def validate_extraction(raw: dict) -> tuple[ExtractionResult, list[str]]:
    """
    Attempt to parse *raw* (a plain dict from ``json.loads``) into an
    :class:`ExtractionResult`.

    Returns
    -------
    (result, errors) :
        ``result`` is the parsed :class:`ExtractionResult` on success, or a
        best-effort partial object on failure.
        ``errors`` is an empty list on success, or a list of human-readable
        error strings on validation failure.
    """
    from pydantic import ValidationError

    errors: list[str] = []
    try:
        result = ExtractionResult.model_validate(raw)
        return result, errors
    except ValidationError as exc:
        for error in exc.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")

        # Build a degraded result with confidence=0 for any concepts present
        concepts_raw: list[dict] = raw.get("extracted_concepts", [])
        degraded_concepts: list[MathObject] = []
        for c in concepts_raw:
            try:
                obj = MathObject.model_validate(c)
                degraded_concepts.append(obj)
            except Exception:
                pass

        degraded = ExtractionResult(
            one_liner=str(raw.get("one_liner", ""))[:2000],
            active_themes=list(raw.get("active_themes", [])),
            extracted_concepts=degraded_concepts,
        )
        return degraded, errors


def latex_sanity_check(latex: str) -> list[str]:
    """
    Run lightweight sanity checks on a LaTeX string.

    Checks
    ------
    1. Balanced curly braces ``{}``.
    2. Every ``\\begin{env}`` has a matching ``\\end{env}`` (same env name).

    Returns
    -------
    list[str]
        Empty list if no issues; otherwise a list of human-readable
        problem descriptions.
    """
    issues: list[str] = []

    # ── Check 1: balanced braces ───────────────────────────────────────────────
    depth = 0
    for i, ch in enumerate(latex):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                issues.append(
                    f"Unmatched closing brace '}}' at position {i}."
                )
                depth = 0  # reset to continue checking
    if depth > 0:
        issues.append(f"Unclosed opening brace: {depth} '{{' left unmatched.")

    # ── Check 2: \begin{env} / \end{env} pairing ──────────────────────────────
    begins_iter = list(re.finditer(r"\\begin\{([^}]+)\}", latex))
    ends_iter = list(re.finditer(r"\\end\{([^}]+)\}", latex))

    # Stack-based matching: environments must close in reverse order of opening.
    env_stack: list[str] = []
    end_index = 0

    for m in begins_iter:
        env_stack.append(m.group(1))

    # Reset and do a proper stack walk over the combined token stream.
    env_stack = []
    tokens = sorted(
        [(m.start(), "begin", m.group(1)) for m in begins_iter]
        + [(m.start(), "end", m.group(1)) for m in ends_iter],
        key=lambda t: t[0],
    )

    for _pos, kind, env in tokens:
        if kind == "begin":
            env_stack.append(env)
        else:  # "end"
            if env_stack and env_stack[-1] == env:
                env_stack.pop()
            else:
                if env in env_stack:
                    issues.append(
                        f"\\end{{{env}}} closes out of order "
                        f"(expected \\end{{{env_stack[-1] if env_stack else '?'}}})."
                    )
                    # Remove the specific occurrence using its index so that
                    # duplicate environment names are handled correctly.
                    env_stack.pop(env_stack.index(env))
                else:
                    issues.append(
                        f"\\end{{{env}}} has no matching \\begin{{{env}}}."
                    )

    for env in env_stack:
        issues.append(f"\\begin{{{env}}} has no matching \\end{{{env}}}.")

    return issues


# ── Stage 3 sub-models ─────────────────────────────────────────────────────────


class LinkEdge(BaseModel):
    """A single directed edge in the concept graph."""

    target_concept_id: str = Field(
        ...,
        description="Notion page ID of the target concept in Second Brain.",
    )
    target_title: str = Field(
        ...,
        description="Human-readable title of the target concept.",
    )
    rationale: str = Field(
        ...,
        description="One-sentence explanation of why this edge exists.",
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ConceptLinkResult(BaseModel):
    """
    Graph edges for a single extracted concept (Stage 3 output).

    Each edge list is capped to EDGE_CAPS at validation time.
    """

    depends_on: list[LinkEdge] = Field(default_factory=list)
    enables: list[LinkEdge] = Field(default_factory=list)
    generalizes: list[LinkEdge] = Field(default_factory=list)
    special_case_of: list[LinkEdge] = Field(default_factory=list)
    related: list[LinkEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_caps(self) -> "ConceptLinkResult":
        """Silently trim any edge list that exceeds its cap."""
        self.depends_on = self.depends_on[: EDGE_CAPS["depends_on"]]
        self.enables = self.enables[: EDGE_CAPS["enables"]]
        self.generalizes = self.generalizes[: EDGE_CAPS["generalizes"]]
        self.special_case_of = self.special_case_of[: EDGE_CAPS["special_case_of"]]
        self.related = self.related[: EDGE_CAPS["related"]]
        return self


# ── Stage 3 validation helpers ─────────────────────────────────────────────────


def validate_link_result(raw: dict) -> tuple["ConceptLinkResult", list[str]]:
    """
    Attempt to parse *raw* into a :class:`ConceptLinkResult`.

    Returns
    -------
    (result, errors)
        On failure returns an empty ConceptLinkResult so the pipeline can
        continue in degraded mode (empty edge lists).
    """
    from pydantic import ValidationError

    errors: list[str] = []
    try:
        return ConceptLinkResult.model_validate(raw), errors
    except ValidationError as exc:
        for error in exc.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")

        # Best-effort: parse each edge list individually.
        partial: dict = {}
        for edge_type in EDGE_CAPS:
            valid_edges: list[LinkEdge] = []
            for e in raw.get(edge_type, []):
                try:
                    valid_edges.append(LinkEdge.model_validate(e))
                except Exception:
                    pass
            partial[edge_type] = valid_edges

        return ConceptLinkResult(**partial), errors

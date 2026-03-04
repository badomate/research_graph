"""
modules/extraction_schema.py — Pydantic models for OpenAI extraction output
────────────────────────────────────────────────────────────────────────────
Defines the canonical schema that the OpenAI extraction prompt must produce,
plus validation helpers used by the ingestion engine.

EXTRACTION_VERSION should be bumped whenever the prompt or schema changes
so that the job ledger can detect stale extractions and re-run them.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Version string ─────────────────────────────────────────────────────────────
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


# ── Sub-models ─────────────────────────────────────────────────────────────────


class MathObject(BaseModel):
    """
    A single extracted mathematical concept from a paper.

    Fields mirror the OpenAI system prompt schema so that JSON deserialization
    is a direct, lossless mapping.
    """

    type: str = Field(
        ...,
        description="One of: Definition, Theorem, Lemma, Algorithm, Assumption, Proof",
    )
    title: str = Field(
        ...,
        description="Short label for the concept (name / theorem number).",
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
        allowed = {
            "Definition",
            "Theorem",
            "Lemma",
            "Algorithm",
            "Assumption",
            "Proof",
        }
        if v not in allowed:
            raise ValueError(
                f"type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v

    @field_validator("source_quotes")
    @classmethod
    def validate_quote_length(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        word_count = len(v.split())
        if word_count > 25:
            # Truncate rather than reject — model may overshoot by one word
            return " ".join(v.split()[:25])
        return v


class ExtractionResult(BaseModel):
    """
    Top-level extraction result returned by the OpenAI extraction call.
    """

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

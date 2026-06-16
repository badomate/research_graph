"""
modules/cost.py — pre-flight cost estimation for Marker and Claude.

Pure functions (no I/O) so the web app can show an estimate before the user
commits to a parse/analysis job, and the workers can record both the estimate
and (where available) the actual cost on the job rows.

Rates come from :class:`Config` and are user-configurable — vendor pricing
changes, so never hard-code a price in calling code.
"""
from __future__ import annotations

from dataclasses import dataclass

# Heuristic output-token budgets per analysis type (the model rarely emits more
# structured output than this for a single paper scope). Used only to *bound* the
# Claude estimate; actuals are recorded from the API response.
ANALYSIS_OUTPUT_TOKENS: dict[str, int] = {
    "triage_summary": 700,
    "claim_extraction": 1500,
    "math_object_extraction": 2500,
    "theorem_assumption_extraction": 2500,
    "novelty_risk": 1200,
    "project_relevance": 800,
    "citation_suggestions": 1000,
    "baseline_detection": 1000,
    "limitation_extraction": 900,
}
_DEFAULT_OUTPUT_TOKENS = 1200

# cl100k_base undercounts Claude tokens on LaTeX-dense math; mirror the safety
# margin used in the extractor so estimates don't understate dense papers.
_TOKEN_MARGIN = 1.2


@dataclass(frozen=True)
class MarkerEstimate:
    pages: int
    price_per_page: float
    cost: float


@dataclass(frozen=True)
class ClaudeEstimate:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    # A low/high band: structured extraction output is variable, so we present a
    # range rather than a single misleadingly-precise number.
    cost_low: float
    cost_high: float

    @property
    def cost_mid(self) -> float:
        return (self.cost_low + self.cost_high) / 2


def estimate_tokens(text: str) -> int:
    """Approximate Claude input tokens for a block of text (with safety margin)."""
    if not text:
        return 0
    approx = 0
    try:
        import tiktoken  # type: ignore

        approx = len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        approx = 0
    if approx <= 0:                 # tiktoken missing/stubbed → char heuristic
        approx = max(1, len(text) // 4)
    return int(approx * _TOKEN_MARGIN)


def estimate_marker_cost(pages: int, price_per_page: float) -> MarkerEstimate:
    pages = max(0, int(pages))
    return MarkerEstimate(pages=pages, price_per_page=price_per_page, cost=round(pages * price_per_page, 4))


def estimate_claude_cost(
    *,
    input_tokens: int,
    analysis_type: str,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    system_overhead_tokens: int = 1200,
) -> ClaudeEstimate:
    """Estimate Claude cost for one analysis call.

    Output tokens are heuristic per analysis type; we surface a ±40% band to
    reflect that structured-output length is hard to predict.
    """
    in_tok = max(0, int(input_tokens)) + max(0, int(system_overhead_tokens))
    out_tok = ANALYSIS_OUTPUT_TOKENS.get(analysis_type, _DEFAULT_OUTPUT_TOKENS)

    input_cost = in_tok / 1_000_000 * input_price_per_mtok
    output_cost = out_tok / 1_000_000 * output_price_per_mtok
    mid = input_cost + output_cost
    return ClaudeEstimate(
        input_tokens=in_tok,
        output_tokens=out_tok,
        input_cost=round(input_cost, 4),
        output_cost=round(output_cost, 4),
        cost_low=round(input_cost + output_cost * 0.6, 4),
        cost_high=round(input_cost + output_cost * 1.4, 4),
    )


def actual_claude_cost(
    *, input_tokens: int, output_tokens: int, input_price_per_mtok: float, output_price_per_mtok: float
) -> float:
    """Compute the realized Claude cost from an API response's token counts."""
    return round(
        input_tokens / 1_000_000 * input_price_per_mtok
        + output_tokens / 1_000_000 * output_price_per_mtok,
        6,
    )

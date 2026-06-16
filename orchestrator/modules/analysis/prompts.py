"""
modules/analysis/prompts.py — analysis-type registry for the analysis worker.

Each analysis type maps to a system prompt, the AiSuggestion type its outputs
become, and whether the model returns a list of items or a single object. Every
prompt instructs Claude to emit STRICT JSON only — the worker parses it and never
trusts free text. Prompt text is versioned via ``PROMPT_VERSION`` so suggestions
record exactly which prompt produced them (requirement 13).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..store import SuggestionType

PROMPT_VERSION = "analysis-v1"


@dataclass(frozen=True)
class AnalysisSpec:
    system_prompt: str
    suggestion_type: str
    is_list: bool          # True → JSON array of items; False → single JSON object
    result_key: str = "items"


_JSON_RULES = (
    "\n\nOUTPUT RULES:\n"
    "- Respond with STRICT JSON only. No prose, no markdown fences.\n"
    "- Use [] or {} exactly as specified. Do not invent fields.\n"
    "- Ground every item in the provided text; if unsupported, omit it.\n"
)

REGISTRY: dict[str, AnalysisSpec] = {
    "triage_summary": AnalysisSpec(
        system_prompt=(
            "You are triaging a math-heavy research paper for a PhD researcher. "
            "From the provided excerpts, produce a concise triage summary. "
            'Return a JSON object: {"text": "<3-5 sentence summary>", '
            '"topics": ["..."], "why_relevant": "<one sentence>"}.' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.SUMMARY.value,
        is_list=False,
    ),
    "claim_extraction": AnalysisSpec(
        system_prompt=(
            "Extract the paper's concrete contributions/claims from the excerpts. "
            'Return a JSON array of objects: '
            '[{"text": "<claim>", "kind": "result|method|empirical", '
            '"source_quote": "<short quote>"}].' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.CLAIM.value,
        is_list=True,
    ),
    "math_object_extraction": AnalysisSpec(
        system_prompt=(
            "Extract definitions, theorems, lemmas, propositions and assumptions "
            "from the excerpts. Preserve LaTeX. Return a JSON array of objects: "
            '[{"type": "definition|theorem|lemma|proposition|corollary|assumption", '
            '"title": "...", "statement_latex": "...", "assumptions": "...", '
            '"variables": "...", "conclusion": "...", "source_quotes": ["..."], '
            '"confidence": 0.0}].' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.MATH_OBJECT.value,
        is_list=True,
    ),
    "theorem_assumption_extraction": AnalysisSpec(
        system_prompt=(
            "Extract ONLY theorem statements and the assumptions they rely on. "
            "Preserve LaTeX; do not include proofs. Return a JSON array: "
            '[{"type": "theorem|assumption", "title": "...", '
            '"statement_latex": "...", "assumptions": "...", "conclusion": "...", '
            '"source_quotes": ["..."], "confidence": 0.0}].' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.THEOREM.value,
        is_list=True,
    ),
    "novelty_risk": AnalysisSpec(
        system_prompt=(
            "Assess how this paper affects the novelty of the reader's own project. "
            'Return a JSON object: {"risk_level": "low|medium|high", '
            '"assessment": "<2-4 sentences>", "overlaps": ["..."], '
            '"differentiators": ["..."]}.' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.NOVELTY_RISK.value,
        is_list=False,
    ),
    "project_relevance": AnalysisSpec(
        system_prompt=(
            "Classify this paper's relevance to the reader's project. "
            'Return a JSON object: {"role": '
            '"core|direct_competitor|baseline|theory_tool|background|citation_only|maybe_relevant|irrelevant", '
            '"rationale": "<one sentence>"}.' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.PROJECT_LINK.value,
        is_list=False,
    ),
    "citation_suggestions": AnalysisSpec(
        system_prompt=(
            "Suggest where the reader should cite this paper. Return a JSON array: "
            '[{"context": "<where/why to cite>", "claim": "<what it supports>"}].' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.CITATION_USE.value,
        is_list=True,
    ),
    "baseline_detection": AnalysisSpec(
        system_prompt=(
            "Identify methods/results that could serve as baselines or direct "
            'comparisons. Return a JSON array: [{"name": "...", '
            '"why": "...", "metric": "..."}].' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.BASELINE_CANDIDATE.value,
        is_list=True,
    ),
    "limitation_extraction": AnalysisSpec(
        system_prompt=(
            "Extract the paper's stated or implied limitations and assumptions the "
            'reader must mention. Return a JSON array: [{"text": "...", '
            '"kind": "assumption|limitation", "source_quote": "..."}].' + _JSON_RULES
        ),
        suggestion_type=SuggestionType.LIMITATION.value,
        is_list=True,
    ),
}

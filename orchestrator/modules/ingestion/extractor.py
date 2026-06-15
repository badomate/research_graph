"""
modules/ingestion/extractor.py — Stage 1 LLM extraction service.

Dispatches to single-shot, two-pass, or section-by-section extraction.
"""
from __future__ import annotations

import logging
import os
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config
from ..exceptions import ExtractionError
from ..extraction_schema import (
    ExtractionResult,
    MathObject,
    SkeletonConcept,
    SkeletonResult,
    latex_sanity_check,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    SKELETON_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

TOKEN_THRESHOLD_CHUNK = 30_000
TOKEN_THRESHOLD_WARN  = 60_000

ENABLE_TWO_PASS_EXTRACTION: bool = os.environ.get(
    "ENABLE_TWO_PASS_EXTRACTION", "false"
).lower() in ("1", "true", "yes")

_TWO_PASS_MIN_TOKENS: int = 15_000
_TWO_PASS_LATEX_DENSITY: float = 8.0
_TWO_PASS_SHORTHAND_RE = re.compile(r'\(H\d+\)|\(A\d+\)|\(C\d+\)')
_TWO_PASS_MIN_CONFIDENCE: float = 0.60
_PASS2_BLOCK_TOKENS: int = 400
_PASS2_MAX_CONTEXT_TOKENS: int = 4_000

_SKIP_SECTION_KEYWORDS = (
    "proof of", "proofs of", "deferred", "technical lemma",
)


# cl100k_base (OpenAI's tokenizer) is only an approximation for Claude — it
# undercounts by ~15-20% on prose and considerably more on LaTeX-dense math.
# Since this estimate gates chunking and input truncation, apply a safety margin
# so dense papers chunk early enough rather than silently overrunning the budget.
_CLAUDE_TOKEN_MARGIN = 1.2


def _build_extraction_user_message(markdown: str) -> str:
    """Build the Stage-1 extraction user message (kept pure for testability)."""
    return (
        "Extract structured knowledge from the following paper.\n\n"
        "INSTRUCTIONS\n"
        "- Follow the schema strictly.\n"
        "- Prefer 3–12 high-value concepts.\n"
        "- Do not output theorem/lemma numbers as titles.\n"
        "- Do not include proof-only microlemmas.\n\n"
        "PAPER MARKDOWN:\n\n"
        f"{markdown[:100_000]}"
    )


def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        approx = len(enc.encode(text))
    except Exception:
        approx = len(text) // 4
    return int(approx * _CLAUDE_TOKEN_MARGIN)


def is_dense_paper(markdown: str, token_count: int) -> bool:
    if token_count < _TWO_PASS_MIN_TOKENS:
        return False
    char_count = len(markdown)
    if char_count == 0:
        return False
    latex_density = len(re.findall(r'\\[a-zA-Z]+', markdown)) / (char_count / 1000)
    if latex_density <= _TWO_PASS_LATEX_DENSITY:
        return False
    if not _TWO_PASS_SHORTHAND_RE.search(markdown):
        return False
    return True


class ExtractionService:
    """Wraps all Stage 1 LLM extraction logic."""

    _SKELETON_SYSTEM_PROMPT: str = SKELETON_SYSTEM_PROMPT

    def __init__(
        self,
        claude_client,
        anthropic_raw,
        config: Config | None = None,
    ) -> None:
        self.claude_client = claude_client
        self.anthropic_raw = anthropic_raw
        self._model = config.claude_model if config is not None else CLAUDE_MODEL
        self._token_threshold_chunk = (
            config.token_threshold_chunk if config is not None else TOKEN_THRESHOLD_CHUNK
        )
        self._token_threshold_warn = (
            config.token_threshold_warn if config is not None else TOKEN_THRESHOLD_WARN
        )
        self._enable_two_pass = (
            config.enable_two_pass_extraction
            if config is not None
            else ENABLE_TWO_PASS_EXTRACTION
        )

    def run_extraction(
        self,
        markdown: str,
        token_count: int,
        hubs: dict[str, str],
        run_id: str,
    ) -> ExtractionResult:
        if token_count > self._token_threshold_warn:
            logger.warning(
                "[%s] Paper is very long (%d tokens) — may risk output truncation.",
                run_id, token_count,
            )
        if token_count > self._token_threshold_chunk:
            logger.info(
                "[%s] Paper exceeds %d tokens — using section-by-section extraction.",
                run_id, self._token_threshold_chunk,
            )
            return self._chunked_extract(markdown, hubs, run_id, token_count)
        if self._enable_two_pass and is_dense_paper(markdown, token_count):
            logger.info(
                "[%s] Dense paper detected (%d tokens) — using two-pass extraction.",
                run_id, token_count,
            )
            return self._two_pass_extract(markdown, hubs, run_id, token_count)
        return self._extract_and_validate(markdown, hubs, run_id)

    def _extract_preamble(self, markdown: str, max_tokens: int = 3000) -> str:
        lines = markdown.split("\n")
        preamble_lines: list[str] = []
        in_intro = True
        for line in lines:
            if line.startswith("## ") or line.startswith("# "):
                heading_lower = line.lstrip("#").strip().lower()
                intro_keywords = ("abstract", "introduction", "notation", "preliminaries", "setup")
                if not any(kw in heading_lower for kw in intro_keywords):
                    if not in_intro:
                        break
                    in_intro = False
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines)
        while _count_tokens(preamble) > max_tokens and "\n" in preamble:
            preamble = preamble[:preamble.rfind("\n")]
        return preamble.strip()

    def _split_by_sections(self, markdown: str) -> list[tuple[str, str]]:
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
        return [
            (h, c) for h, c in sections
            if not any(kw in h.lower() for kw in _SKIP_SECTION_KEYWORDS)
        ]

    @staticmethod
    def normalize_concept_title(title: str) -> str:
        t = title.lower()
        t = re.sub(r'\$[^$]*\$', '', t)
        t = re.sub(r'\\\[.*?\\\]', '', t, flags=re.DOTALL)
        t = re.sub(r'[^\w\s]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    def _chunked_extract(
        self, markdown: str, hubs: dict[str, str], run_id: str, token_count: int
    ) -> ExtractionResult:
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
            if _count_tokens(content) < 100:
                continue
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
                norm = self.normalize_concept_title(concept.title)
                if norm and norm not in seen_titles:
                    seen_titles.add(norm)
                    all_concepts.append(concept)
                else:
                    logger.debug("[%s] Dedup: skipping duplicate concept '%s'.", run_id, concept.title)

        logger.info("[%s] Chunked extraction complete: %d unique concept(s).", run_id, len(all_concepts))
        return ExtractionResult(
            one_liner=merged_one_liner,
            active_themes=merged_themes,
            extracted_concepts=all_concepts,
        )

    def _two_pass_extract(
        self, markdown: str, hubs: dict[str, str], run_id: str, token_count: int
    ) -> ExtractionResult:
        logger.info("[%s] Two-pass: starting Pass 1 skeleton call.", run_id)
        skeleton = self._pass1_skeleton(markdown, run_id)
        if not skeleton:
            logger.warning(
                "[%s] Two-pass: Pass 1 returned no candidates — falling back to single-shot.",
                run_id,
            )
            return self._extract_and_validate(markdown, hubs, run_id)

        high_conf = [s for s in skeleton if s.confidence_preliminary >= _TWO_PASS_MIN_CONFIDENCE]
        logger.info(
            "[%s] Two-pass: %d skeleton candidate(s), %d above threshold.",
            run_id, len(skeleton), len(high_conf),
        )

        all_concepts: list[MathObject] = []
        seen_titles: set[str] = set()
        merged_one_liner: str = ""
        merged_themes: list[str] = []

        for skel in high_conf:
            context = self._build_targeted_context(markdown, skel)
            preamble = (
                f"Extract ONE concept. "
                f"Title hint: {skel.title}. "
                f"Location: {skel.source_anchors}. "
                "Return a single-element extracted_concepts array or [] if not extractable."
            )
            try:
                result = self._extract_and_validate(f"{preamble}\n\n{context}", hubs, run_id)
            except Exception:
                logger.warning(
                    "[%s] Two-pass Pass 2: extraction failed for '%s' — skipping.",
                    run_id, skel.title,
                )
                continue
            if not merged_one_liner and result.one_liner:
                merged_one_liner = result.one_liner
            for theme in result.active_themes:
                if theme not in merged_themes:
                    merged_themes.append(theme)
            for concept in result.extracted_concepts:
                norm = self.normalize_concept_title(concept.title)
                if norm and norm not in seen_titles:
                    seen_titles.add(norm)
                    all_concepts.append(concept)

        logger.info("[%s] Two-pass complete: %d unique concept(s).", run_id, len(all_concepts))
        if not all_concepts:
            logger.warning("[%s] Two-pass returned no concepts — falling back to single-shot.", run_id)
            return self._extract_and_validate(markdown, hubs, run_id)

        return ExtractionResult(
            one_liner=merged_one_liner,
            active_themes=merged_themes,
            extracted_concepts=all_concepts,
        )

    def _pass1_skeleton(self, markdown: str, run_id: str) -> list[SkeletonConcept]:
        try:
            result = self.claude_client.messages.create(
                model=self._model,
                max_tokens=1000,
                system=[{
                    "type": "text",
                    "text": self._SKELETON_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        "Identify candidate concepts in the following paper.\n\n"
                        f"PAPER MARKDOWN:\n\n{markdown[:100_000]}"
                    ),
                }],
                response_model=SkeletonResult,
            )
            return result.concepts
        except Exception:
            logger.warning("[%s] Two-pass Pass 1: skeleton call failed.", run_id, exc_info=True)
            return []

    def _build_targeted_context(self, markdown: str, skeleton: SkeletonConcept) -> str:
        blocks: list[str] = []

        def _slice_around(anchor: str | None, token_budget: int) -> str:
            if not anchor:
                return ""
            idx = markdown.lower().find(anchor.lower()[:60])
            if idx < 0:
                return ""
            char_budget = token_budget * 4
            half = char_budget // 2
            start = max(0, idx - half)
            end = min(len(markdown), idx + half)
            return markdown[start:end]

        stmt_block = _slice_around(skeleton.source_anchors or None, _PASS2_BLOCK_TOKENS)
        if stmt_block:
            blocks.append(stmt_block)

        remaining = _PASS2_MAX_CONTEXT_TOKENS - _count_tokens("\n\n".join(blocks))
        if remaining > 0 and skeleton.assumption_anchor:
            assume_block = _slice_around(skeleton.assumption_anchor, remaining)
            if assume_block:
                blocks.append(assume_block)

        remaining = _PASS2_MAX_CONTEXT_TOKENS - _count_tokens("\n\n".join(blocks))
        if remaining > 0 and skeleton.notation_anchor:
            notation_block = _slice_around(skeleton.notation_anchor, remaining)
            if notation_block:
                blocks.append(notation_block)

        context = "\n\n".join(blocks)
        while _count_tokens(context) > _PASS2_MAX_CONTEXT_TOKENS and "\n" in context:
            context = context[: context.rfind("\n")]
        return context.strip() or markdown[:_PASS2_MAX_CONTEXT_TOKENS * 4]

    def _extract_and_validate(
        self, markdown: str, hubs: dict[str, str], run_id: str
    ) -> ExtractionResult:
        try:
            result = self._call_claude_extract(markdown, hubs)
        except Exception as exc:
            raise ExtractionError(
                f"[{run_id}] Claude extraction failed after retries"
            ) from exc
        for concept in result.extracted_concepts:
            issues = latex_sanity_check(concept.statement_latex)
            if issues:
                logger.warning(
                    "[%s] LaTeX issues in concept '%s': %s", run_id, concept.title, issues
                )
                concept.confidence = min(concept.confidence, 0.5)
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _call_claude_extract(self, markdown: str, hubs: dict[str, str]) -> ExtractionResult:
        hub_names_str = (
            ", ".join(f'"{name}"' for name in hubs) if hubs else '"Uncategorized"'
        )
        system_prompt = EXTRACTION_SYSTEM_PROMPT.replace(
            "[INJECT_DYNAMIC_HUBS_HERE]", hub_names_str
        )
        user_message = _build_extraction_user_message(markdown)
        result = self.claude_client.messages.create(
            model=self._model,
            max_tokens=8192,
            # The system prompt (with injected hubs) is stable across every paper
            # and concept in a run — cache it so only the per-paper markdown is
            # billed at full input price after the first call.
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
            response_model=ExtractionResult,
        )
        logger.info("Claude extraction response received.")
        return result

"""
modules/ingestion/linker.py — Stage 3 LLM linking service.

Calls Claude with the dual-channel edge confirmation prompt and routes
proposals into auto vs. suggest channels.

Two execution modes share the same prompt-building and routing logic:
  - Synchronous (default): one Claude call per concept via ``run_stage_link``.
  - Batched (opt-in, ``link_use_batch_api``): all of a paper's concepts are
    submitted as one Message Batch via ``run_stage_link_batch`` — 50% cheaper
    and processed in parallel server-side, at the cost of async polling and no
    two-temperature validation pass.
"""
from __future__ import annotations

import logging
import os
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config
from ..exceptions import LinkingError
from ..extraction_schema import (
    ConceptLinkResult,
    CrossPaperLinkResult,
    EdgeProposal,
    MathObject,
)
from ..scoring.candidate_scorer import (
    CandidateScore,
    ConceptData,
    _dominant_signal,
    route_edge_proposals,
)
from .prompts import (
    EDGE_CONFIRMATION_SYSTEM_PROMPT,
    LINKING_SYSTEM_PROMPT_V1,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

ENABLE_TWO_TEMPERATURE_VALIDATION: bool = os.environ.get(
    "ENABLE_TWO_TEMPERATURE_VALIDATION", "false"
).lower() in ("1", "true", "yes")


class ConceptLinker:
    """Wraps Stage 3 LLM linking logic."""

    _LINK_TOOL_NAME = "record_links"

    def __init__(
        self,
        claude_client,
        config: Config | None = None,
        anthropic_raw=None,
    ) -> None:
        self.claude_client = claude_client
        # Raw (non-instructor) Anthropic client — required only for the batch path.
        self._anthropic_raw = anthropic_raw
        self._model = config.claude_model if config is not None else CLAUDE_MODEL
        self._enable_two_temperature_validation = (
            config.enable_two_temperature_validation
            if config is not None
            else ENABLE_TWO_TEMPERATURE_VALIDATION
        )
        self._use_batch_api = config.link_use_batch_api if config is not None else False
        self._batch_poll_seconds = (
            config.link_batch_poll_seconds if config is not None else 30
        )
        self._batch_timeout_seconds = (
            config.link_batch_timeout_seconds if config is not None else 1800
        )

    def run_stage_link(
        self,
        concept: MathObject,
        candidates: list[dict],
        run_id: str,
    ) -> ConceptLinkResult | CrossPaperLinkResult:
        """
        Dispatch to the correct linking prompt based on candidate type.

        TF-IDF path (no Qdrant): write all candidates as suggest-only, no LLM call.
        Same-paper-only candidates: legacy v1 prompt.
        Cross-paper candidates: dual-channel v2 prompt.
        """
        if not candidates:
            return ConceptLinkResult()

        has_cross_paper = any(c.get("_concept_data") is not None for c in candidates)
        is_tfidf_path = not any(c.get("_score_obj") is not None for c in candidates)

        if is_tfidf_path and not has_cross_paper:
            return self._tfidf_suggestions(concept, candidates)

        try:
            if has_cross_paper:
                result = self._call_claude_link_v2(concept, candidates)
            else:
                result = self._call_claude_link_v1(concept, candidates)
        except Exception as exc:
            raise LinkingError(
                f"[{run_id}] LLM linking failed for '{concept.title}'"
            ) from exc
        return result

    def _tfidf_suggestions(
        self, concept: MathObject, candidates: list[dict]
    ) -> CrossPaperLinkResult:
        """TF-IDF fallback (no Qdrant): emit suggest-only edges, no LLM call."""
        proposals = [
            EdgeProposal(
                source_concept_title=concept.title,
                target_concept_title=c.get("title", "(unknown)"),
                target_notion_page_id=c.get("id", ""),
                relation_type="related",
                direction="A_to_B",
                channel="suggest",
                confidence=0.0,
                justification="TF-IDF fallback — no LLM confirmation",
                driving_fields=["keywords"],
                falsifiability="",
                needs_review=True,
            )
            for c in candidates
            if c.get("id")
        ]
        return CrossPaperLinkResult(proposals=proposals)

    @staticmethod
    def _build_link_v1_user_message(
        concept: MathObject, candidates: list[dict]
    ) -> str:
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
        return (
            f"CONCEPT:\n{concept_summary}\n\n"
            f"CANDIDATES:\n{candidate_lines}\n\n"
            "Identify relationships. Return JSON only."
        )

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=10, max=120))
    def _call_claude_link_v1(
        self, concept: MathObject, candidates: list[dict]
    ) -> ConceptLinkResult:
        user_message = self._build_link_v1_user_message(concept, candidates)
        return self.claude_client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=[{
                "type": "text",
                "text": LINKING_SYSTEM_PROMPT_V1,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
            response_model=ConceptLinkResult,
        )

    def _build_link_v2_user_message(
        self, concept: MathObject, candidates: list[dict]
    ) -> str:
        _MAX_LATEX = 800

        def _fmt_latex(s: str) -> str:
            if not s:
                return "(not recorded)"
            return (s[:_MAX_LATEX] + " [truncated]") if len(s) > _MAX_LATEX else s

        def _fmt_field(s: str) -> str:
            return s if s else "(not recorded)"

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
            score_obj: CandidateScore | None = cand.get("_score_obj")

            if cd is not None:
                setting_str = ", ".join(cd.setting) if isinstance(cd.setting, list) else str(cd.setting or "")
                named_tools_str = ", ".join(cd.named_tools) if cd.named_tools else "none"
                keywords_str = ", ".join(cd.keywords) if cd.keywords else "none"
                signal_desc = ""
                if score_obj is not None:
                    parts = []
                    if score_obj.named_tool_match:
                        parts.append("named_tool_match=True")
                    if score_obj.assumption_conclusion_overlap > 0:
                        parts.append(f"assumption_conclusion_overlap={score_obj.assumption_conclusion_overlap:.2f}")
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
        return "\n".join(src_lines)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=10, max=60))
    def _call_edge_confirmation(
        self,
        concept: MathObject,
        candidates: list[dict],
        temperature: float = 0.0,
    ) -> CrossPaperLinkResult:
        user_message = self._build_link_v2_user_message(concept, candidates)
        return self.claude_client.messages.create(
            model=self._model,
            max_tokens=4096,
            # Static system prompt reused for every concept's edge confirmation —
            # cache it so only the per-concept candidate block is full-price.
            system=[{
                "type": "text",
                "text": EDGE_CONFIRMATION_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
            response_model=CrossPaperLinkResult,
            temperature=temperature,
        )

    def _validate_auto_edges_two_temperature(
        self,
        concept: MathObject,
        candidates: list[dict],
        first_pass_auto: list,
    ) -> list:
        if not first_pass_auto:
            return []

        target_ids = {p.target_notion_page_id for p in first_pass_auto}
        filtered_candidates = [
            c for c in candidates
            if (c.get("id") in target_ids)
            or (c.get("_concept_data") is not None and c["_concept_data"].notion_page_id in target_ids)
        ]
        if not filtered_candidates:
            return first_pass_auto

        try:
            second_pass_result = self._call_edge_confirmation(
                concept, filtered_candidates, temperature=0.3
            )
        except Exception:
            logger.warning(
                "_validate_auto_edges_two_temperature: second-pass call failed "
                "for '%s' — keeping first-pass auto edges.",
                concept.title, exc_info=True,
            )
            return first_pass_auto

        second_pass_auto, _ = route_edge_proposals(second_pass_result.proposals, scores={})
        second_pass_index = {
            (p.target_notion_page_id, p.relation_type, p.direction): p
            for p in second_pass_auto
        }

        stable: list = []
        for p in first_pass_auto:
            key = (p.target_notion_page_id, p.relation_type, p.direction)
            if key in second_pass_index:
                stable.append(p)
            else:
                p.channel = "suggest"
                p.needs_review = True
                p.demoted_from_auto = True

        return stable

    def _call_claude_link_v2(
        self, concept: MathObject, candidates: list[dict]
    ) -> CrossPaperLinkResult:
        try:
            raw_result = self._call_edge_confirmation(concept, candidates, temperature=0)
        except Exception:
            logger.warning(
                "_call_claude_link_v2: edge-confirmation call failed for '%s'.",
                concept.title, exc_info=True
            )
            return CrossPaperLinkResult()

        return self._route_cross_paper(concept, candidates, raw_result, allow_two_temp=True)

    def _route_cross_paper(
        self,
        concept: MathObject,
        candidates: list[dict],
        raw_result: CrossPaperLinkResult,
        allow_two_temp: bool = True,
    ) -> CrossPaperLinkResult:
        """Annotate, route, and (optionally) two-temperature-validate raw proposals.

        Shared by the synchronous and batch paths. ``allow_two_temp=False``
        skips the second LLM call (used by the batch path, which is single-pass).
        """
        scores_by_id: dict[str, CandidateScore] = {
            cand.get("id", ""): cand["_score_obj"]
            for cand in candidates
            if cand.get("_score_obj") is not None
        }
        concept_data_by_id: dict[str, ConceptData] = {
            cand["_concept_data"].notion_page_id: cand["_concept_data"]
            for cand in candidates
            if cand.get("_concept_data") is not None
        }

        for p in raw_result.proposals:
            p.source_type = concept.type
            cd = concept_data_by_id.get(p.target_notion_page_id)
            if cd:
                p.target_type = cd.concept_type
            score_obj = scores_by_id.get(p.target_notion_page_id)
            if score_obj:
                p.pre_filter_signal = _dominant_signal(score_obj)

        auto_edges, suggest_edges = route_edge_proposals(raw_result.proposals, scores_by_id)

        if allow_two_temp and self._enable_two_temperature_validation and auto_edges:
            same_paper_ids = {
                c.get("id", "") for c in candidates if c.get("_concept_data") is None
            }
            intra_auto = [p for p in auto_edges if p.target_notion_page_id in same_paper_ids]
            cross_auto = [p for p in auto_edges if p.target_notion_page_id not in same_paper_ids]

            stable_cross = self._validate_auto_edges_two_temperature(concept, candidates, cross_auto)
            demoted_from_temp = [
                p for p in cross_auto
                if p.demoted_from_auto and p not in stable_cross
            ]
            suggest_edges = sorted(
                suggest_edges + demoted_from_temp,
                key=lambda x: x.confidence,
                reverse=True,
            )[:4]
            auto_edges = intra_auto + stable_cross

        return CrossPaperLinkResult(
            proposals=auto_edges + suggest_edges,
            low_confidence_suggestions=[],
        )

    # ── Batch path (opt-in via link_use_batch_api) ─────────────────────────────

    @staticmethod
    def _structured_tool(model_cls) -> dict:
        """Build a forced-tool definition that mirrors instructor's tool mode."""
        return {
            "name": ConceptLinker._LINK_TOOL_NAME,
            "description": (
                (model_cls.__doc__ or "Emit the structured linking result.").strip()[:1024]
            ),
            "input_schema": model_cls.model_json_schema(),
        }

    @classmethod
    def _parse_tool_message(cls, message, model_cls):
        """Validate the forced tool_use input from a batch result Message."""
        for block in getattr(message, "content", None) or []:
            if getattr(block, "type", None) == "tool_use":
                try:
                    return model_cls.model_validate(block.input)
                except Exception:
                    logger.warning(
                        "Batch: could not validate %s tool output — using empty result.",
                        model_cls.__name__, exc_info=True,
                    )
                    return model_cls()
        return model_cls()

    def _build_link_batch_params(
        self, concept: MathObject, candidates: list[dict], kind: str
    ) -> dict:
        """Build the Messages params for one concept's batch request (v1 or v2)."""
        if kind == "v2":
            system_prompt = EDGE_CONFIRMATION_SYSTEM_PROMPT
            user_message = self._build_link_v2_user_message(concept, candidates)
            tool = self._structured_tool(CrossPaperLinkResult)
        else:
            system_prompt = LINKING_SYSTEM_PROMPT_V1
            user_message = self._build_link_v1_user_message(concept, candidates)
            tool = self._structured_tool(ConceptLinkResult)
        return {
            "model": self._model,
            "max_tokens": 4096,
            "system": [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": user_message}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": self._LINK_TOOL_NAME},
        }

    def run_stage_link_batch(
        self,
        concept_candidates: list[tuple[MathObject, str, list[dict]]],
        run_id: str,
    ) -> dict[str, ConceptLinkResult | CrossPaperLinkResult]:
        """
        Stage-3 linking via the Message Batches API (50% cost, async).

        ``concept_candidates`` is a list of (concept, ki_page_id, candidates).
        Concepts needing no LLM call (no candidates / TF-IDF-only) are resolved
        locally; the rest are submitted as a single batch and polled to
        completion. Returns ``{ki_page_id: result}`` — a missing key means that
        concept's batch item errored or timed out (logged), and it is left
        unlinked for this run rather than re-billed synchronously.
        """
        results: dict[str, ConceptLinkResult | CrossPaperLinkResult] = {}
        requests: list[dict] = []
        meta: dict[str, tuple] = {}  # custom_id -> (ki_page_id, concept, candidates, kind)

        for idx, (concept, ki_page_id, candidates) in enumerate(concept_candidates):
            if not candidates:
                results[ki_page_id] = ConceptLinkResult()
                continue
            has_cross_paper = any(c.get("_concept_data") is not None for c in candidates)
            is_tfidf_path = not any(c.get("_score_obj") is not None for c in candidates)
            if is_tfidf_path and not has_cross_paper:
                results[ki_page_id] = self._tfidf_suggestions(concept, candidates)
                continue
            kind = "v2" if has_cross_paper else "v1"
            custom_id = f"link_{idx}"
            requests.append({
                "custom_id": custom_id,
                "params": self._build_link_batch_params(concept, candidates, kind),
            })
            meta[custom_id] = (ki_page_id, concept, candidates, kind)

        if not requests:
            return results

        if self._enable_two_temperature_validation:
            logger.warning(
                "[%s] Batch linking routes auto edges single-pass; "
                "two-temperature validation is skipped in batch mode.", run_id,
            )

        logger.info(
            "[%s] Stage 3: submitting %d concept(s) as one batch.", run_id, len(requests)
        )
        messages_by_id = self._submit_and_poll(requests, run_id)

        for custom_id, (ki_page_id, concept, candidates, kind) in meta.items():
            message = messages_by_id.get(custom_id)
            if message is None:
                logger.warning(
                    "[%s] Batch: no result for concept '%s' — left unlinked this run.",
                    run_id, concept.title,
                )
                continue
            if kind == "v2":
                raw = self._parse_tool_message(message, CrossPaperLinkResult)
                results[ki_page_id] = self._route_cross_paper(
                    concept, candidates, raw, allow_two_temp=False
                )
            else:
                results[ki_page_id] = self._parse_tool_message(message, ConceptLinkResult)

        return results

    def _submit_and_poll(self, requests: list[dict], run_id: str) -> dict:
        """Submit one Message Batch, block until it ends (or times out), return
        ``{custom_id: Message}`` for succeeded items."""
        if self._anthropic_raw is None:
            raise LinkingError(
                f"[{run_id}] Batch linking requires a raw Anthropic client "
                "(anthropic_raw was not provided to ConceptLinker)."
            )

        batch = self._anthropic_raw.messages.batches.create(requests=requests)
        logger.info(
            "[%s] Batch %s submitted (%d request(s)); polling every %ds.",
            run_id, batch.id, len(requests), self._batch_poll_seconds,
        )

        deadline = time.monotonic() + self._batch_timeout_seconds
        while True:
            current = self._anthropic_raw.messages.batches.retrieve(batch.id)
            if current.processing_status == "ended":
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "[%s] Batch %s exceeded %ds timeout — cancelling.",
                    run_id, batch.id, self._batch_timeout_seconds,
                )
                try:
                    self._anthropic_raw.messages.batches.cancel(batch.id)
                except Exception:
                    logger.warning("[%s] Batch %s cancel failed.", run_id, batch.id, exc_info=True)
                break
            time.sleep(self._batch_poll_seconds)

        out: dict = {}
        try:
            for result in self._anthropic_raw.messages.batches.results(batch.id):
                if result.result.type == "succeeded":
                    out[result.custom_id] = result.result.message
                else:
                    logger.warning(
                        "[%s] Batch item %s did not succeed: %s",
                        run_id, result.custom_id, result.result.type,
                    )
        except Exception:
            logger.exception("[%s] Batch %s: failed to retrieve results.", run_id, batch.id)
        return out

"""
modules/ingestion/linker.py — Stage 3 LLM linking service.

Calls Claude with the dual-channel edge confirmation prompt and routes
proposals into auto vs. suggest channels.
"""
from __future__ import annotations

import logging
import os

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

    def __init__(self, claude_client, config: Config | None = None) -> None:
        self.claude_client = claude_client
        self._model = config.claude_model if config is not None else CLAUDE_MODEL
        self._enable_two_temperature_validation = (
            config.enable_two_temperature_validation
            if config is not None
            else ENABLE_TWO_TEMPERATURE_VALIDATION
        )

    def run_stage_link(
        self,
        concept: MathObject,
        candidates: list[dict],
        run_id: str,
    ) -> ConceptLinkResult | CrossPaperLinkResult:
        """
        Dispatch to the correct linking prompt based on candidate type.

        TF-IDF path (no Qdrant): write all candidates as suggest-only, no GPT.
        Same-paper-only candidates: legacy v1 prompt.
        Cross-paper candidates: dual-channel v2 prompt.
        """
        if not candidates:
            return ConceptLinkResult()

        has_cross_paper = any(c.get("_concept_data") is not None for c in candidates)
        is_tfidf_path = not any(c.get("_score_obj") is not None for c in candidates)

        if is_tfidf_path and not has_cross_paper:
            proposals = [
                EdgeProposal(
                    source_concept_title=concept.title,
                    target_concept_title=c.get("title", "(unknown)"),
                    target_notion_page_id=c.get("id", ""),
                    relation_type="related",
                    direction="A_to_B",
                    channel="suggest",
                    confidence=0.0,
                    justification="TF-IDF fallback — no GPT confirmation",
                    driving_fields=["keywords"],
                    falsifiability="",
                    needs_review=True,
                )
                for c in candidates
                if c.get("id")
            ]
            return CrossPaperLinkResult(proposals=proposals)

        try:
            if has_cross_paper:
                result = self._call_claude_link_v2(concept, candidates)
            else:
                result = self._call_openai_link(concept, candidates)
        except Exception as exc:
            raise LinkingError(
                f"[{run_id}] LLM linking failed for '{concept.title}'"
            ) from exc
        return result

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=10, max=120))
    def _call_openai_link(
        self, concept: MathObject, candidates: list[dict]
    ) -> ConceptLinkResult:
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
        return self.claude_client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=LINKING_SYSTEM_PROMPT_V1,
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
    def _call_edge_confirmation_gpt(
        self,
        concept: MathObject,
        candidates: list[dict],
        temperature: float = 0.0,
    ) -> CrossPaperLinkResult:
        user_message = self._build_link_v2_user_message(concept, candidates)
        return self.claude_client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=EDGE_CONFIRMATION_SYSTEM_PROMPT,
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
            second_pass_result = self._call_edge_confirmation_gpt(
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
            raw_result = self._call_edge_confirmation_gpt(concept, candidates, temperature=0)
        except Exception:
            logger.warning(
                "_call_claude_link_v2: GPT call failed for '%s'.", concept.title, exc_info=True
            )
            return CrossPaperLinkResult()

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

        if self._enable_two_temperature_validation and auto_edges:
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

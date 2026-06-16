"""
modules/analysis/analysis_worker.py — runs AnalysisJobs.

For one job:
  1. Gather the selected chunks (the AnalysisScope) and build the user message.
  2. Call Claude with the analysis-type system prompt (+ optional reviewer
     regeneration instruction).
  3. Parse the strict-JSON response into AiSuggestion rows — always status
     ``pending`` (the quarantine; AI output never lands in accepted tables).
  4. Record actual input/output tokens + cost on the job; for a regeneration
     targeting one suggestion, link the new suggestion to its parent for lineage.

The model is called via the raw Anthropic client (no instructor schema coupling),
so this worker degrades gracefully and records a clear job error if the key is
missing or the response isn't valid JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

from ..config import Config, get_config
from ..cost import actual_claude_cost, estimate_claude_cost, estimate_tokens
from ..store import JobStatus, Store, SuggestionStatus
from .prompts import PROMPT_VERSION, REGISTRY

logger = logging.getLogger(__name__)

_MAX_CHUNK_CHARS = 120_000


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _extract_json(text: str):
    """Best-effort parse of a JSON object/array from the model's response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} or [...] block.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("response was not valid JSON")


class AnalysisWorker:
    def __init__(self, store: Store, config: Config | None = None, client=None) -> None:
        self.store = store
        self.config = config or get_config()
        self._client = client  # injectable for tests

    def _anthropic(self):
        if self._client is not None:
            return self._client
        if not self.config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def _build_user_message(self, chunks, instruction: str) -> str:
        parts = []
        budget = _MAX_CHUNK_CHARS
        for ch in chunks:
            block = f"### {ch.heading}\n{ch.text}" if ch.heading else ch.text
            block = block[:budget]
            parts.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        body = "\n\n".join(parts)
        msg = f"PAPER EXCERPTS (selected scope):\n\n{body}"
        if instruction.strip():
            msg += f"\n\nADDITIONAL INSTRUCTION FROM THE REVIEWER:\n{instruction.strip()}"
        return msg

    def run_job(self, job_id: str) -> None:
        job = self.store.get_analysis_job(job_id)
        if job is None:
            return
        spec = REGISTRY.get(job.analysis_type)
        if spec is None:
            self.store.update_analysis_job(
                job_id, status=JobStatus.FAILED.value, error=f"unknown analysis_type {job.analysis_type}"
            )
            return

        chunks = self.store.get_chunks(list(job.chunk_ids or []))
        if not chunks:
            # Default to all of the paper's chunks if none were explicitly selected.
            chunks = self.store.chunks_for_paper(job.paper_id)
        if not chunks:
            # Defer (re-queue) while a parse for this paper is still pending/running —
            # supports "save + triage", where the analysis is created before parsing
            # finishes. Give up after enough attempts so it can't loop forever.
            pending_parse = [
                j for j in self.store.list_parse_jobs(job.paper_id)
                if j.status in (JobStatus.PENDING.value, JobStatus.RUNNING.value)
            ]
            if pending_parse and job.attempts < 30:
                self.store.update_analysis_job(job_id, status=JobStatus.PENDING.value,
                                               error="waiting for parse to produce chunks")
                return
            self.store.update_analysis_job(
                job_id, status=JobStatus.FAILED.value, error="no parsed chunks to analyze"
            )
            return

        user_message = self._build_user_message(chunks, job.instruction or "")
        in_tok_est = estimate_tokens(user_message)
        est = estimate_claude_cost(
            input_tokens=in_tok_est, analysis_type=job.analysis_type,
            input_price_per_mtok=self.config.claude_input_price_per_mtok,
            output_price_per_mtok=self.config.claude_output_price_per_mtok,
        )
        model = job.model or self.config.claude_model
        input_hash = _sha(f"{model}|{PROMPT_VERSION}|{job.analysis_type}|{job.instruction}|" +
                          "|".join(c.content_hash for c in chunks))
        self.store.update_analysis_job(
            job_id, model=model, prompt_version=PROMPT_VERSION,
            input_token_estimate=est.input_tokens, output_token_estimate=est.output_tokens,
            cost_estimate=est.cost_mid,
        )

        try:
            client = self._anthropic()
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=[{"type": "text", "text": spec.system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_message}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content)
            data = _extract_json(text)
            items = data if isinstance(data, list) else [data]

            in_tok = getattr(resp.usage, "input_tokens", in_tok_est)
            out_tok = getattr(resp.usage, "output_tokens", 0)
            cost = actual_claude_cost(
                input_tokens=in_tok, output_tokens=out_tok,
                input_price_per_mtok=self.config.claude_input_price_per_mtok,
                output_price_per_mtok=self.config.claude_output_price_per_mtok,
            )

            created = 0
            for item in items:
                if not isinstance(item, dict) or not item:
                    continue
                sug = self.store.create_suggestion(
                    paper_id=job.paper_id, project_id=job.project_id,
                    analysis_job_id=job.id, suggestion_type=spec.suggestion_type,
                    payload_json=item, model=model, prompt_version=PROMPT_VERSION,
                    input_hash=input_hash, output_hash=_sha(json.dumps(item, sort_keys=True)),
                    status=SuggestionStatus.PENDING.value,
                    parent_generation_id=job.target_suggestion_id,
                    regeneration_reason=job.instruction or "",
                )
                created += 1
                # 1:1 regeneration links new→old once; subsequent items are independent.
                if job.target_suggestion_id:
                    job = self.store.update_analysis_job(job_id, target_suggestion_id=None) or job

            self.store.update_analysis_job(
                job_id, status=JobStatus.SUCCEEDED.value,
                input_tokens_actual=in_tok, output_tokens_actual=out_tok,
                cost_actual=cost, error="" if created else "model returned no usable items",
            )
            logger.info("[analysis %s] %s → %d suggestion(s)", job_id, job.analysis_type, created)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[analysis %s] failed", job_id)
            self.store.update_analysis_job(job_id, status=JobStatus.FAILED.value, error=str(exc)[:1000])

    def run_pending(self, limit: int = 5) -> int:
        processed = 0
        while processed < limit:
            job = self.store.claim_next_analysis_job()
            if job is None:
                break
            self.run_job(job.id)
            processed += 1
        return processed

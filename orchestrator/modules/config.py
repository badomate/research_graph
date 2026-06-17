"""
modules/config.py — Centralized configuration via Pydantic Settings
────────────────────────────────────────────────────────────────────
Single source of truth for every environment variable used by the pipeline.

Usage:
    from modules.config import get_config
    cfg = get_config()

`get_config()` returns a module-level singleton so the .env file is parsed
once at startup rather than on every access.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database (SQLite source of truth; replaces Notion) ──────────────────────
    # Default targets the shared Docker volume; override for local dev, e.g.
    #   DATABASE_URL=sqlite:///./app.db
    database_url: str = "sqlite:////data/app.db"
    # Where uploaded PDFs are stored (in-app "Add Paper").
    uploads_dir: str = "/data/uploads"

    # ── Anthropic / LLM ───────────────────────────────────────────────────────
    # Optional so the stack boots with zero config; extraction logs a warning and
    # is skipped if unset.
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    claude_fast_model: str = "claude-sonnet-4-6"

    # ── OpenAI (embeddings only) ──────────────────────────────────────────────
    openai_api_key: str = ""

    # ── Qdrant / vector index ─────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    vector_embedding_backend: str = "openai"
    vector_embedding_model: str = "text-embedding-3-small"
    vector_embedding_dim: int = 1536
    vector_index_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("VECTOR_INDEX_ENABLED", "ENABLE_VECTOR_INDEX"),
    )
    retrieve_candidates_k: int = 30

    # ── Koofr / WebDAV (optional — only the Zotero-attachment intake path) ─────
    koofr_user: str = ""
    koofr_app_password: str = ""
    koofr_pdf_path: str = "/zotero"
    koofr_markdown_path: str = "/zotero_markdown"

    # ── Zotero (optional — intake sync + attachment/notes fetch) ────────────────
    zotero_user_id: str = ""
    zotero_api_key: str = ""
    # Background poller that imports new Zotero library items (replaces Notero).
    zotero_poll_enabled: bool = False
    zotero_poll_minutes: int = 60

    # ── Cost estimation (rates are user-configurable; set to current vendor pricing) ──
    # Marker/Datalab is billed per page; Claude per million tokens. These drive the
    # pre-flight estimates shown before a parse/analysis runs (see modules/cost.py).
    marker_price_per_page: float = 0.01
    # Defaults reflect Claude Sonnet list pricing ($/million tokens). Override per
    # model via CLAUDE_INPUT_PRICE_PER_MTOK / CLAUDE_OUTPUT_PRICE_PER_MTOK.
    claude_input_price_per_mtok: float = 3.0
    claude_output_price_per_mtok: float = 15.0

    # ── Pipeline I/O ──────────────────────────────────────────────────────────
    marker_api_url: str = "http://marker-api:8080"
    pipeline_tmp_dir: str = "/tmp/pipeline"
    extraction_version: str = "v3"
    sb_concept_level: str = "Concept"

    # ── Token / chunking thresholds ───────────────────────────────────────────
    token_threshold_chunk: int = 30_000
    token_threshold_warn: int = 60_000
    enable_two_pass_extraction: bool = False
    enable_two_temperature_validation: bool = False
    two_pass_min_tokens: int = 15_000
    two_pass_latex_density: float = 8.0
    two_pass_min_confidence: float = 0.60
    pass2_block_tokens: int = 400
    pass2_max_context_tokens: int = 4_000

    # ── Candidate pre-filter weights (must sum to 1.0) ────────────────────────
    weight_qdrant: float = 0.40
    weight_named_tool: float = 0.25
    weight_assumption_overlap: float = 0.20
    weight_setting_containment: float = 0.10
    weight_keyword_jaccard: float = 0.05

    # ── Candidate pre-filter drop thresholds ─────────────────────────────────
    named_tool_match_threshold: int = 85
    setting_containment_threshold: int = 80
    assumption_overlap_drop_threshold: float = 0.05
    keyword_jaccard_drop_threshold: float = 0.10
    qdrant_similarity_drop_threshold: float = 0.75
    composite_drop_threshold: float = 0.12

    # ── Edge creation thresholds ──────────────────────────────────────────────
    edge_auto_create_confidence: float = 0.80
    edge_review_flag_confidence: float = 0.65
    edge_max_candidates_to_gpt: int = 20

    # ── When are edges proposed? ──────────────────────────────────────────────
    # Edges connect concepts in the *accepted* graph. By default they are proposed
    # at PROMOTION time (when a concept enters the Second Brain) against the
    # current accepted graph — so an accepted concept links to everything else you
    # have accepted, including the batch promoted alongside it. Set this True to
    # additionally link at extraction time (the old behaviour: links a fresh inbox
    # concept against whatever was promoted at that moment — usually stale/sparse
    # and wasteful, since rejected concepts get linked too).
    link_at_extraction: bool = False

    # ── Stage 3 batch linking (opt-in) ────────────────────────────────────────
    # When enabled, all of a paper's Stage-3 LLM linking calls are submitted as a
    # single Message Batch (50% cheaper, processed server-side in parallel) and
    # polled to completion. Off by default — preserves the synchronous per-concept
    # path. Note: the two-temperature validation second pass is skipped in batch
    # mode (single-pass routing only).
    link_use_batch_api: bool = False
    link_batch_poll_seconds: int = 30
    link_batch_timeout_seconds: int = 1800

    # ── Hydration concurrency ─────────────────────────────────────────────────
    notion_hydration_concurrency: int = 5

    # ── ArXiv sniper ──────────────────────────────────────────────────────────
    arxiv_keywords: str = "Mean Field Games,Master Equation"
    arxiv_relevance_threshold: int = 8

    # ── Dependency grapher ────────────────────────────────────────────────────
    graph_static_dir: str = "/app/static"

    # ── Edge caps per relation type ───────────────────────────────────────────
    # Not individually configurable via env; override by subclassing Config.
    edge_caps: dict[str, int] = {
        "depends_on": 3,
        "enables": 3,
        "generalizes": 2,
        "special_case_of": 2,
        "related": 5,
    }

    @field_validator("vector_embedding_backend")
    @classmethod
    def _validate_embedding_backend(cls, v: str) -> str:
        allowed = {"openai", "local"}
        if v.lower() not in allowed:
            raise ValueError(f"vector_embedding_backend must be one of {allowed}, got {v!r}")
        return v.lower()

    @model_validator(mode="after")
    def _validate_weights_sum(self) -> "Config":
        total = (
            self.weight_qdrant
            + self.weight_named_tool
            + self.weight_assumption_overlap
            + self.weight_setting_containment
            + self.weight_keyword_jaccard
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Candidate scoring weights must sum to 1.0, got {total:.6f}. "
                "Check WEIGHT_QDRANT, WEIGHT_NAMED_TOOL, WEIGHT_ASSUMPTION_OVERLAP, "
                "WEIGHT_SETTING_CONTAINMENT, WEIGHT_KEYWORD_JACCARD."
            )
        return self


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config instance, parsed from .env and environment."""
    return Config()

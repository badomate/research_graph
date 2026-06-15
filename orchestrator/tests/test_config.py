"""Tests for modules/config.py — Config validation.

The app now boots with zero configuration (SQLite + sensible defaults); external
service credentials are optional. Only the candidate-scoring weight-sum invariant
is still enforced.
"""
import os
import sys

import pytest
from pydantic import ValidationError


_MANAGED_ENVS = [
    "DATABASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_FAST_MODEL",
    "QDRANT_URL", "VECTOR_EMBEDDING_BACKEND", "RETRIEVE_CANDIDATES_K",
    "ENABLE_VECTOR_INDEX", "VECTOR_INDEX_ENABLED", "MARKER_API_URL",
    "ZOTERO_POLL_ENABLED", "ZOTERO_POLL_MINUTES",
    "WEIGHT_QDRANT", "WEIGHT_NAMED_TOOL", "WEIGHT_ASSUMPTION_OVERLAP",
    "WEIGHT_SETTING_CONTAINMENT", "WEIGHT_KEYWORD_JACCARD",
]


def _make_config(env: dict | None = None):
    """Import Config fresh inside a controlled env, isolated from the real .env."""
    env = env or {}
    backup = {k: os.environ.get(k) for k in list(env.keys()) + _MANAGED_ENVS}
    try:
        for k in _MANAGED_ENVS:
            if k not in env:
                os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        sys.modules.pop("modules.config", None)
        from modules.config import Config
        return Config(_env_file=None)
    finally:
        for k, orig in backup.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig
        sys.modules.pop("modules.config", None)


class TestConfigValidation:
    def test_boots_with_zero_config(self):
        cfg = _make_config({})
        assert cfg.anthropic_api_key == ""          # optional now
        assert cfg.database_url.startswith("sqlite")

    def test_defaults(self):
        cfg = _make_config({})
        assert cfg.claude_model == "claude-sonnet-4-6"
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.retrieve_candidates_k == 30
        assert cfg.zotero_poll_enabled is False

    def test_database_url_override(self):
        cfg = _make_config({"DATABASE_URL": "sqlite:///./custom.db"})
        assert cfg.database_url == "sqlite:///./custom.db"

    def test_optional_fields_can_be_overridden(self):
        cfg = _make_config({"RETRIEVE_CANDIDATES_K": "50"})
        assert cfg.retrieve_candidates_k == 50

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            _make_config({"WEIGHT_QDRANT": "0.99", "WEIGHT_NAMED_TOOL": "0.99"})

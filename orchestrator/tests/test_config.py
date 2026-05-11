"""Tests for modules/config.py — Config validation."""
import os
import pytest
from pydantic import ValidationError


_REQUIRED_FIELDS = {
    "NOTION_TOKEN": "tok",
    "NOTION_PAPER_TRACKER_DB_ID": "db1",
    "NOTION_KNOWLEDGE_INBOX_DB_ID": "db2",
    "NOTION_SECOND_BRAIN_DB_ID": "db3",
    "ANTHROPIC_API_KEY": "key",
    "KOOFR_USER": "user",
    "KOOFR_APP_PASSWORD": "pass",
    "ZOTERO_USER_ID": "zid",
    "ZOTERO_API_KEY": "zkey",
}


_OPTIONAL_ENVS = [
    "NOTION_EDGES_DB_ID", "NOTION_DEFERRED_EDGES_DB_ID", "NOTION_PROJECTS_DB_ID",
    "CLAUDE_MODEL", "CLAUDE_FAST_MODEL", "QDRANT_URL", "VECTOR_EMBEDDING_BACKEND",
    "RETRIEVE_CANDIDATES_K", "ENABLE_VECTOR_INDEX", "MARKER_API_URL",
    "PIPELINE_TMP_DIR", "TOKEN_THRESHOLD_CHUNK", "NAMED_TOOL_MATCH_THRESHOLD",
    "WEIGHT_QDRANT", "WEIGHT_NAMED_TOOL", "WEIGHT_ASSUMPTION_OVERLAP",
    "WEIGHT_SETTING_CONTAINMENT", "WEIGHT_KEYWORD_JACCARD",
    "EDGE_AUTO_CREATE_CONFIDENCE", "EDGE_REVIEW_FLAG_CONFIDENCE",
]


def _make_config(env: dict):
    """Import Config fresh inside a controlled env, isolated from real .env."""
    import sys
    # Back up everything we'll touch (supplied keys + optionals we clear).
    all_keys = list(env.keys()) + _OPTIONAL_ENVS
    backup = {k: os.environ.get(k) for k in all_keys}
    try:
        # Clear optional vars so real .env values don't bleed in.
        for k in _OPTIONAL_ENVS:
            if k not in env:
                os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        if "modules.config" in sys.modules:
            del sys.modules["modules.config"]
        from modules.config import Config
        return Config(_env_file=None)
    finally:
        for k, orig in backup.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig
        if "modules.config" in sys.modules:
            del sys.modules["modules.config"]


class TestConfigValidation:
    def test_raises_when_notion_token_missing(self):
        env = {k: v for k, v in _REQUIRED_FIELDS.items() if k != "NOTION_TOKEN"}
        env.pop("NOTION_TOKEN", None)
        # Ensure key is absent
        os.environ.pop("NOTION_TOKEN", None)
        with pytest.raises(ValidationError):
            _make_config(env)

    def test_raises_when_anthropic_key_missing(self):
        env = {k: v for k, v in _REQUIRED_FIELDS.items() if k != "ANTHROPIC_API_KEY"}
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValidationError):
            _make_config(env)

    def test_loads_with_all_required_fields(self):
        cfg = _make_config(_REQUIRED_FIELDS)
        assert cfg.notion_token == "tok"
        assert cfg.anthropic_api_key == "key"
        assert cfg.koofr_user == "user"
        assert cfg.zotero_user_id == "zid"

    def test_defaults(self):
        cfg = _make_config(_REQUIRED_FIELDS)
        assert cfg.claude_model == "claude-sonnet-4-6"
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.retrieve_candidates_k == 30
        assert cfg.notion_edges_db_id == ""

    def test_optional_fields_can_be_overridden(self):
        env = {**_REQUIRED_FIELDS, "RETRIEVE_CANDIDATES_K": "50", "NOTION_EDGES_DB_ID": "edges_db"}
        cfg = _make_config(env)
        assert cfg.retrieve_candidates_k == 50
        assert cfg.notion_edges_db_id == "edges_db"

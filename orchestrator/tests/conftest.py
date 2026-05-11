"""
conftest.py — Stubs for packages not installed in the local test environment.

The orchestrator runs inside Docker where all dependencies are present.
Outside Docker, pure-module tests (edge_parser, block_builders, properties,
logging_utils, candidate_scorer) must import without triggering transitive
imports of missing packages. This file stubs them before collection begins.
"""
import sys
from unittest.mock import MagicMock

_MISSING = [
    # Notion API
    "notion_client",
    "notion_client.errors",
    # HTTP / WebDAV
    "requests",
    "requests.adapters",
    "webdav3",
    "webdav3.client",
    # Anthropic / LLM
    "anthropic",
    "instructor",
    # Graph
    "pyvis",
    "pyvis.network",
    "networkx",
    # Templating
    "jinja2",
    # Vector index
    "qdrant_client",
    "qdrant_client.models",
    # Fuzzy matching — added below with numeric return values
    # "rapidfuzz",
    # "rapidfuzz.fuzz",
    # Scheduler / feedparser
    "APScheduler",
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.background",
    "feedparser",
    # Retry
    "tenacity",
    # Embeddings
    "openai",
    "sentence_transformers",
    "tiktoken",
]

for _mod in _MISSING:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

try:
    import rapidfuzz  # noqa: F401
except ImportError:
    # rapidfuzz.fuzz methods must return integers for comparisons to work.
    _fuzz_mock = MagicMock()
    _fuzz_mock.token_sort_ratio.return_value = 0
    _fuzz_mock.partial_ratio.return_value = 0
    _rapidfuzz_mock = MagicMock()
    _rapidfuzz_mock.fuzz = _fuzz_mock
    sys.modules["rapidfuzz"] = _rapidfuzz_mock
    sys.modules["rapidfuzz.fuzz"] = _fuzz_mock

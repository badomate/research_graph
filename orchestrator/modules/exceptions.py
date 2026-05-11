"""
modules/exceptions.py — Domain exception hierarchy
────────────────────────────────────────────────────
All pipeline-specific exceptions inherit from PipelineError so callers
can catch the entire family with `except PipelineError` when needed.

Usage:
    from modules.exceptions import ExtractionError, KoofrError
    raise ExtractionError("Claude returned 0 concepts")
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all paper_pipeline errors."""


class ConfigurationError(PipelineError):
    """Missing or invalid configuration (env var or Config field)."""


class NotionError(PipelineError):
    """Notion API error after all retries are exhausted."""


class ZoteroError(PipelineError):
    """Zotero API error (attachment resolution, note fetch)."""


class KoofrError(PipelineError):
    """Koofr / WebDAV storage error (download, upload, existence check)."""


class MarkerError(PipelineError):
    """Marker OCR API error (PDF-to-Markdown conversion)."""


class ExtractionError(PipelineError):
    """Stage 1 LLM extraction failure (zero concepts, parse error)."""


class RetrievalError(PipelineError):
    """Stage 2 candidate retrieval failure (Qdrant + TF-IDF both failed)."""


class LinkingError(PipelineError):
    """Stage 3 LLM linking failure (edge proposal parse error)."""


class PromotionError(PipelineError):
    """Concept promotion failure (KI → Second Brain)."""


class VectorIndexError(PipelineError):
    """Qdrant vector index error (upsert, query, rebuild)."""


class EdgeParseError(PipelineError):
    """Edge Suggestions JSON could not be parsed in any known format."""

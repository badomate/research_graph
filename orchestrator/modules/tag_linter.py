"""
modules/tag_linter.py — Tag registry loader and lint validator
──────────────────────────────────────────────────────────────
Provides:
  - TagRegistry : loads and indexes tags_registry.yaml
  - TagLinter   : validates a list of tag strings against the registry
  - LintReport  : structured result dataclass
  - lint_report_to_text() : formats a LintReport for Notion rich_text storage
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_VALID_TAG_RE = re.compile(r"^(d|m|k)-[a-z0-9-]+$")
_DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent / "tags_registry.yaml"

# Notion rich_text property hard limit and safe truncation ceiling.
_NOTION_RICH_TEXT_LIMIT = 2000
_SAFE_TRUNCATE_LIMIT = _NOTION_RICH_TEXT_LIMIT - 10


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class LintReport:
    """Result of a lint pass on a list of tags."""

    valid_tags: list[str] = field(default_factory=list)
    """Tags that are present in the registry and not deprecated."""

    invalid_tags: list[str] = field(default_factory=list)
    """Tags that fail format validation or are not in the registry."""

    synonym_mappings: dict[str, str] = field(default_factory=dict)
    """Mapping of synonym → canonical id for tags that were auto-corrected."""

    deprecated_tags: list[str] = field(default_factory=list)
    """Tags that are in the registry but marked deprecated."""

    errors: list[str] = field(default_factory=list)
    """Human-readable error messages for every problem found."""


# ── Registry ──────────────────────────────────────────────────────────────────


class TagRegistry:
    """
    Loads and indexes tags_registry.yaml.

    The registry path is resolved in this order:
      1. ``TAGS_REGISTRY_PATH`` environment variable (absolute or relative to cwd).
      2. Default path: ``../../../tags_registry.yaml`` relative to this file.

    After loading, three indexes are built for O(1) look-up:
      - ``_by_id``      : id → tag dict
      - ``_synonyms``   : synonym string (lowercased) → canonical id
      - ``_deprecated`` : set of deprecated ids
    """

    def __init__(self) -> None:
        registry_path_str = os.environ.get("TAGS_REGISTRY_PATH", "")
        if registry_path_str:
            registry_path = Path(registry_path_str)
            if not registry_path.is_absolute():
                registry_path = Path.cwd() / registry_path
        else:
            registry_path = _DEFAULT_REGISTRY_PATH

        logger.debug("TagRegistry: loading from %s", registry_path)
        self._path = registry_path
        self._by_id: dict[str, dict[str, Any]] = {}
        self._synonyms: dict[str, str] = {}
        self._deprecated: set[str] = set()

        self._load(registry_path)

    def _load(self, path: Path) -> None:
        """Parse the YAML file and build internal indexes."""
        if not path.exists():
            logger.warning("TagRegistry: registry file not found at %s — linter will reject all tags.", path)
            return

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        tags: list[dict[str, Any]] = raw.get("tags", [])
        for entry in tags:
            tag_id: str = entry.get("id", "")
            if not tag_id:
                logger.warning("TagRegistry: skipping entry with missing id: %s", entry)
                continue

            self._by_id[tag_id] = entry

            if entry.get("deprecated", False):
                self._deprecated.add(tag_id)

            for syn in entry.get("synonyms", []):
                self._synonyms[syn.lower()] = tag_id

        logger.debug(
            "TagRegistry: loaded %d tag(s), %d synonym(s), %d deprecated.",
            len(self._by_id),
            len(self._synonyms),
            len(self._deprecated),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, tag: str) -> str | None:
        """
        Return the canonical id for *tag*, or None if unknown.

        Lookup order:
          1. Exact match in _by_id.
          2. Synonym match (case-insensitive).
        """
        if tag in self._by_id:
            return tag
        return self._synonyms.get(tag.lower())

    def is_deprecated(self, tag_id: str) -> bool:
        """Return True if *tag_id* is marked deprecated in the registry."""
        return tag_id in self._deprecated

    def __len__(self) -> int:
        return len(self._by_id)


# ── Linter ────────────────────────────────────────────────────────────────────


class TagLinter:
    """
    Validates a list of tag strings against the registry.

    Usage::

        registry = TagRegistry()
        linter = TagLinter(registry)
        report = linter.lint(["d-mfg", "MFG", "bad-tag"])
    """

    def __init__(self, registry: TagRegistry | None = None) -> None:
        self._registry = registry or TagRegistry()

    def lint(self, tags: list[str]) -> LintReport:
        """
        Validate *tags* and return a :class:`LintReport`.

        Checks performed (in order):
          1. Format check: prefix must be d-/m-/k-, chars [a-z0-9-] only.
          2. Synonym resolution: if tag matches a known synonym, map to
             canonical id and record in ``synonym_mappings``.
          3. Registry membership: tag (or resolved id) must be in registry.
          4. Deprecated flag: warn if tag is deprecated.
        """
        report = LintReport()

        for raw_tag in tags:
            tag = raw_tag.strip()
            if not tag:
                continue

            # ── Step 1: Format check ──────────────────────────────────────────
            if not _VALID_TAG_RE.match(tag):
                # Try synonym resolution before rejecting on format
                canonical = self._registry.resolve(tag)
                if canonical is None:
                    report.invalid_tags.append(tag)
                    report.errors.append(
                        f"'{tag}' fails format check (expected d-/m-/k- prefix, "
                        f"chars [a-z0-9-]) and is not a known synonym."
                    )
                    continue
                # It's a synonym — record the mapping
                report.synonym_mappings[tag] = canonical
                tag = canonical

            # ── Step 2: Synonym / registry look-up ───────────────────────────
            canonical = self._registry.resolve(tag)
            if canonical is None:
                report.invalid_tags.append(tag)
                report.errors.append(
                    f"'{tag}' passes format check but is not registered in the tag registry."
                )
                continue

            if canonical != tag:
                report.synonym_mappings[tag] = canonical
                tag = canonical

            # ── Step 3: Deprecated check ──────────────────────────────────────
            if self._registry.is_deprecated(canonical):
                report.deprecated_tags.append(canonical)
                report.errors.append(
                    f"'{canonical}' is deprecated. Please use the replacement tag."
                )
                # Still count as valid (downstream can decide what to do)

            if canonical not in report.valid_tags:
                report.valid_tags.append(canonical)

        return report


# ── Formatting helpers ────────────────────────────────────────────────────────


def lint_report_to_text(report: LintReport) -> str:
    """
    Serialise a :class:`LintReport` to a plain-text string suitable for
    storage in a Notion rich_text property.

    Kept under 2000 characters (Notion's hard limit) by truncating
    the error list if necessary.
    """
    lines: list[str] = []

    if report.valid_tags:
        lines.append(f"Valid: {', '.join(report.valid_tags)}")
    else:
        lines.append("Valid: (none)")

    if report.invalid_tags:
        lines.append(f"Invalid: {', '.join(report.invalid_tags)}")

    if report.synonym_mappings:
        mappings = "; ".join(f"{k}→{v}" for k, v in report.synonym_mappings.items())
        lines.append(f"Synonyms mapped: {mappings}")

    if report.deprecated_tags:
        lines.append(f"Deprecated: {', '.join(report.deprecated_tags)}")

    if report.errors:
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  • {err}")

    text = "\n".join(lines)
    # Truncate to Notion's 2000-char limit with a trailing ellipsis
    if len(text) > _SAFE_TRUNCATE_LIMIT:
        text = text[:_SAFE_TRUNCATE_LIMIT - 3] + "…"
    return text

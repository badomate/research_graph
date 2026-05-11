"""
modules/notion/properties.py — Notion property name constants
──────────────────────────────────────────────────────────────
Single source of truth for every Notion DB property name used by the
pipeline. Using these constants instead of bare string literals means
a rename in Notion only requires a change in this file.

Usage:
    from modules.notion.properties import KIProps, SBProps
    notion.update_page(ki_id, {KIProps.GRAPH_LINK_STATUS: ...})
"""

from __future__ import annotations


class PaperTrackerProps:
    """Property names for the Paper Tracker database."""

    STATUS = "Status"
    ZOTERO_URI = "Zotero URI"
    ZOTERO_ATTACHMENT_KEY = "Zotero Attachment Key"
    PRIMARY_PDF_FILENAME = "primary_pdf_filename"
    PDF_SHA256 = "PDF SHA256"
    ONE_LINER = "One Liner"
    ACTIVE_THEMES = "Active Themes"
    AI_STATUS = "AI Status"
    EXTRACTION_VERSION = "Extraction Version"
    EXTRACTION_COUNT = "Extraction Count"
    EXTRACTION_TOKENS = "Extraction Tokens"
    EXTRACTION_ERROR = "Extraction Error"
    LAST_RUN_ID = "Last Run ID"
    RE_EXTRACT_HINTS = "Re-extract Hints"
    REJECTED_CONCEPTS = "Rejected Concepts"
    PROCESSED_AT = "Processed At"

    # Status select values
    class Status:
        INBOX = "s0-inbox"
        SKIM = "s1-skim"
        PROCESSING = "s1-processing"
        WAITING_ATTACHMENT = "s1b-waiting-attachment"
        BLOCKED_EXTRACTION = "blocked-extraction"
        EXTRACTED = "s2-extracted"
        REEXTRACT = "s2-reextract"
        READ = "s2-read"
        DISTILLED = "s3-distilled"


class KIProps:
    """Property names for the Knowledge Inbox database."""

    STATUS = "Status"
    TYPE = "Type"
    VERIFICATION_STATUS = "verification_status"
    GRAPH_LINK_STATUS = "Graph Link Status"
    SOURCE_PAPER = "Source Paper"
    SOURCE_PAGES = "Source Pages"
    SUGGESTED_HUB = "Suggested Hub"
    AI_CONFIDENCE = "AI Confidence"
    KEYWORDS = "Keywords"
    PREREQ_KEYWORDS = "Prereq Keywords"
    DOWNSTREAM_KEYWORDS = "Downstream Keywords"
    SOURCE_ANCHORS = "Source Anchors"
    INTERPRETATION = "Interpretation"
    PROOF_IDEA = "Proof Idea"
    ALIASES = "Aliases"
    ASSUMPTIONS = "Assumptions"
    STATEMENT_LATEX = "Statement LaTeX"
    SOURCE_QUOTE = "Source Quote"
    NAMED_TOOLS = "Named Tools"
    SETTING = "Setting"
    RESULT_CATEGORY = "Result Category"
    EDGE_SUGGESTIONS = "Edge Suggestions"
    CANDIDATE_MATCHES = "Candidate Matches"
    CORRECTED_TITLE = "Corrected Title"
    PROMOTION_TARGET = "Promotion Target"

    # verification_status values
    class VerificationStatus:
        VERIFIED = "verified"
        REJECTED = "rejected"
        FLAGGED = "flagged"

    # Graph Link Status values
    class GraphLinkStatus:
        UNLINKED = "unlinked"
        LINKED_AI = "linked-ai"


class SBProps:
    """Property names for the Second Brain database."""

    NOTE_LEVEL = "Note Level"
    TYPE = "Type"
    VERIFIED = "Verified"
    LAST_VERIFIED_AT = "Last Verified At"
    SOURCES = "Sources"
    SOURCE_PAGES = "Source Pages"
    SOURCE_ANCHORS = "Source Anchors"
    STATEMENT_LATEX = "Statement LaTeX"
    ASSUMPTIONS = "Assumptions"
    INTERPRETATION = "Interpretation"
    PROOF_IDEA = "Proof Idea"
    ALIASES = "Aliases"
    NAMED_TOOLS = "Named Tools"
    KEYWORDS = "Keywords"
    PREREQ_KEYWORDS = "Prereq Keywords"
    DOWNSTREAM_KEYWORDS = "Downstream Keywords"
    SETTING = "Setting"
    RESULT_CATEGORY = "Result Category"


class EdgeProps:
    """Property names for the Edges database."""

    FROM_CONCEPT = "From Concept"
    TO_CONCEPT = "To Concept"
    RELATION_TYPE = "Relation Type"
    RATIONALE = "Rationale"
    AI_CONFIDENCE = "AI Confidence"
    CREATED_BY = "Created By"
    STATUS = "Status"
    SOURCE_PAPERS = "Source Papers"
    NEEDS_REVIEW = "needs_review"
    CHANNEL = "channel"
    DRIVING_FIELDS = "driving_fields"
    FALSIFIABILITY = "falsifiability"
    JUSTIFICATION = "justification"
    PRE_FILTER_SIGNAL = "pre_filter_signal"


class DeferredEdgeProps:
    """Property names for the Deferred Edges database."""

    STATUS = "Status"
    FROM_CONCEPT = "From Concept"
    TARGET_TITLE = "Target Title"
    AI_CONFIDENCE = "AI Confidence"
    RELATION_TYPE = "Relation Type"
    RATIONALE = "Rationale"
    SOURCE_PAPERS = "Source Papers"

    class Status:
        PENDING = "pending"
        RESOLVED = "resolved"
        STALE = "stale"


class ProjectProps:
    """Property names for the Projects (Hubs) database."""

    HUB = "Hub"


# ── Shared property accessor helpers ──────────────────────────────────────────

def get_page_title(page: dict) -> str:
    """Extract plain-text title from a raw Notion page object."""
    for value in page.get("properties", {}).values():
        if value.get("type") == "title":
            try:
                return value["title"][0]["plain_text"]
            except (KeyError, IndexError):
                return ""
    return ""


def get_text_prop(props: dict, key: str) -> str:
    """Extract plain text from a Notion rich_text or url property."""
    prop = props.get(key, {})
    if prop.get("type") == "url":
        return prop.get("url") or ""
    try:
        return prop["rich_text"][0]["plain_text"]
    except (KeyError, IndexError):
        return ""


def get_title_prop(props: dict) -> str:
    """Extract plain text from a Notion title property."""
    for value in props.values():
        if value.get("type") == "title":
            try:
                return value["title"][0]["plain_text"]
            except (KeyError, IndexError):
                return "unknown"
    return "unknown"


def get_multi_select_prop(props: dict, key: str) -> list[str]:
    """Extract option names from a Notion multi_select property."""
    try:
        return [opt["name"] for opt in props[key]["multi_select"]]
    except (KeyError, TypeError):
        return []


def get_select_prop(props: dict, key: str) -> str:
    """Extract the selected option name from a Notion select property."""
    try:
        return props[key]["select"]["name"]
    except (KeyError, TypeError):
        return ""


def get_number_prop(props: dict, key: str) -> float | None:
    """Extract a number from a Notion number property."""
    try:
        return props[key]["number"]
    except (KeyError, TypeError):
        return None

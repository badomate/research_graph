"""Tests for modules/notion/properties.py — property name constants."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.notion.properties import (
    DeferredEdgeProps,
    EdgeProps,
    KIProps,
    PaperTrackerProps,
    ProjectProps,
    SBProps,
    get_multi_select_prop,
    get_number_prop,
    get_page_title,
    get_select_prop,
    get_text_prop,
)


class TestKIProps:
    def test_edge_suggestions(self):
        assert KIProps.EDGE_SUGGESTIONS == "Edge Suggestions"

    def test_verification_status(self):
        assert KIProps.VERIFICATION_STATUS == "verification_status"

    def test_promotion_target(self):
        assert KIProps.PROMOTION_TARGET == "Promotion Target"

    def test_source_paper(self):
        assert KIProps.SOURCE_PAPER == "Source Paper"

    def test_status_values(self):
        assert KIProps.VerificationStatus.VERIFIED == "verified"
        assert KIProps.VerificationStatus.REJECTED == "rejected"
        assert KIProps.VerificationStatus.FLAGGED == "flagged"


class TestSBProps:
    def test_note_level(self):
        assert SBProps.NOTE_LEVEL == "Note Level"

    def test_statement_latex(self):
        assert SBProps.STATEMENT_LATEX == "Statement LaTeX"

    def test_sources(self):
        assert SBProps.SOURCES == "Sources"

    def test_last_verified_at(self):
        assert SBProps.LAST_VERIFIED_AT == "Last Verified At"


class TestEdgeProps:
    def test_from_concept(self):
        assert EdgeProps.FROM_CONCEPT == "From Concept"

    def test_to_concept(self):
        assert EdgeProps.TO_CONCEPT == "To Concept"

    def test_relation_type(self):
        assert EdgeProps.RELATION_TYPE == "Relation Type"

    def test_ai_confidence(self):
        assert EdgeProps.AI_CONFIDENCE == "AI Confidence"


class TestPaperTrackerProps:
    def test_status(self):
        assert PaperTrackerProps.STATUS == "Status"

    def test_pdf_sha256(self):
        assert PaperTrackerProps.PDF_SHA256 == "PDF SHA256"

    def test_status_values(self):
        assert PaperTrackerProps.Status.SKIM == "s1-skim"
        assert PaperTrackerProps.Status.EXTRACTED == "s2-extracted"
        assert PaperTrackerProps.Status.DISTILLED == "s3-distilled"


class TestDeferredEdgeProps:
    def test_target_title(self):
        assert DeferredEdgeProps.TARGET_TITLE == "Target Title"

    def test_status_values(self):
        assert DeferredEdgeProps.Status.PENDING == "pending"
        assert DeferredEdgeProps.Status.RESOLVED == "resolved"
        assert DeferredEdgeProps.Status.STALE == "stale"


class TestPropertyAccessorHelpers:
    def _page(self, title_text: str) -> dict:
        return {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": title_text}],
                }
            }
        }

    def test_get_page_title_extracts_text(self):
        page = self._page("Gradient Descent")
        assert get_page_title(page) == "Gradient Descent"

    def test_get_page_title_empty_when_missing(self):
        assert get_page_title({"properties": {}}) == ""

    def test_get_text_prop_rich_text(self):
        props = {
            "Summary": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "A summary."}],
            }
        }
        assert get_text_prop(props, "Summary") == "A summary."

    def test_get_text_prop_missing_key(self):
        assert get_text_prop({}, "Missing") == ""

    def test_get_select_prop(self):
        props = {"Type": {"select": {"name": "Theorem"}}}
        assert get_select_prop(props, "Type") == "Theorem"

    def test_get_select_prop_missing(self):
        assert get_select_prop({}, "Type") == ""

    def test_get_multi_select_prop(self):
        props = {"Keywords": {"multi_select": [{"name": "convex"}, {"name": "lipschitz"}]}}
        assert get_multi_select_prop(props, "Keywords") == ["convex", "lipschitz"]

    def test_get_multi_select_prop_empty(self):
        assert get_multi_select_prop({}, "Keywords") == []

    def test_get_number_prop(self):
        props = {"AI Confidence": {"number": 0.85}}
        assert get_number_prop(props, "AI Confidence") == 0.85

    def test_get_number_prop_none_when_missing(self):
        assert get_number_prop({}, "AI Confidence") is None

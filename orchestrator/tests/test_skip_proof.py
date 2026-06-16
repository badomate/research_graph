"""Tests for the skip-proof scope suggester (pure)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.parsing import skip_proof as sp


def test_classify_keep_vs_skip():
    assert sp.classify_heading("Introduction")[0] == "keep"
    assert sp.classify_heading("Assumptions and main results")[0] == "keep"
    assert sp.classify_heading("Numerical experiments")[0] == "keep"
    assert sp.classify_heading("Proof of Theorem 2")[0] == "skip"
    assert sp.classify_heading("Appendix B: technical lemmas")[0] == "skip"
    assert sp.classify_heading("References")[0] == "skip"


def test_skip_beats_keep_when_both_match():
    # Contains "theorem" (keep) but is a proof (skip) — skip must win.
    decision, reason = sp.classify_heading("Proof of the main theorem")
    assert decision == "skip" and reason == "proof section"


def test_unknown_heading_kept_by_default():
    assert sp.classify_heading("On some curious phenomena")[0] == "keep"


def test_parse_section_lines_variants():
    text = "1-2 Abstract\n3 Introduction\n9-30: Proof of Theorem 1\ngarbage line\n"
    secs = sp.parse_section_lines(text)
    assert [(s.page_start, s.page_end, s.heading) for s in secs] == [
        (1, 2, "Abstract"), (3, 3, "Introduction"), (9, 30, "Proof of Theorem 1")
    ]


def test_propose_scope_skips_proofs_and_appendix():
    secs = sp.parse_section_lines(
        "1-2 Abstract\n3-8 Introduction\n9-12 Assumptions and main results\n"
        "13-34 Proof of the main theorem\n35-38 Numerical experiments\n"
        "39 Conclusion\n40-50 Appendix A\n"
    )
    prop = sp.propose_scope(secs, total_pages=50)
    assert prop["scope_json"]["page_ranges"] == [[1, 12], [35, 39]]
    assert 13 in prop["skipped_pages"] and 45 in prop["skipped_pages"]
    assert sp.ranges_to_string(prop["scope_json"]["page_ranges"]) == "1-12, 35-39"


def test_propose_scope_empty():
    prop = sp.propose_scope([], total_pages=10)
    assert prop["selected_pages"] == [] and prop["scope_json"]["page_ranges"] == []


class _Chunk:
    def __init__(self, heading, pf, pt):
        self.heading, self.page_from, self.page_to = heading, pf, pt


def test_sections_from_chunks():
    chunks = [_Chunk("Introduction", 1, 3), _Chunk("Proof of Lemma 1", 4, 9), _Chunk("", 10, 10)]
    secs = sp.sections_from_chunks(chunks)
    assert len(secs) == 2  # the empty-heading chunk is dropped
    prop = sp.propose_scope(secs)
    assert prop["scope_json"]["page_ranges"] == [[1, 3]]

"""Tests for parse-scope page math (pure helpers)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.parsing import scope_utils


def test_full_scope_enumerates_with_total():
    assert scope_utils.selected_pages({"kind": "full"}, total_pages=3) == [1, 2, 3]
    assert scope_utils.selected_pages({"kind": "full"}) == []


def test_page_ranges_union_and_dedup():
    scope = {"kind": "page_range", "page_ranges": [[1, 8], [35, 40], [3, 5]]}
    pages = scope_utils.selected_pages(scope, total_pages=50)
    assert pages == list(range(1, 9)) + list(range(35, 41))
    assert scope_utils.page_count(scope, 50) == 14


def test_reversed_range_is_normalized_and_clamped():
    scope = {"kind": "page_range", "page_ranges": [[8, 1]]}
    assert scope_utils.selected_pages(scope, total_pages=5) == [1, 2, 3, 4, 5]


def test_mixed_includes_region_pages():
    scope = {
        "kind": "mixed",
        "page_ranges": [[1, 2]],
        "regions": [{"page": 7, "bbox": [0.1, 0.1, 0.5, 0.5], "label": "theorem"}],
    }
    assert scope_utils.selected_pages(scope, total_pages=10) == [1, 2, 7]


def test_to_marker_page_range_is_zero_indexed_compact():
    assert scope_utils.to_marker_page_range([1, 2, 3, 5, 8, 9]) == "0-2,4,7-8"
    assert scope_utils.to_marker_page_range([]) == ""


def test_scope_summary_reports_skipped():
    scope = {"kind": "page_range", "page_ranges": [[1, 8]]}
    summary = scope_utils.scope_summary(scope, total_pages=40)
    assert summary["selected_count"] == 8
    assert summary["skipped_count"] == 32
    assert summary["skipped_pages"][0] == 9


def test_bbox_to_cropbox_top_left_origin_flips_y():
    # A 100x200 page (origin 0,0). Top-left quarter: x∈[0,0.5], y∈[0,0.5] (top half).
    mediabox = (0.0, 0.0, 100.0, 200.0)
    llx, lly, urx, ury = scope_utils.bbox_to_cropbox(mediabox, [0.0, 0.0, 0.5, 0.5])
    assert (llx, urx) == (0.0, 50.0)
    # top-left y0=0 → ury at the page top (200); y1=0.5 → lly at the middle (100).
    assert (lly, ury) == (100.0, 200.0)


def test_bbox_to_cropbox_respects_mediabox_offset_and_clamps():
    mediabox = (10.0, 20.0, 110.0, 220.0)        # offset origin, 100x200
    llx, lly, urx, ury = scope_utils.bbox_to_cropbox(mediabox, [-0.1, 0.5, 1.2, 1.0])
    assert llx == 10.0 and urx == 110.0          # x clamped to [0,1] → full width
    assert lly == 20.0                           # y1=1.0 → page bottom (20)
    assert ury == 120.0                          # y0=0.5 → mid (20 + 0.5*200)

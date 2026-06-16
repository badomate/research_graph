"""Tests for the cost estimator."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import cost


def test_marker_cost_scales_with_pages():
    est = cost.estimate_marker_cost(14, 0.01)
    assert est.pages == 14
    assert est.cost == 0.14


def test_marker_cost_clamps_negative():
    assert cost.estimate_marker_cost(-3, 0.01).pages == 0


def test_claude_estimate_has_band_and_overhead():
    est = cost.estimate_claude_cost(
        input_tokens=45_000, analysis_type="novelty_risk",
        input_price_per_mtok=3.0, output_price_per_mtok=15.0,
    )
    assert est.input_tokens == 45_000 + 1200       # includes system overhead
    assert est.cost_low < est.cost_mid < est.cost_high
    # input cost dominates here: 46.2k * $3/M ≈ $0.1386
    assert round(est.input_cost, 3) == 0.139


def test_unknown_analysis_type_uses_default_output_budget():
    est = cost.estimate_claude_cost(
        input_tokens=1000, analysis_type="does_not_exist",
        input_price_per_mtok=3.0, output_price_per_mtok=15.0,
    )
    assert est.output_tokens == cost._DEFAULT_OUTPUT_TOKENS


def test_actual_cost_matches_manual():
    c = cost.actual_claude_cost(
        input_tokens=1_000_000, output_tokens=1_000_000,
        input_price_per_mtok=3.0, output_price_per_mtok=15.0,
    )
    assert c == 18.0


def test_estimate_tokens_nonzero_for_text():
    assert cost.estimate_tokens("the quick brown fox " * 50) > 0
    assert cost.estimate_tokens("") == 0

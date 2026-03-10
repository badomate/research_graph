"""
tests/test_completeness.py
──────────────────────────
Unit tests for the Layer 1 Pipeline Completeness Gate.

Covers:
  - check_completeness: empty statement → reject
  - check_completeness: short statement → reject
  - check_completeness: empty conclusion → reject
  - check_completeness: low confidence → reject
  - check_completeness: Theorem with no assumptions → flag
  - check_completeness: low confidence (< 0.75) → flag
  - check_completeness: low keyword count → flag
  - check_completeness: Theorem with no named_tools → flag
  - check_completeness: well-formed concept → accept
  - MathObject.confidence coerced from string to float
  - is_dense_paper: correctly classifies dense vs. clean papers
"""

from __future__ import annotations

import sys
import os
import types
import unittest

# ── Minimal stubs so ingestion.py can be imported without real dependencies ──

def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub_module("anthropic")
_stub_module("instructor")
_stub_module("webdav3")
_stub_module("webdav3.client", Client=object)

tenacity_mod = _stub_module(
    "tenacity",
    retry=lambda **kw: (lambda fn: fn),
    stop_after_attempt=lambda n: None,
    wait_exponential=lambda **kw: None,
    retry_if_exception=lambda fn: None,
    retry_if_exception_type=lambda exc: None,
    stop_never=None,
    wait_fixed=lambda n: None,
)

_stub_module("notion_client", Client=object)
_stub_module("notion_client.errors", APIResponseError=Exception)

vi_mod = _stub_module("orchestrator.modules.vector_index")
vi_mod.VectorIndexEngine = type("VectorIndexEngine", (), {
    "available": property(lambda self: False),
    "retrieve_candidates": lambda *a, **kw: [],
})

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import code under test ────────────────────────────────────────────────────

from orchestrator.modules.extraction_schema import (
    MathObject,
    check_completeness,
)
from orchestrator.modules.ingestion import is_dense_paper


# ── Helper factories ──────────────────────────────────────────────────────────

def _make_concept(
    type: str = "Theorem",
    title: str = "Test Theorem",
    statement_latex: str = r"\[ f(x) = \int_0^T e^{-\rho t} L(x(t), u(t)) \, dt \]",
    assumptions: str = "Finite state/action spaces; $\\alpha \\in [0,1]$.",
    variables: str = "$x$ (state), $u$ (control)",
    conclusion: str = "The optimal value function satisfies the HJB equation.",
    confidence: float = 0.90,
    canonical_keywords: list | None = None,
    named_tools: list | None = None,
) -> MathObject:
    return MathObject(
        type=type,
        title=title,
        statement_latex=statement_latex,
        assumptions=assumptions,
        variables=variables,
        conclusion=conclusion,
        confidence=confidence,
        canonical_keywords=canonical_keywords if canonical_keywords is not None
        else ["optimal-control", "hjb-equation", "value-function"],
        named_tools=named_tools if named_tools is not None else ["Dynamic Programming"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: check_completeness — Reject conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCompletenessReject(unittest.TestCase):

    def test_empty_statement_is_rejected(self):
        """Empty statement_latex → reject."""
        concept = _make_concept(statement_latex="")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")
        self.assertTrue(any("empty" in r.lower() or "none" in r.lower()
                            for r in verdict.reasons))

    def test_none_statement_is_rejected(self):
        """statement_latex = 'None' → reject."""
        concept = _make_concept(statement_latex="None")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")

    def test_na_statement_is_rejected(self):
        """statement_latex = 'N/A' → reject."""
        concept = _make_concept(statement_latex="N/A")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")

    def test_short_statement_is_rejected(self):
        """statement_latex with < 40 stripped chars → reject."""
        concept = _make_concept(statement_latex=r"\[ f = 0 \]")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")
        self.assertTrue(any("short" in r.lower() for r in verdict.reasons))

    def test_empty_conclusion_is_rejected(self):
        """conclusion is empty → reject."""
        concept = _make_concept(conclusion="")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")
        self.assertTrue(any("conclusion" in r.lower() for r in verdict.reasons))

    def test_whitespace_conclusion_is_rejected(self):
        """conclusion with only whitespace → reject."""
        concept = _make_concept(conclusion="   ")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")

    def test_low_confidence_below_055_is_rejected(self):
        """confidence < 0.55 → reject."""
        concept = _make_concept(confidence=0.50)
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "reject")
        self.assertTrue(any("0.55" in r for r in verdict.reasons))

    def test_confidence_exactly_055_is_not_rejected(self):
        """confidence == 0.55 → should NOT be rejected on confidence alone."""
        concept = _make_concept(confidence=0.55)
        verdict = check_completeness(concept)
        # May be flagged (< 0.75) but must not be rejected
        self.assertNotEqual(verdict.status, "reject")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: check_completeness — Flag conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCompletenessFlag(unittest.TestCase):

    def test_theorem_with_no_assumptions_is_flagged(self):
        """Theorem with empty assumptions → flag."""
        concept = _make_concept(type="Theorem", assumptions="")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")
        self.assertTrue(
            any("assumption" in r.lower() for r in verdict.reasons)
        )

    def test_theorem_with_none_explicitly_stated_is_flagged(self):
        """Theorem with 'None explicitly stated.' → flag."""
        concept = _make_concept(
            type="Theorem", assumptions="None explicitly stated."
        )
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")

    def test_lemma_with_no_assumptions_is_flagged(self):
        """Lemma with empty assumptions → flag."""
        concept = _make_concept(type="Lemma", assumptions="")
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")

    def test_definition_with_no_assumptions_is_not_flagged_for_it(self):
        """
        Definition with empty assumptions is NOT flagged for missing
        assumptions (that rule only applies to Theorem/Lemma).
        """
        # With enough keywords and named_tools, high confidence — should accept.
        concept = _make_concept(
            type="Definition",
            assumptions="",
            confidence=0.90,
            canonical_keywords=["stochastic-control", "value-function", "mfg"],
            named_tools=["Dynamic Programming"],
        )
        verdict = check_completeness(concept)
        # Should accept since no assumption-flag rule applies to Definition
        self.assertEqual(verdict.status, "accept")

    def test_low_confidence_075_is_flagged(self):
        """confidence < 0.75 → flag (with all other fields good)."""
        concept = _make_concept(confidence=0.65)
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")
        self.assertTrue(any("0.75" in r for r in verdict.reasons))

    def test_confidence_exactly_075_is_not_flagged_for_confidence(self):
        """confidence == 0.75 → NOT flagged for confidence alone."""
        concept = _make_concept(
            confidence=0.75,
            canonical_keywords=["a", "b", "c"],
            named_tools=["Banach"],
        )
        verdict = check_completeness(concept)
        self.assertNotEqual(verdict.status, "reject")
        # Should not contain a confidence < 0.75 reason
        self.assertFalse(any("0.75" in r for r in verdict.reasons))

    def test_fewer_than_3_keywords_is_flagged(self):
        """Only 2 canonical_keywords → flag."""
        concept = _make_concept(
            canonical_keywords=["keyword-one", "keyword-two"],
        )
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")
        self.assertTrue(any("keyword" in r.lower() for r in verdict.reasons))

    def test_exactly_3_keywords_not_flagged_for_keywords(self):
        """Exactly 3 canonical_keywords → not flagged for keywords."""
        concept = _make_concept(
            confidence=0.90,
            canonical_keywords=["a", "b", "c"],
            named_tools=["Banach"],
        )
        verdict = check_completeness(concept)
        # Should not be flagged for keyword count
        self.assertFalse(any("keyword" in r.lower() for r in verdict.reasons))

    def test_theorem_with_no_named_tools_is_flagged(self):
        """Theorem with empty named_tools → flag."""
        concept = _make_concept(type="Theorem", named_tools=[])
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")
        self.assertTrue(any("named_tools" in r for r in verdict.reasons))

    def test_lemma_with_no_named_tools_is_flagged(self):
        """Lemma with empty named_tools → flag."""
        concept = _make_concept(type="Lemma", named_tools=[])
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "flag")

    def test_algorithm_with_no_named_tools_not_flagged_for_it(self):
        """Algorithm with empty named_tools is NOT flagged for it."""
        concept = _make_concept(
            type="Algorithm",
            named_tools=[],
            confidence=0.90,
            canonical_keywords=["policy-iteration", "mdp", "convergence"],
        )
        verdict = check_completeness(concept)
        # Should accept — no named_tools flag for Algorithm
        self.assertEqual(verdict.status, "accept")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: check_completeness — Accept condition
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCompletenessAccept(unittest.TestCase):

    def test_well_formed_theorem_is_accepted(self):
        """Well-formed Theorem concept → accept."""
        concept = _make_concept()
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "accept")
        self.assertEqual(verdict.reasons, [])

    def test_well_formed_definition_is_accepted(self):
        """Well-formed Definition with all fields → accept."""
        concept = _make_concept(
            type="Definition",
            assumptions="",  # assumptions not required for Definition
            canonical_keywords=["mfg", "equilibrium", "graphon"],
            named_tools=[],  # not required for Definition
            confidence=0.90,
        )
        verdict = check_completeness(concept)
        self.assertEqual(verdict.status, "accept")

    def test_verdict_status_is_a_string(self):
        """Verdict status is always a string."""
        concept = _make_concept()
        verdict = check_completeness(concept)
        self.assertIsInstance(verdict.status, str)
        self.assertIn(verdict.status, ("accept", "flag", "reject"))

    def test_verdict_reasons_is_a_list(self):
        """Verdict reasons is always a list."""
        concept = _make_concept()
        verdict = check_completeness(concept)
        self.assertIsInstance(verdict.reasons, list)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: MathObject confidence coercion
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceCoercion(unittest.TestCase):

    def test_string_confidence_coerced_to_float(self):
        """confidence given as a string is cast to float."""
        concept = MathObject(
            type="Theorem",
            title="Test",
            statement_latex=r"\[ f(x) = g(x) + \epsilon \quad \forall x \in \Omega \]",
            confidence="0.85",  # type: ignore[arg-type]
        )
        self.assertIsInstance(concept.confidence, float)
        self.assertAlmostEqual(concept.confidence, 0.85)

    def test_invalid_string_confidence_defaults_to_1(self):
        """confidence given as an unparseable string defaults to 1.0."""
        concept = MathObject(
            type="Theorem",
            title="Test",
            statement_latex=r"\[ f(x) = g(x) + \epsilon \quad \forall x \in \Omega \]",
            confidence="high",  # type: ignore[arg-type]
        )
        self.assertIsInstance(concept.confidence, float)
        self.assertAlmostEqual(concept.confidence, 1.0)

    def test_float_confidence_unchanged(self):
        """Float confidence is passed through unchanged."""
        concept = MathObject(
            type="Theorem",
            title="Test",
            statement_latex=r"\[ f(x) = g(x) + \epsilon \quad \forall x \in \Omega \]",
            confidence=0.72,
        )
        self.assertAlmostEqual(concept.confidence, 0.72)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: is_dense_paper
# ─────────────────────────────────────────────────────────────────────────────

class TestIsDensePaper(unittest.TestCase):

    # Dense paper: all three criteria met.
    _DENSE_MD = (
        r"\int_0^T e^{-\rho t} H(x, \nabla V) \, dt "
        r"+ \sum_{i=1}^N \lambda_i \phi_i(x) = 0 "
        r"\forall x \in \Omega, \quad (H1)-(H4) \text{ hold} "
        r"\mathbb{E}[X_t | \mathcal{F}_t] \leq C(t) "
        r"\partial_t V + H(x, \nabla V, m) = 0 "
        r"\mu_t \in \mathcal{P}(\mathbb{R}^d) "
    ) * 400   # Repeat to get enough tokens.

    # Clean paper: no LaTeX, no shorthands, not dense.
    _CLEAN_MD = "The algorithm converges if the learning rate is small enough. " * 1000

    def test_dense_paper_detected(self):
        """Paper with high LaTeX density and (H1) shorthands is dense."""
        token_count = len(self._DENSE_MD) // 4
        self.assertGreaterEqual(token_count, 15_000)
        result = is_dense_paper(self._DENSE_MD, token_count)
        self.assertTrue(result)

    def test_clean_paper_not_dense(self):
        """Plain English paper without LaTeX/shorthands is not dense."""
        token_count = len(self._CLEAN_MD) // 4
        result = is_dense_paper(self._CLEAN_MD, token_count)
        self.assertFalse(result)

    def test_below_token_threshold_not_dense(self):
        """Even a heavy paper is not dense if token_count < 15_000."""
        result = is_dense_paper(self._DENSE_MD, 10_000)
        self.assertFalse(result)

    def test_no_shorthand_not_dense(self):
        """Paper with high LaTeX density but no (H1) shorthand is not dense."""
        no_shorthand = (
            r"\int_0^T e^{-\rho t} H(x, \nabla V) \, dt "
            r"\forall x \in \Omega \mathbb{E}[X] \leq C "
        ) * 800
        token_count = len(no_shorthand) // 4
        self.assertGreaterEqual(token_count, 15_000)
        result = is_dense_paper(no_shorthand, token_count)
        self.assertFalse(result)

    def test_low_latex_density_not_dense(self):
        """Paper with (H1) shorthand but low LaTeX density is not dense."""
        sparse = ("Some text. " * 200) + " (H1)-(H4) hold. " + ("More text. " * 1000)
        token_count = max(len(sparse) // 4, 16_000)
        result = is_dense_paper(sparse, token_count)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

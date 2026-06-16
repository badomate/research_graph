"""
webapp/seed.py — load demo data so the UI is explorable before the pipeline cutover.

    DATABASE_URL=sqlite:///./app.db python -m webapp.seed
    # (DATABASE_URL defaults to sqlite:///./app.db if unset)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ORCH = (Path(__file__).resolve().parent.parent / "orchestrator").resolve()
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
os.environ.setdefault("DATABASE_URL", "sqlite:///./app.db")

from modules.store import (  # noqa: E402
    ConceptState,
    EdgeChannel,
    EdgeStatus,
    PaperStatus,
    Store,
    VerificationStatus,
)


def seed() -> None:
    store = Store()
    store.create_all()

    if store.list_papers():
        print("Database already has papers — skipping seed. "
              "Delete the DB file to reseed.")
        return

    # ── A promoted hub + concepts (populate Brain + Graph) ──────────────────────
    hub = store.upsert_hub("Mean Field Games")

    done_paper = store.create_paper(
        title="Existence and Uniqueness for McKean–Vlasov SDEs",
        authors="A. Researcher, B. Coauthor",
        arxiv_id="2009.01234",
        status=PaperStatus.S3_DISTILLED.value,
        one_liner="Well-posedness of McKean–Vlasov SDEs under monotone coefficients.",
        extraction_count=2,
    )
    mv = store.create_concept(
        paper_id=done_paper.id, state=ConceptState.PROMOTED.value, type="Theorem",
        title="Well-posedness of McKean–Vlasov SDEs", suggested_hub="Mean Field Games",
        statement_latex=r"$dX_t = b(X_t, \mathcal{L}(X_t))\,dt + \sigma\,dW_t$ admits a unique strong solution.",
        conclusion="Existence and uniqueness of a strong solution under Lipschitz drift.",
        canonical_keywords=["McKean-Vlasov", "well-posedness", "propagation of chaos"],
        verification_status=VerificationStatus.VERIFIED.value,
    )
    banach = store.create_concept(
        paper_id=done_paper.id, state=ConceptState.PROMOTED.value, type="ProofTechnique",
        title="Banach Fixed-Point Argument", suggested_hub="Mean Field Games",
        statement_latex=r"A contraction $T$ on a complete metric space has a unique fixed point.",
        canonical_keywords=["fixed point", "contraction", "Picard iteration"],
        verification_status=VerificationStatus.VERIFIED.value,
    )
    store.create_edge(
        source_concept_id=mv.id, target_concept_id=banach.id,
        relation_type="depends_on", status=EdgeStatus.VERIFIED.value,
        channel=EdgeChannel.AUTO.value, needs_review=False, ai_confidence=0.92,
        justification="The well-posedness proof builds the solution map as a contraction.",
    )
    store.create_edge(
        source_concept_id=mv.id, target_concept_id=hub.id,
        relation_type="related", status=EdgeStatus.VERIFIED.value, ai_confidence=0.8,
    )

    # ── A paper awaiting review (inbox concepts + proposed edges) ───────────────
    review_paper = store.create_paper(
        title="Mean Field Games and the Master Equation",
        authors="C. Author",
        arxiv_id="2401.05678",
        status=PaperStatus.S2_EXTRACTED.value,
        one_liner="The master equation characterises the MFG limit of N-player games.",
        extraction_count=3,
    )
    master = store.create_concept(
        paper_id=review_paper.id, type="Definition",
        title="The Master Equation", suggested_hub="Mean Field Games",
        statement_latex=r"$\partial_t U + H(x, D_x U) + \int \dots = 0$ on $[0,T]\times\mathbb{R}^d \times \mathcal{P}_2$.",
        assumptions="Sufficient regularity of the Hamiltonian $H$ and monotone couplings.",
        conclusion="A single PDE on the space of measures encoding the MFG system.",
        canonical_keywords=["master equation", "Wasserstein", "Hamilton-Jacobi"],
        ai_confidence=0.88,
    )
    monotone = store.create_concept(
        paper_id=review_paper.id, type="Assumption",
        title="Lasry–Lions Monotonicity", suggested_hub="Mean Field Games",
        statement_latex=r"$\int (f(x,m) - f(x,m'))\,d(m-m')(x) \ge 0$.",
        conclusion="Monotone couplings yield uniqueness of the MFG equilibrium.",
        canonical_keywords=["monotonicity", "uniqueness", "Lasry-Lions"],
        ai_confidence=0.79,
    )
    conv = store.create_concept(
        paper_id=review_paper.id, type="Theorem",
        title="Convergence of N-player Nash Equilibria",
        suggested_hub="Mean Field Games",
        statement_latex=r"$v^{N,i} \to U$ as $N\to\infty$ uniformly on compacts.",
        conclusion="N-player equilibria converge to the MFG solution.",
        canonical_keywords=["propagation of chaos", "Nash", "convergence rate"],
        ai_confidence=0.83,
    )
    # Proposed edges to existing promoted concepts + within the paper.
    store.create_edge(
        source_concept_id=conv.id, target_concept_id=master.id,
        relation_type="depends_on", status=EdgeStatus.PROPOSED.value,
        channel=EdgeChannel.AUTO.value, ai_confidence=0.86,
        justification="The convergence proof differentiates the master equation solution U.",
    )
    store.create_edge(
        source_concept_id=master.id, target_concept_id=monotone.id,
        relation_type="depends_on", status=EdgeStatus.PROPOSED.value,
        channel=EdgeChannel.SUGGEST.value, ai_confidence=0.7,
        justification="Uniqueness for the master equation invokes Lasry–Lions monotonicity.",
    )
    store.create_edge(
        source_concept_id=master.id, target_concept_id=mv.id,
        relation_type="related", status=EdgeStatus.PROPOSED.value,
        channel=EdgeChannel.SUGGEST.value, ai_confidence=0.6,
        justification="Both describe mean-field dynamics on the space of measures.",
    )

    # ── A couple more papers in other states (dashboard variety) ────────────────
    store.create_paper(title="Reinforcement Learning for Mean Field Control",
                       status=PaperStatus.S1_SKIM.value, arxiv_id="2403.09999",
                       one_liner="Q-learning convergence for mean-field MDPs.")
    store.create_paper(title="Graphon Mean Field Systems",
                       status=PaperStatus.S0_INBOX.value, arxiv_id="2003.13180")

    print("Seeded demo data:")
    print(f"  papers   : {len(store.list_papers())}")
    print(f"  concepts : {len(store.list_concepts())}")
    print(f"  edges    : {len(store.list_edges())}")
    print("Run:  uvicorn webapp.main:app --reload   then open http://127.0.0.1:8000")


if __name__ == "__main__":
    seed()

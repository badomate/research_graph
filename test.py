"""
ic_mfg_ingest.py

One-shot ingestion of all 56 mathematical objects from the ic_mfg catalog
into the Knowledge Inbox and Qdrant vector index.

Usage:
    docker exec -it orchestrator python scripts/ic_mfg_ingest.py \
        --paper-id <NOTION_PAPER_TRACKER_PAGE_ID> [--dry-run]

Prerequisites:
    1. Create a Paper Tracker page manually with:
         Title:           ic_mfg: Influence-Signature Coarsening for MFG
         Status:          s3-distilled
         Thesis Relevance: core
    2. Copy that page's Notion ID and pass it as --paper-id.
    3. All 56 KI pages will be created and indexed.
    4. Bulk-set verification_status = verified in Notion UI (filter by Source Paper).
    5. Set paper Status = s2-read to trigger PromotionEngine.
"""

import argparse
import logging
import os
import sys
import time

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.modules.extraction_schema import MathObject
from orchestrator.modules.ingestion import IngestionEngine
from orchestrator.modules.vector_index import VectorIndexEngine

from dotenv import load_dotenv
load_dotenv("C:\\Users\\blkv0u\\Desktop\\paper_pipeline\\.env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("ic_mfg_ingest")

# ── Hub ───────────────────────────────────────────────────────────────────────
HUB = "Mean Field Games"

# ── Type constants ────────────────────────────────────────────────────────────
DEFINITION  = "Definition"
THEOREM     = "Theorem"
ALGORITHM   = "Algorithm"
LEMMA       = "Lemma"

# ── Full 56-entry catalog ─────────────────────────────────────────────────────
CATALOG: list[dict] = [

    # ── 1. Problem Domain ─────────────────────────────────────────────────────

    {
        "title": "Mean Field Game System",
        "type": DEFINITION,
        "statement_latex": (
            r"\[-\partial_t u + H(x,\partial_x u, m) = \nu\,\partial_{xx}u, \quad"
            r"\partial_t m - \nu\,\partial_{xx}m + \partial_x(m\,\alpha^*) = 0\]"
        ),
        "assumptions": (
            "Periodic domain $\\mathbb{T}^1 = [0,1)$. Viscosity $\\nu > 0$. "
            "Terminal condition $u(T,x) = g(x,m_T)$, initial condition "
            "$m(0,x) = m_0(x)$ with $\\int m_0\\,dx = 1$."
        ),
        "conclusion": (
            "Models the Nash equilibrium of a continuum of identical agents with "
            "dynamics $dX_t = \\alpha_t\\,dt + \\sqrt{2\\nu}\\,dW_t$. Solved as a "
            "fixed point: HJB backward given $m$, FP forward given $u$."
        ),
        "interpretation": (
            "Central system of the codebase. All downstream computation — adjoint, "
            "coarsening, QoI evaluation — is defined relative to this fixed point."
        ),
        "canonical_keywords": ["mean field game", "HJB", "Fokker-Planck", "Nash equilibrium", "coupled PDE"],
        "prereq_keywords": ["stochastic optimal control", "viscosity solutions", "probability measure"],
        "downstream_keywords": ["MFG equilibrium", "Picard iteration", "adjoint system", "QoI"],
        "suggested_hub": HUB,
        "setting": ["continuous time", "periodic domain", "diffusion"],
        "named_tools": ["Schauder fixed point", "Lasry-Lions monotonicity"],
        "result_category": "system definition",
    },

    {
        "title": "Hamiltonian H(x,p,m)",
        "type": DEFINITION,
        "statement_latex": r"\[H(x,p,m) = V(x,m) - \tfrac{1}{2}p^2, \qquad p = \partial_x u\]",
        "assumptions": (
            "Quadratic control cost $\\frac{1}{2}|\\alpha|^2$. Sign convention "
            "$H = V - p^2/2$ encodes minimisation (not maximisation)."
        ),
        "conclusion": (
            "Arises from the Pontryagin minimum principle. The Legendre transform of "
            "$\\alpha \\mapsto \\frac{1}{2}|\\alpha|^2 + \\alpha p$ gives "
            "$H = V - \\frac{1}{2}p^2$ under minimisation."
        ),
        "interpretation": (
            "The Hamiltonian structure determines the feedback control law and "
            "the sign convention is critical: a wrong sign causes solver divergence."
        ),
        "canonical_keywords": ["Hamiltonian", "Pontryagin", "optimal control", "Legendre transform"],
        "prereq_keywords": ["control cost", "value function"],
        "downstream_keywords": ["optimal control alpha*", "HJB equation", "adjoint HJB"],
        "suggested_hub": HUB,
        "setting": ["continuous time", "periodic domain"],
        "named_tools": ["Pontryagin minimum principle", "Legendre transform"],
        "result_category": "definition",
    },

    {
        "title": "Optimal Control alpha*(t,x)",
        "type": DEFINITION,
        "statement_latex": r"\[\alpha^*(t,x) = -\partial_x u^*(t,x)\]",
        "assumptions": "MFG equilibrium $(u^*, m^*)$ exists. Quadratic control cost.",
        "conclusion": (
            "The minimiser of the Hamiltonian over controls at the MFG equilibrium. "
            "Agents drift in the direction of steepest descent of $u^*$."
        ),
        "interpretation": (
            "This feedback policy defines the drift in the Fokker-Planck equation. "
            "Derived from $\\operatorname{argmin}_\\alpha\\{V - \\alpha\\partial_x u^* + \\frac{1}{2}|\\alpha|^2\\}$."
        ),
        "canonical_keywords": ["optimal control", "feedback policy", "drift", "steepest descent"],
        "prereq_keywords": ["Hamiltonian", "value function", "Pontryagin"],
        "downstream_keywords": ["Fokker-Planck", "coarse generator", "IMEX scheme"],
        "suggested_hub": HUB,
        "setting": ["continuous time"],
        "named_tools": ["Pontryagin minimum principle"],
        "result_category": "definition",
    },

    {
        "title": "MFG Equilibrium",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\text{A pair }(u^*,m^*)\text{ s.t. } u^*\text{ solves HJB given }m^*,"
            r"\; m^*\text{ solves FP given }\alpha^*=-\partial_x u^*,"
            r"\;\int m^*(t,\cdot)\,dx=1.\]"
        ),
        "assumptions": (
            "Lasry-Lions monotonicity for uniqueness. Schauder fixed-point theorem for existence. "
            "$\\nu > 0$ for smoothing and compactness."
        ),
        "conclusion": (
            "The central fixed point of the system. Existence follows from Schauder; "
            "uniqueness follows from Lasry-Lions monotonicity."
        ),
        "interpretation": (
            "All downstream computation — adjoint, coarsening, QoI evaluation — is defined "
            "relative to this fixed point. Computed via Picard iteration."
        ),
        "canonical_keywords": ["MFG equilibrium", "fixed point", "Nash equilibrium", "HJB-FP system"],
        "prereq_keywords": ["Lasry-Lions monotonicity", "Schauder", "HJB", "Fokker-Planck"],
        "downstream_keywords": ["adjoint system", "QoI", "coarse system", "partition"],
        "suggested_hub": HUB,
        "setting": ["continuous time", "periodic domain"],
        "named_tools": ["Schauder fixed point theorem", "Picard iteration"],
        "result_category": "existence and uniqueness",
    },

    {
        "title": "Lasry-Lions Monotonicity Condition",
        "type": THEOREM,
        "statement_latex": (
            r"\[\int_{\mathbb{T}^1}\bigl(F(x,m_1)-F(x,m_2)\bigr)"
            r"\bigl(m_1(x)-m_2(x)\bigr)\,dx \geq 0 \quad \forall\, m_1,m_2\]"
        ),
        "assumptions": "Coupling $F(x,m)$ is a functional of the measure $m$.",
        "conclusion": (
            "Sufficient condition for uniqueness of the MFG equilibrium and "
            "convergence of Picard iteration."
        ),
        "interpretation": (
            "At the coarse level, this condition is not guaranteed to hold under "
            "mass-weighted aggregation, leading to potential non-uniqueness for "
            "large coupling strengths."
        ),
        "canonical_keywords": ["Lasry-Lions", "monotonicity", "uniqueness", "MFG"],
        "prereq_keywords": ["coupling function", "probability measure"],
        "downstream_keywords": ["Picard iteration", "MFG equilibrium", "PI uniqueness check"],
        "suggested_hub": HUB,
        "setting": ["functional analysis", "mean field games"],
        "named_tools": [],
        "result_category": "uniqueness condition",
    },

    # ── 2. Coupling Functions ─────────────────────────────────────────────────

    {
        "title": "DoubleWell Coupling",
        "type": DEFINITION,
        "statement_latex": (
            r"\[V_{\text{DW}}(x,m) = \beta(x-a)^2(x-b)^2, \quad D_m F \equiv 0\]"
            r"\[\text{Parameters: } a=0.25,\; b=0.75,\; \beta=10.0\]"
        ),
        "assumptions": "$D_m F \\equiv 0$: zero mean-field coupling.",
        "conclusion": (
            "Pure state-dependent potential with two stable equilibria at $a$ and $b$. "
            "HJB solution $u^*$ is independent of $m_0$; FP equation is linear in $m$; "
            "adjoint FP has no nonlocal source term."
        ),
        "interpretation": (
            "Used for clean adjoint verification. Strong trapping at $\\beta=10$ "
            "causes $\\delta J/\\delta m_0 \\approx \\text{const}$, flattening $s_\\eta$. "
            "Correct verification scenario, wrong demonstration scenario."
        ),
        "canonical_keywords": ["double well", "potential", "zero coupling", "verification scenario"],
        "prereq_keywords": ["MFG system", "Hamiltonian"],
        "downstream_keywords": ["adjoint system", "QoI linearity", "DoubleWell scenario"],
        "suggested_hub": HUB,
        "setting": ["periodic domain"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Herding Coupling",
        "type": DEFINITION,
        "statement_latex": (
            r"\[V_{\text{H}}(x,m) = \tfrac{1}{2}(x - \bar{x}_m)^2, \quad"
            r"\bar{x}_m = \int_{\mathbb{T}^1} x\,m(x)\,dx\]"
        ),
        "assumptions": "Nonlocal $m$-dependence through population mean $\\bar{x}_m$.",
        "conclusion": (
            "Models agents attracted to the population mean. Creates genuine nonlocal "
            "$m$-dependence so the adjoint FP carries the coupling derivative term "
            "$D_m F^*[\\lambda, m^*]$."
        ),
        "interpretation": (
            "Standard LQ-MFG benchmark. The Gâteaux derivative is "
            "$D_m F(x)[h] = -(x-\\bar{x}_m)\\int y\\,h(y)\\,dy$. "
            "Omitting this caused 20-50% adjoint error."
        ),
        "canonical_keywords": ["herding", "mean field coupling", "nonlocal", "LQ-MFG"],
        "prereq_keywords": ["MFG system", "population mean"],
        "downstream_keywords": ["coupling Gateaux derivative", "adjoint FP", "herding scenario"],
        "suggested_hub": HUB,
        "setting": ["periodic domain", "nonlocal coupling"],
        "named_tools": ["Gâteaux derivative"],
        "result_category": "definition",
    },

    {
        "title": "Terminal Cost g(x,mT)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[g_{\text{DW}}(x,m_T) = 0, \qquad"
            r"g_{\text{H}}(x,m_T) = \tfrac{1}{2}(x-\bar{x}_{m_T})^2\]"
        ),
        "assumptions": "Sets terminal boundary condition for HJB: $u(T,x) = g(x,m_T)$.",
        "conclusion": "Determines what agents are penalised for at time $T$.",
        "interpretation": (
            "In DoubleWell, $g=0$ so $u^*(T,x)=0$ exactly — verified to $10^{-10}$. "
            "In herding, terminal cost matches running cost structure."
        ),
        "canonical_keywords": ["terminal cost", "boundary condition", "HJB terminal"],
        "prereq_keywords": ["HJB equation", "coupling function"],
        "downstream_keywords": ["MFG equilibrium", "fine solver conservation laws"],
        "suggested_hub": HUB,
        "setting": ["continuous time"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Running QoI Kernel l(t,x)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\ell(t,x) = \exp\!\Bigl(-\tfrac{1}{2}"
            r"\Bigl(\tfrac{x-x_0}{\sigma}\Bigr)^2\Bigr), \quad x_0\in[0,1],\; \sigma>0\]"
        ),
        "assumptions": (
            "Gaussian kernel centred at $x_0$ with width $\\sigma$. "
            "Method requires $\\sigma \\lesssim h_K = 1/K$ for partition to be beneficial."
        ),
        "conclusion": (
            "Defines the quantity of interest: population density passing through the "
            "region near $x_0$ over $[0,T]$. The choice of $x_0$ and $\\sigma$ "
            "determines the sensitivity landscape $s_\\eta$ and partition quality."
        ),
        "interpretation": (
            "A narrow, asymmetrically placed $\\ell$ produces a peaked $s_\\eta$ and "
            "a useful partition. If $\\sigma$ is large, $J_{\\text{coarse}}$ becomes "
            "insensitive to $m_0$."
        ),
        "canonical_keywords": ["QoI kernel", "Gaussian", "goal-oriented", "sensitivity"],
        "prereq_keywords": ["MFG system"],
        "downstream_keywords": ["QoI J", "adjoint influence signature", "partition"],
        "suggested_hub": HUB,
        "setting": ["goal-oriented analysis"],
        "named_tools": [],
        "result_category": "definition",
    },

    # ── 3. Quantity of Interest ───────────────────────────────────────────────

    {
        "title": "Quantity of Interest J(m0)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[J(m_0) = \int_0^T\int_{\mathbb{T}^1}\ell(t,x)\,m^*(t,x;m_0)\,dx\,dt"
            r"+ \int_{\mathbb{T}^1}\ell_T(x)\,m^*(T,x;m_0)\,dx\]"
        ),
        "assumptions": (
            "Primary scalar output of the MFG. $m^*$ depends on $m_0$ nonlinearly "
            "in general due to $m$-dependence in $u^*$."
        ),
        "conclusion": (
            "The many-query problem: evaluate $J(m_0^q)$ for a family of query ICs "
            "efficiently, without running a full MFG solve per query."
        ),
        "interpretation": (
            "Linearity holds when $D_m F = 0$ (DoubleWell). "
            "Nonlinearity in herding requires the full offline/online decomposition."
        ),
        "canonical_keywords": ["QoI", "quantity of interest", "output functional", "many-query"],
        "prereq_keywords": ["MFG equilibrium", "QoI kernel", "Fokker-Planck"],
        "downstream_keywords": ["QoI linearity", "error certificate", "offline/online decomposition"],
        "suggested_hub": HUB,
        "setting": ["goal-oriented analysis"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "QoI Linearity for Zero Coupling",
        "type": THEOREM,
        "statement_latex": (
            r"\[J(m_0) = \langle s_\eta, m_0\rangle_{L^2(\mathbb{T}^1)}"
            r"= \int_{\mathbb{T}^1} s_\eta(x)\,m_0(x)\,dx \quad \text{when } D_m F \equiv 0\]"
        ),
        "assumptions": "$D_m F \\equiv 0$: zero mean-field coupling (DoubleWell scenario).",
        "conclusion": (
            "$s_\\eta$ is the exact functional derivative of $J$ w.r.t. $m_0$. "
            "FP equation becomes linear in $m$ so $J$ is linear in $m_0$."
        ),
        "interpretation": (
            "Mathematical basis for the error certificate. Partition error bounds "
            "the $L^\\infty$ error in approximating $s_\\eta$ within clusters, "
            "which bounds $|J(m_0) - J_{\\text{coarse}}(m_0)|$."
        ),
        "canonical_keywords": ["QoI linearity", "linear response", "sensitivity", "zero coupling"],
        "prereq_keywords": ["QoI J", "adjoint influence signature", "DoubleWell coupling"],
        "downstream_keywords": ["error certificate", "intra-cluster oscillation"],
        "suggested_hub": HUB,
        "setting": ["linear response theory"],
        "named_tools": [],
        "result_category": "structural property",
    },

    # ── 4. Adjoint System ─────────────────────────────────────────────────────

    {
        "title": "Adjoint Variables (lambda, eta)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\lambda(t,x):\text{ costate for HJB, backward in }t\]"
            r"\[\eta(t,x):\text{ costate for FP, forward in }t\]"
        ),
        "assumptions": (
            "Derived by differentiating the MFG fixed-point conditions w.r.t. $m_0$ "
            "in direction $h$. Linearised around $(u^*, m^*)$."
        ),
        "conclusion": (
            "Together characterise $\\delta J/\\delta m_0$. Computed once at cost $O(NM)$. "
            "Their combination at $t=0$ yields $s_\\eta = \\eta(0,\\cdot)$."
        ),
        "interpretation": (
            "$\\eta$ can be solved independently of $\\lambda$ when $D_m F = 0$. "
            "The adjoint system is the linearisation of the MFG fixed-point operator."
        ),
        "canonical_keywords": ["adjoint variables", "costate", "sensitivity", "adjoint-state method"],
        "prereq_keywords": ["MFG equilibrium", "QoI J"],
        "downstream_keywords": ["adjoint HJB", "adjoint FP", "adjoint influence signature"],
        "suggested_hub": HUB,
        "setting": ["adjoint-state method", "PDE-constrained optimisation"],
        "named_tools": ["adjoint-state method"],
        "result_category": "definition",
    },

    {
        "title": "Adjoint HJB Equation",
        "type": DEFINITION,
        "statement_latex": (
            r"\[-\partial_t\lambda - \nu\,\partial_{xx}\lambda + \alpha^*\,\partial_x\lambda"
            r"= -\ell(t,x) - D_m F^*[\eta,m^*](x), \quad \lambda(T,x) = -\ell_T(x)\]"
        ),
        "assumptions": (
            "Terminal condition $\\lambda(T,x) = -\\ell_T(x)$. "
            "For $D_m F=0$: source reduces to $-\\ell(t,x)$ and $\\lambda$ decouples from $\\eta$."
        ),
        "conclusion": (
            "Propagates QoI sensitivity backward through the HJB equation. "
            "The coupling derivative source links the two adjoint equations."
        ),
        "interpretation": (
            "For herding: $D_m F^*[\\eta,m^*](x) = -(x-\\bar{x}_{m^*})\\int y\\,\\eta(t,y)\\,dy$, "
            "computed at each time step."
        ),
        "canonical_keywords": ["adjoint HJB", "backward PDE", "QoI sensitivity", "adjoint"],
        "prereq_keywords": ["adjoint variables", "coupling Gateaux derivative", "MFG equilibrium"],
        "downstream_keywords": ["adjoint influence signature"],
        "suggested_hub": HUB,
        "setting": ["adjoint-state method", "PDE"],
        "named_tools": [],
        "result_category": "PDE",
    },

    {
        "title": "Adjoint Fokker-Planck Equation",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\partial_t\eta = \nu\,\partial_{xx}\eta - \partial_x(\alpha^*\eta)"
            r"+ D_m F^*[\lambda,m^*](x), \quad \eta(0,x) = 0\]"
        ),
        "assumptions": "Initial condition $\\eta(0,x) = 0$.",
        "conclusion": (
            "Propagates perturbation to $m_0$ forward through the FP equation. "
            "The term $D_m F^*[\\lambda,m^*]$ was initially missing, causing 20-50% adjoint error."
        ),
        "interpretation": (
            "Including $D_m F^*[\\lambda,m^*]$ reduced finite-difference error from "
            "~30% at $N=32$ to 1.6% at $N=128$, confirming $O(h)$ convergence."
        ),
        "canonical_keywords": ["adjoint Fokker-Planck", "forward PDE", "adjoint", "sensitivity"],
        "prereq_keywords": ["adjoint variables", "coupling Gateaux derivative", "optimal control"],
        "downstream_keywords": ["adjoint influence signature", "adjoint FD check"],
        "suggested_hub": HUB,
        "setting": ["adjoint-state method", "PDE"],
        "named_tools": [],
        "result_category": "PDE",
    },

    {
        "title": "Coupling Gateaux Derivative DmF",
        "type": DEFINITION,
        "statement_latex": (
            r"\[D_m F(x)[h] = -(x-\bar{x}_m)\int_{\mathbb{T}^1}y\,h(y)\,dy\]"
            r"\[D_m F^*[\lambda,m^*](x) = -\lambda(t,x)\int(y-\bar{x}_{m^*})m^*(t,y)\,dy"
            r"+ \bar{x}_{m^*}\int\lambda(t,y)m^*(t,y)\,dy\]"
        ),
        "assumptions": "Herding coupling $F(x,m) = \\frac{1}{2}(x-\\bar{x}_m)^2$.",
        "conclusion": (
            "Appears as a nonlocal source term in both adjoint equations. "
            "Required for the adjoint to be consistent with the full MFG fixed-point. "
            "For DoubleWell, $D_m F \\equiv 0$ identically."
        ),
        "interpretation": "Standard Gâteaux derivative of the coupling functional w.r.t. $m$.",
        "canonical_keywords": ["Gateaux derivative", "coupling derivative", "nonlocal", "functional calculus"],
        "prereq_keywords": ["herding coupling", "adjoint variables"],
        "downstream_keywords": ["adjoint HJB", "adjoint FP"],
        "suggested_hub": HUB,
        "setting": ["functional calculus"],
        "named_tools": ["Gâteaux derivative"],
        "result_category": "definition",
    },

    {
        "title": "Adjoint Influence Signature s_eta",
        "type": DEFINITION,
        "statement_latex": (
            r"\[s_\eta(x) = \eta(0,x), \qquad"
            r"\frac{\delta J}{\delta m_0}(x) \approx s_\eta(x)\]"
        ),
        "assumptions": (
            "Exact functional derivative in zero-coupling case. "
            "Approximation for nonzero coupling."
        ),
        "conclusion": (
            "Primary input to the partition algorithm. Regions where $s_\\eta$ varies "
            "rapidly require fine clusters; flat regions can be coarsened."
        ),
        "interpretation": (
            "For zero coupling, $s_\\eta$ is the exact sensitivity (not an approximation). "
            "Partition quality fully determined by how well $\\Pi$ resolves $s_\\eta$ variation."
        ),
        "canonical_keywords": ["influence signature", "sensitivity", "adjoint", "functional derivative"],
        "prereq_keywords": ["adjoint FP", "QoI J"],
        "downstream_keywords": ["partition", "error certificate", "intra-cluster oscillation"],
        "suggested_hub": HUB,
        "setting": ["goal-oriented analysis", "adjoint-state method"],
        "named_tools": [],
        "result_category": "definition",
    },

    # ── 5. Discretization ─────────────────────────────────────────────────────

    {
        "title": "Discrete Grid",
        "type": DEFINITION,
        "statement_latex": (
            r"\[N\text{ spatial nodes}, h=1/N, x_i=ih, i=0,\ldots,N-1\]"
            r"\[M\text{ time steps}, \Delta t=T/M, t^n=n\Delta t, n=0,\ldots,M\]"
        ),
        "assumptions": "Uniform grid on $\\mathbb{T}^1=[0,1)$. Periodic boundary conditions.",
        "conclusion": (
            "All PDEs discretised on this grid. Convergence verified at rates "
            "$\\alpha \\approx 0.98$-$1.48$ for $N=64,128,256$ with reference at $N=512$, $M=1024$."
        ),
        "interpretation": "Foundation for all numerical discretisations in the codebase.",
        "canonical_keywords": ["discrete grid", "spatial discretisation", "periodic", "uniform grid"],
        "prereq_keywords": [],
        "downstream_keywords": ["Laplacian matrix", "IMEX scheme", "upwind flux"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "periodic domain"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Laplacian Matrix Lh",
        "type": DEFINITION,
        "statement_latex": (
            r"\[[L_h]_{ij} = \begin{cases}"
            r"-2/h^2 & j=i \\ +1/h^2 & |i-j|=1\text{ or }\{i,j\}=\{0,N-1\} \\ 0 & \text{otherwise}"
            r"\end{cases}\]"
        ),
        "assumptions": "Periodic second-order finite difference on $N$ nodes.",
        "conclusion": (
            "Discretises $\\partial_{xx}$ on $\\mathbb{T}^1$. Symmetric negative semi-definite "
            "with one-dimensional null space (constant vector). Kernel reflects mass conservation."
        ),
        "interpretation": "Used in implicit diffusion step of FP and in coarse operator builder.",
        "canonical_keywords": ["Laplacian matrix", "finite difference", "periodic", "diffusion"],
        "prereq_keywords": ["discrete grid"],
        "downstream_keywords": ["IMEX scheme", "semi-implicit HJB", "coarse generator"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "finite differences"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Conservative Upwind Flux for Advection",
        "type": DEFINITION,
        "statement_latex": (
            r"\[[\partial_x(m\alpha)]_i \approx \frac{1}{h}"
            r"\bigl[\max(\alpha_i,0)(m_i-m_{i-1}) + \min(\alpha_i,0)(m_{i+1}-m_i)\bigr]\]"
        ),
        "assumptions": "Sign of $\\alpha_i$ determines upwind direction per node.",
        "conclusion": (
            "Ensures positivity and mass conservation of the FP discretisation. "
            "First-order in space, satisfies discrete maximum principle."
        ),
        "interpretation": (
            "Essential for numerical stability when $|\\alpha^*|$ is large "
            "(DoubleWell with $\\beta=10$, $|\\alpha^*|\\sim 2$)."
        ),
        "canonical_keywords": ["upwind scheme", "conservative flux", "advection", "FP discretisation"],
        "prereq_keywords": ["discrete grid", "optimal control"],
        "downstream_keywords": ["IMEX scheme", "coarse generator"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "conservation laws"],
        "named_tools": [],
        "result_category": "numerical scheme",
    },

    {
        "title": "IMEX Time-Stepping Scheme",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\frac{m^{n+1}-m^n}{\Delta t} = \nu\,\partial_{xx}^h m^{n+1}"
            r"- \partial_x^h(m^n\alpha^{*,n})\]"
        ),
        "assumptions": "Diffusion treated implicitly, advection treated explicitly. First-order in time.",
        "conclusion": (
            "Avoids CFL restriction $\\Delta t \\leq h^2/(2\\nu)$ from explicit diffusion. "
            "Implicit step solves tridiagonal system $(I - \\Delta t\\nu L_h^T)m^{n+1} = \\text{rhs}$, $O(N)$ per step."
        ),
        "interpretation": "Convergence rate verified at $\\alpha \\approx 1.0$.",
        "canonical_keywords": ["IMEX", "implicit-explicit", "time stepping", "Fokker-Planck"],
        "prereq_keywords": ["Laplacian matrix", "upwind flux", "discrete grid"],
        "downstream_keywords": ["Picard iteration", "coarse IMEX split"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "time integration"],
        "named_tools": [],
        "result_category": "numerical scheme",
    },

    {
        "title": "Semi-Implicit HJB Discretisation",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[(I - \Delta t\,\nu\,L_h)u^n = u^{n+1}"
            r"+ \Delta t\bigl(\tfrac{1}{2}|\partial_x^h u^{n+1}|^2 + V(x,m^n)\bigr)\]"
        ),
        "assumptions": "Diffusion implicit for A-stability; Hamiltonian term explicit. $u^M = g(x,m^M)$.",
        "conclusion": (
            "Keeps scheme linear in $u^n$. Tridiagonal system solved in $O(N)$ per step. "
            "Gradient $\\partial_x^h u$ by central differences."
        ),
        "interpretation": "Backward time-stepping for the HJB equation.",
        "canonical_keywords": ["semi-implicit HJB", "backward time-stepping", "A-stability"],
        "prereq_keywords": ["Laplacian matrix", "discrete grid", "Hamiltonian"],
        "downstream_keywords": ["Picard iteration", "coarse HJB"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "time integration"],
        "named_tools": [],
        "result_category": "numerical scheme",
    },

    {
        "title": "Picard (Policy) Iteration",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\text{(1) Given }m^k\text{, solve HJB backward }\to u^{k+1}\]"
            r"\[\text{(2) Set }\alpha^{k+1}=-\partial_x u^{k+1}\]"
            r"\[\text{(3) Solve FP forward }\to m^{k+1}\]"
            r"\[\text{(4) Stop if }\|m^{k+1}-m^k\|_\infty < \varepsilon\]"
        ),
        "assumptions": "Convergence guaranteed under Lasry-Lions monotonicity.",
        "conclusion": (
            "Standard solver for the MFG fixed point. Applied at both fine level "
            "(FineMFGSolver) and coarse level (CoarseMFGSolverGraphFV)."
        ),
        "interpretation": (
            "For non-monotone coarse systems (large $K$, strong coupling), "
            "multiple equilibria may exist and initialisation determines which one is found."
        ),
        "canonical_keywords": ["Picard iteration", "policy iteration", "fixed point", "MFG solver"],
        "prereq_keywords": ["MFG system", "Lasry-Lions monotonicity", "IMEX scheme", "semi-implicit HJB"],
        "downstream_keywords": ["MFG equilibrium", "PI uniqueness check", "coarse PI residual"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "fixed-point iteration"],
        "named_tools": [],
        "result_category": "algorithm",
    },

    # ── 6. Coarse System ──────────────────────────────────────────────────────

    {
        "title": "State Space Partition Pi",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\Pi = \{C_1,\ldots,C_K\},\quad C_k\cap C_l=\emptyset\text{ for }k\neq l,"
            r"\quad\bigcup_{k=1}^K C_k = \{0,\ldots,N-1\}\]"
        ),
        "assumptions": "Each $C_k$ is a subset of grid node indices. Contiguous partitions preferred in 1D.",
        "conclusion": (
            "Defines aggregation from fine $N$-dimensional to coarse $K$-dimensional state space. "
            "Built by minimising intra-cluster oscillation of $s_\\eta$, directly minimising $\\mathcal{E}(K)$."
        ),
        "interpretation": "Analogous to aggregation in multi-scale methods and goal-oriented mesh refinement.",
        "canonical_keywords": ["partition", "state space", "aggregation", "coarsening", "clustering"],
        "prereq_keywords": ["adjoint influence signature", "discrete grid"],
        "downstream_keywords": ["coarse probability vector", "coarse generator", "error certificate"],
        "suggested_hub": HUB,
        "setting": ["coarse graining", "multi-scale methods"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Coarse Probability Vector m-bar(t)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\bar{m}_k(t) = \int_{C_k}m^*(t,x)\,dx \approx h\sum_{i\in C_k}m^*(t,x_i),"
            r"\quad\sum_{k=1}^K\bar{m}_k(t) = 1\]"
        ),
        "assumptions": "$m^* \\geq 0$. Conservation: $\\sum_k \\bar{m}_k(t) = 1$.",
        "conclusion": (
            "Coarse state representing fraction of total population in cluster $k$. "
            "Mass conservation verified to $2\\times10^{-16}$; positivity holds by construction."
        ),
        "interpretation": "Standard Galerkin projection / aggregation of the fine density.",
        "canonical_keywords": ["coarse state", "aggregation", "mass fractions", "Galerkin projection"],
        "prereq_keywords": ["state space partition", "MFG equilibrium"],
        "downstream_keywords": ["coarse generator", "coarse HJB", "coarse QoI"],
        "suggested_hub": HUB,
        "setting": ["coarse graining"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Coarse Kolmogorov Generator Q-bar(t)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\bar{Q}_{kl}(t) = \frac{F_{k\to l}(t)}{\bar{m}_k(t)},\quad"
            r"F_{k\to l}(t) = \sum_{\substack{i\in C_k\\j\in C_l}}Q^{\text{fine}}_{ij}(t)\cdot m^*(t,x_i)\cdot h\]"
        ),
        "assumptions": (
            "$\\bar{Q}_{kk} = -\\sum_{l\\neq k}\\bar{Q}_{kl}$. "
            "Built once offline from reference trajectory."
        ),
        "conclusion": (
            "Valid Markov generator: rows sum to zero, off-diagonal entries $\\geq 0$. "
            "Original implementation missing $m^*(t,x_i)\\cdot h$ factor caused inflated generators ($\\sim10^{12}$)."
        ),
        "interpretation": "Coarse analogue of fine-grid Kolmogorov generator via mass-weighted flux aggregation.",
        "canonical_keywords": ["Kolmogorov generator", "Markov chain", "aggregation", "coarse operator"],
        "prereq_keywords": ["state space partition", "coarse probability vector", "optimal control"],
        "downstream_keywords": ["IMEX coarse split", "detailed balance", "coarse HJB", "trivial partition check"],
        "suggested_hub": HUB,
        "setting": ["Markov chains", "coarse graining"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "IMEX Split of the Coarse Generator",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\bar{Q}(t) = \bar{Q}^{\text{diff}} + \bar{Q}^{\text{adv}}(t)\]"
            r"\[\bar{Q}^{\text{diff}}\text{: time-independent (from }\nu/h^2\text{ terms)},\quad"
            r"\bar{Q}^{\text{adv}}(t)\text{: time-dependent (from }\alpha^*/h\text{ terms)}\]"
        ),
        "assumptions": "$\\bar{Q}^{\\text{diff}}$ treated implicitly (precomputed factorisation); $\\bar{Q}^{\\text{adv}}$ treated explicitly.",
        "conclusion": (
            "Mirrors fine-level IMEX scheme at coarse level. Precomputed implicit diffusion matrix "
            "$(I-\\Delta t(\\bar{Q}^{\\text{diff}})^T)$ factorised once, reused across all $M$ steps and $Q$ queries."
        ),
        "interpretation": "Custom extension of IMEX to the coarse Markov chain.",
        "canonical_keywords": ["IMEX coarse", "diffusion advection split", "Markov chain", "precomputed"],
        "prereq_keywords": ["coarse generator", "IMEX scheme"],
        "downstream_keywords": ["coarse HJB", "fixed-HJB forward pass"],
        "suggested_hub": HUB,
        "setting": ["numerical methods", "coarse graining"],
        "named_tools": [],
        "result_category": "algorithm",
    },

    {
        "title": "Detailed Balance for Coarse Diffusion",
        "type": THEOREM,
        "statement_latex": (
            r"\[\bar{m}_k^*(t)\cdot\bar{Q}^{\text{diff}}_{kl}(t)"
            r"= \bar{m}_l^*(t)\cdot\bar{Q}^{\text{diff}}_{lk}(t) \quad \forall\,k\neq l,\;t\]"
        ),
        "assumptions": "Pure diffusion is a reversible process.",
        "conclusion": (
            "$\\bar{m}^*$ is the stationary distribution of the diffusion generator. "
            "Coarse diffusion generator is generally asymmetric but satisfies detailed balance w.r.t. $\\bar{m}^*$."
        ),
        "interpretation": "Replaces naive symmetry check. Verified in Level 1 validation.",
        "canonical_keywords": ["detailed balance", "reversible Markov chain", "stationary distribution"],
        "prereq_keywords": ["coarse generator", "coarse probability vector"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["Markov chains", "reversibility"],
        "named_tools": [],
        "result_category": "structural property",
    },

    {
        "title": "Coarse HJB Equation",
        "type": DEFINITION,
        "statement_latex": (
            r"\[(I-\Delta t\,\bar{Q}^{\text{diff}})\bar{u}^n = \bar{u}^{n+1}"
            r"+ \Delta t\bigl(\tfrac{1}{2}\|\nabla_{\text{graph}}\bar{u}^{n+1}\|^2"
            r"+ \bar{V}(\bar{m}^n)\bigr), \quad \bar{u}^M = \bar{g}\]"
        ),
        "assumptions": "Aggregated terminal cost $\\bar{g}$. At $K=N$: must reproduce fine HJB exactly.",
        "conclusion": "Coarse analogue of fine HJB. Determines coarse value function $\\bar{u}^*(t)$.",
        "interpretation": "Graph-based HJB discretisation. Verified to reproduce fine HJB at $K=N$ (Level 3).",
        "canonical_keywords": ["coarse HJB", "graph HJB", "value function", "backward equation"],
        "prereq_keywords": ["coarse generator IMEX split", "coarse probability vector", "graph gradient proxy"],
        "downstream_keywords": ["Picard iteration", "fixed-HJB forward pass"],
        "suggested_hub": HUB,
        "setting": ["coarse graining", "graph methods"],
        "named_tools": [],
        "result_category": "PDE",
    },

    {
        "title": "Graph Gradient Energy Proxy",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\bigl[\|\nabla_{\text{graph}}\bar{u}\|^2\bigr]_k"
            r"= \sum_{l\sim k}\bar{Q}^{\text{adv}}_{kl}(\bar{u}_l-\bar{u}_k)^2\]"
        ),
        "assumptions": "Exact for $K=N$ (singleton clusters). Accuracy degrades as $K$ decreases.",
        "conclusion": "Approximates kinetic energy term in coarse Hamiltonian without a spatial gradient.",
        "interpretation": "Substitutes for $\\frac{1}{2}|\\partial_x u|^2$ which has no direct meaning at the cluster level.",
        "canonical_keywords": ["graph gradient", "kinetic energy", "graph Laplacian", "coarse Hamiltonian"],
        "prereq_keywords": ["coarse generator", "coarse HJB"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["graph methods", "spectral graph theory"],
        "named_tools": ["graph Laplacian"],
        "result_category": "definition",
    },

    {
        "title": "Coarse PI Fixed-Point Residual",
        "type": DEFINITION,
        "statement_latex": r"\[R = \|\bar{m}^{\text{HJB}(\bar{m})} - \bar{m}\|_\infty\]",
        "assumptions": "PI has converged. $\\bar{m}^{\\text{HJB}(\\bar{m})}$ is FP trajectory from HJB given $\\bar{m}$.",
        "conclusion": (
            "Actual fixed-point condition, stronger than PI stopping criterion. "
            "Verified to $R \\leq 10^{-16}$ in Level 4 validation."
        ),
        "interpretation": "Separate from PI uniqueness check — PI always converges to some equilibrium.",
        "canonical_keywords": ["fixed-point residual", "PI convergence", "self-consistency"],
        "prereq_keywords": ["Picard iteration", "coarse HJB", "coarse probability vector"],
        "downstream_keywords": ["PI uniqueness check"],
        "suggested_hub": HUB,
        "setting": ["fixed-point theory"],
        "named_tools": [],
        "result_category": "definition",
    },

    # ── 7. Lifting Operators ──────────────────────────────────────────────────

    {
        "title": "Uniform Lifting m-tilde-unif",
        "type": DEFINITION,
        "statement_latex": r"\[\tilde{m}^{\text{unif}}(t,x) = \frac{\bar{m}_k(t)}{|C_k|\cdot h}, \quad x\in C_k\]",
        "assumptions": "Mass spread uniformly within each cluster.",
        "conclusion": (
            "$J_{\\text{lifted}}^{\\text{unif}} \\equiv J_{\\text{coarse}}$ algebraically, regardless of $\\ell$. "
            "Uniform lifting useless for improving QoI accuracy — template lifting required."
        ),
        "interpretation": "Standard aggregation-disaggregation. Right-inverse property satisfied.",
        "canonical_keywords": ["uniform lifting", "disaggregation", "lifting operator"],
        "prereq_keywords": ["state space partition", "coarse probability vector"],
        "downstream_keywords": ["coarse QoI", "right-inverse property"],
        "suggested_hub": HUB,
        "setting": ["multi-scale methods"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Template Lifting m-tilde-template",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\tilde{m}^{\text{template}}(t,x)"
            r"= \bar{m}_k(t)\cdot\frac{m^*(t,x)}{\int_{C_k}m^*(t,y)\,dy}, \quad x\in C_k\]"
        ),
        "assumptions": "Reference trajectory $m^*$ available from offline stage.",
        "conclusion": (
            "Breaks algebraic identity $J_{\\text{lifted}}\\equiv J_{\\text{coarse}}$. "
            "When $\\bar{m}_k(t) = \\int_{C_k}m^*(t,x)\\,dx$, recovers $m^*$ exactly "
            "(verified at $4.4\\times10^{-16}$)."
        ),
        "interpretation": "Primary lifting method. Applies fine-resolution structure of $\\ell$ to lifted density.",
        "canonical_keywords": ["template lifting", "lifting operator", "reconstruction", "multiscale"],
        "prereq_keywords": ["state space partition", "coarse probability vector", "MFG equilibrium"],
        "downstream_keywords": ["lifted QoI", "right-inverse property", "template lifting identity"],
        "suggested_hub": HUB,
        "setting": ["multi-scale methods"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Right-Inverse Property",
        "type": THEOREM,
        "statement_latex": r"\[\text{aggregate}\bigl(\mathcal{L}(\bar{m})\bigr) = \bar{m}\]",
        "assumptions": "Holds for both uniform and template lifting modes.",
        "conclusion": (
            "Aggregation after lifting recovers original coarse vector exactly. "
            "Verified at $1.4\\times10^{-17}$ (machine precision)."
        ),
        "interpretation": "Required for lifted QoI to be a valid approximation to the fine QoI.",
        "canonical_keywords": ["right-inverse", "aggregation-lifting", "consistency", "roundtrip"],
        "prereq_keywords": ["uniform lifting", "template lifting"],
        "downstream_keywords": ["lifted QoI"],
        "suggested_hub": HUB,
        "setting": ["multi-scale methods", "operator theory"],
        "named_tools": [],
        "result_category": "structural property",
    },

    # ── 8. QoI Approximation and Error Analysis ───────────────────────────────

    {
        "title": "Coarse QoI J-coarse",
        "type": DEFINITION,
        "statement_latex": (
            r"\[J_{\text{coarse}} = \Delta t\sum_{n=0}^{M-1}\sum_{k=1}^K"
            r"\bar{\ell}_k(t^n)\bar{m}_k(t^n) + \sum_{k=1}^K\bar{\ell}_{T,k}\bar{m}_k(T)\]"
            r"\[\bar{\ell}_k(t) = |C_k|^{-1}\sum_{i\in C_k}\ell(t,x_i)\]"
        ),
        "assumptions": "Computed entirely from coarse quantities, no lifting.",
        "conclusion": (
            "Cheapest QoI estimate. Systematically biased when $\\ell$ not constant within clusters. "
            "Algebraically equal to $J_{\\text{lifted}}^{\\text{unif}}$."
        ),
        "interpretation": "Standard Galerkin projection of the objective functional.",
        "canonical_keywords": ["coarse QoI", "Galerkin projection", "QoI approximation"],
        "prereq_keywords": ["coarse probability vector", "QoI kernel", "state space partition"],
        "downstream_keywords": ["error certificate", "lifted QoI"],
        "suggested_hub": HUB,
        "setting": ["goal-oriented analysis"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "Lifted QoI J-lifted",
        "type": DEFINITION,
        "statement_latex": (
            r"\[J_{\text{lifted}} = \int_0^T\int_{\mathbb{T}^1}\ell(t,x)\tilde{m}(t,x)\,dx\,dt"
            r"+ \int_{\mathbb{T}^1}\ell_T(x)\tilde{m}(T,x)\,dx, \quad"
            r"\tilde{m} = \tilde{m}^{\text{template}}\]"
        ),
        "assumptions": "Uses template lifting. When $\\tilde{m}^{\\text{template}} = m^*$: $J_{\\text{lifted}} = J^*$ exactly.",
        "conclusion": (
            "More accurate than $J_{\\text{coarse}}$ with template lifting. Primary reported metric. "
            "Error bounded by $\\mathcal{E}(K) + O(\\delta)$ for query ICs within $\\delta$ of reference."
        ),
        "interpretation": "Applies fine-resolution structure of $\\ell$ to lifted trajectory.",
        "canonical_keywords": ["lifted QoI", "template lifting", "QoI approximation"],
        "prereq_keywords": ["template lifting", "QoI kernel", "error certificate"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["goal-oriented analysis"],
        "named_tools": [],
        "result_category": "definition",
    },

    {
        "title": "A Priori Error Certificate E(K)",
        "type": THEOREM,
        "statement_latex": (
            r"\[|J_{\text{coarse}}-J_{\text{fine}}| \leq \mathcal{E}(K)"
            r"= \sum_{k=1}^K\operatorname{osc}(s_\eta;C_k)\cdot\bar{m}_k(0)\]"
        ),
        "assumptions": "Derived from duality between QoI functional and adjoint signature.",
        "conclusion": (
            "Computable upper bound on QoI approximation error, evaluable from offline quantities. "
            "Tight when $s_\\eta$ approximately linear within clusters. "
            "Vanishes at $K=N$. Key theoretical contribution."
        ),
        "interpretation": "Partition algorithm minimises $\\mathcal{E}(K)$ directly by minimising intra-cluster oscillation.",
        "canonical_keywords": ["error certificate", "a priori bound", "QoI error", "goal-oriented"],
        "prereq_keywords": ["adjoint influence signature", "state space partition", "QoI linearity"],
        "downstream_keywords": ["intra-cluster oscillation", "contiguous k-means"],
        "suggested_hub": HUB,
        "setting": ["a priori error analysis", "goal-oriented FEM"],
        "named_tools": [],
        "result_category": "error bound",
    },

    {
        "title": "Intra-Cluster Oscillation",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\operatorname{osc}(s_\eta;C_k)"
            r"= \max_{i\in C_k}s_\eta(x_i) - \min_{i\in C_k}s_\eta(x_i)\]"
        ),
        "assumptions": "For smooth $s_\\eta$ and contiguous clusters of width $h_K=1/K$: $\\operatorname{osc}\\sim h_K\\|s_\\eta'\\|_\\infty = O(1/K)$.",
        "conclusion": (
            "Measures sensitivity variation within cluster $k$. "
            "Partition algorithm minimises $\\sum_k \\operatorname{osc}(s_\\eta;C_k)\\cdot\\bar{m}_k(0)$, "
            "which is the error certificate. Gives $\\mathcal{E}(K) = O(1/K)$."
        ),
        "interpretation": "Direct driver of partition quality.",
        "canonical_keywords": ["oscillation", "cluster variation", "partition quality", "sensitivity"],
        "prereq_keywords": ["adjoint influence signature", "state space partition"],
        "downstream_keywords": ["error certificate", "contiguous k-means"],
        "suggested_hub": HUB,
        "setting": ["approximation theory"],
        "named_tools": [],
        "result_category": "definition",
    },

    # ── 9. Partition Algorithms ───────────────────────────────────────────────

    {
        "title": "Contiguous K-Means (cont_kmeans)",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\text{K-means on }\{s_\eta(x_i)\}_{i=0}^{N-1}\text{ constrained to contiguous intervals in 1D}\]"
            r"\[\text{Optimal 1D k-means solvable in }O(KN)\text{ by dynamic programming}\]"
        ),
        "assumptions": "Each $C_k$ is a contiguous interval $[a_k,b_k]$ in 1D.",
        "conclusion": (
            "Primary partition method. Produces physically interpretable partitions. "
            "Headline error: 0.32% on DoubleWell at $K=4$ (Ablation 8). "
            "50x improvement over uniform partition."
        ),
        "interpretation": "Used in all primary ablation results.",
        "canonical_keywords": ["k-means", "contiguous", "partition", "dynamic programming", "1D"],
        "prereq_keywords": ["adjoint influence signature", "intra-cluster oscillation"],
        "downstream_keywords": ["error certificate"],
        "suggested_hub": HUB,
        "setting": ["combinatorial optimisation", "clustering"],
        "named_tools": ["dynamic programming", "k-means"],
        "result_category": "algorithm",
    },

    {
        "title": "Spectral / Hierarchical Partition",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\text{Hierarchical agglomerative clustering on 1D graph with edge weights }|\Delta s_\eta|\]"
            r"\[\text{Merge adjacent clusters minimising increase in }\sum_k\operatorname{osc}(s_\eta;C_k)\]"
        ),
        "assumptions": "Agglomerative on 1D graph.",
        "conclusion": (
            "Alternative to contiguous k-means. Handles multi-modal $s_\\eta$. "
            "8x improvement over uniform in signature diagnostics ablation."
        ),
        "interpretation": "Tends to isolate narrow high-gradient regions of $s_\\eta$ into fine clusters.",
        "canonical_keywords": ["hierarchical clustering", "spectral partition", "agglomerative", "graph"],
        "prereq_keywords": ["adjoint influence signature", "intra-cluster oscillation"],
        "downstream_keywords": ["error certificate"],
        "suggested_hub": HUB,
        "setting": ["clustering", "graph methods"],
        "named_tools": ["Ward linkage", "normalised cuts"],
        "result_category": "algorithm",
    },

    {
        "title": "Value-Based Partition",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\text{Apply contiguous k-means to }\{u^*(0,x_i)\}\text{ instead of }\{s_\eta(x_i)\}\]"
        ),
        "assumptions": "Uses value function $u^*(0,x)$ as proxy for sensitivity.",
        "conclusion": (
            "Baseline comparison. Herding scenario: error = 0.1756 vs 0.0119 for adjoint-based partition. "
            "Value function is a poor sensitivity proxy."
        ),
        "interpretation": "$u^*$ measures agent cost-to-go, not QoI sensitivity.",
        "canonical_keywords": ["value-based partition", "baseline", "value function proxy"],
        "prereq_keywords": ["MFG equilibrium", "contiguous k-means"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["clustering"],
        "named_tools": [],
        "result_category": "algorithm",
    },

    {
        "title": "Uniform and Random Partitions",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\text{Uniform: }|C_k|=N/K\text{ for all }k\]"
            r"\[\text{Random: randomly assigned node-to-cluster labels}\]"
        ),
        "assumptions": "Lower bounds on partition quality.",
        "conclusion": (
            "Any method claiming goal-oriented performance must beat both. "
            "DoubleWell Ablation 8: uniform error = 0.16, random = 0.12, adjoint = 0.003. "
            "50x gap over uniform."
        ),
        "interpretation": "Uniform = no information baseline; random = robustness stress test.",
        "canonical_keywords": ["uniform partition", "random partition", "baseline", "lower bound"],
        "prereq_keywords": ["state space partition"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["clustering"],
        "named_tools": [],
        "result_category": "baseline",
    },

    # ── 10. Many-Query Engine ─────────────────────────────────────────────────

    {
        "title": "Offline / Online Decomposition",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\text{Offline (}{\approx}2T_{\text{fine}}\text{, once): fine solve}\to\text{adjoint}"
            r"\to\text{partition}\to\bar{Q}(t)\to\bar{u}^*\]"
            r"\[\text{Online (}O(K^2M)\text{ per query): aggregate}\to\text{FP with fixed }\bar{Q}"
            r"\to\text{lift}\to J\]"
        ),
        "assumptions": (
            "Offline cost amortised over $Q$ queries. "
            "Break-even at $Q^* = T_{\\text{build}}/(T_{\\text{fine}}-T_{\\text{query}})$. "
            "At $N=64$, $K=4$: $Q^*\\approx 4$."
        ),
        "conclusion": "Total cost $T_{\\text{build}} + Q\\,T_{\\text{query}} \\ll Q\\,T_{\\text{fine}}$ for large $Q$.",
        "interpretation": "Core amortisation strategy. Analogous to reduced-order modelling.",
        "canonical_keywords": ["offline online", "amortisation", "many-query", "reduced order"],
        "prereq_keywords": ["MFG equilibrium", "adjoint influence signature", "state space partition", "coarse generator"],
        "downstream_keywords": ["fixed-HJB forward pass", "projection gate"],
        "suggested_hub": HUB,
        "setting": ["reduced-order modelling", "many-query"],
        "named_tools": [],
        "result_category": "algorithm",
    },

    {
        "title": "Fixed-HJB Forward Pass",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\bar{m}^q(t+1) = \text{FP\_step}(\bar{m}^q(t),\bar{u}^*(t)),"
            r"\quad\bar{m}^q(0)=\text{agg}(m_0^q)\]"
            r"\[\text{Cost per query: }O(K^2M)\]"
        ),
        "assumptions": (
            "Pre-computed $\\bar{u}^*(t)$ from offline stage. Valid when $u^*$ weakly dependent on $m_0$. "
            "Exact for zero coupling, approximate for herding."
        ),
        "conclusion": "Core algorithmic claim: query without HJB re-solve. Invoked via query_mode=\"fixed_hjb\".",
        "interpretation": (
            "closed_pi mode re-solves HJB per query ($O(K^2MI_{\\text{PI}})$) — reserved for strongly coupled scenarios."
        ),
        "canonical_keywords": ["fixed HJB", "forward pass", "online query", "no re-solve"],
        "prereq_keywords": ["offline/online decomposition", "coarse HJB", "IMEX coarse split"],
        "downstream_keywords": ["projection gate", "lifted QoI"],
        "suggested_hub": HUB,
        "setting": ["reduced-order modelling"],
        "named_tools": [],
        "result_category": "algorithm",
    },

    {
        "title": "Projection Gate",
        "type": ALGORITHM,
        "statement_latex": (
            r"\[\text{score}(m_0^q) = \|m_1^{\text{fine}}(m_0^q)"
            r"- \tilde{m}_1^{\text{coarse}}(m_0^q)\|_{L^1} \leq \tau \Rightarrow \text{accept}\]"
        ),
        "assumptions": (
            "One-step flow mismatch between fine and coarse dynamics. "
            "Gate score must be calibrated: Spearman rank correlation $> 0.6$ with actual QoI error."
        ),
        "conclusion": (
            "Detects out-of-distribution queries where coarse approximation is unreliable. "
            "Falls back to fine solver on rejection, guaranteeing worst-case accuracy."
        ),
        "interpretation": "Analogous to trust-region constraints in optimisation-based reduced-order models.",
        "canonical_keywords": ["projection gate", "out-of-distribution", "trust region", "fallback"],
        "prereq_keywords": ["fixed-HJB forward pass", "template lifting"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["reduced-order modelling", "reliability"],
        "named_tools": [],
        "result_category": "algorithm",
    },

    {
        "title": "Query Family Assumption A3",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\|m_0^q - m_0^{\text{ref}}\|_{L^1} \leq \delta \quad \forall\,q\]"
        ),
        "assumptions": "$\\delta \\ll 1$ fixed at analysis time.",
        "conclusion": (
            "Sufficient condition for fixed-HJB approximation to be valid. "
            "Error bounded by $\\mathcal{E}(K) + O(\\delta)$."
        ),
        "interpretation": (
            "Correct query family for DoubleWell: bimodal ICs with varying well-weight $w\\in[0.35,0.65]$. "
            "Random Gaussian mixtures violate A3 at $K=4$."
        ),
        "canonical_keywords": ["query assumption", "locality", "L1 ball", "many-query"],
        "prereq_keywords": ["offline/online decomposition", "error certificate"],
        "downstream_keywords": ["fixed-HJB forward pass"],
        "suggested_hub": HUB,
        "setting": ["reduced-order modelling"],
        "named_tools": [],
        "result_category": "assumption",
    },

    # ── 11. Verification Objects ──────────────────────────────────────────────

    {
        "title": "Fine Solver Conservation Laws (Level 0)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\max_n\bigl|\textstyle\sum_i m_i^n\cdot h - 1\bigr| < 10^{-10},"
            r"\quad\min_{n,i}m_i^n \geq -10^{-12},"
            r"\quad\max_i|u_i^M - g(x_i,m^M)| < 10^{-10}\]"
        ),
        "assumptions": "Run after every solve as assertions.",
        "conclusion": (
            "Zero-cost checks detecting structural bugs in FP and HJB. "
            "Results: mass $4.4\\times10^{-16}$, positivity min $m=1.76\\times10^{-3}$, terminal HJB $0.00$."
        ),
        "interpretation": "Gate check for all downstream computation.",
        "canonical_keywords": ["conservation laws", "mass conservation", "positivity", "verification"],
        "prereq_keywords": ["MFG equilibrium", "IMEX scheme", "semi-implicit HJB"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical verification"],
        "named_tools": [],
        "result_category": "verification",
    },

    {
        "title": "IMEX Convergence Rate Check (Level 2)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\alpha_i = \log_2\!\left(\frac{|J(N_i,M_i)-J_{\text{ref}}|}"
            r"{|J(N_{i+1},M_{i+1})-J_{\text{ref}}|}\right), \quad N_i=2^{i+4},\; M_i=2N_i\]"
        ),
        "assumptions": "$J_{\\text{ref}}$ at $(N_{\\text{ref}},M_{\\text{ref}})=(512,1024)$.",
        "conclusion": (
            "Verifies first-order IMEX accuracy: expected $\\alpha_i\\approx 1.0$. "
            "Results: $\\alpha=0.98,1.32,1.48$ for $N=64,128,256$."
        ),
        "interpretation": "$N=32$ pre-asymptotic, excluded from rate computation.",
        "canonical_keywords": ["convergence rate", "IMEX", "grid refinement", "order of accuracy"],
        "prereq_keywords": ["IMEX scheme", "QoI J"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical verification"],
        "named_tools": [],
        "result_category": "verification",
    },

    {
        "title": "Adjoint Finite-Difference Gradient Check (Level 4)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\frac{J(m_0+\varepsilon v)-J(m_0-\varepsilon v)}{2\varepsilon}"
            r"\approx \langle s_\eta, v\rangle_{L^2} = h\sum_i s_\eta(x_i)v_i\]"
            r"\[\text{err} = |\text{FD} - \langle s_\eta,v\rangle|/|\text{FD}|\]"
        ),
        "assumptions": "Random mass-preserving directions $v$ with $\\int v=0$, $\\|v\\|=1$.",
        "conclusion": (
            "Definitive check that adjoint is correctly implemented. "
            "8/8 directions pass at $N=128$, mean error 1.63%, max 5.0%. "
            "Convergence $O(h)$: 4/8 at $N=32$, 6/8 at $N=64$, 8/8 at $N=128$."
        ),
        "interpretation": "Verifies $s_\\eta = \\eta(0,\\cdot)$ is the true functional derivative.",
        "canonical_keywords": ["adjoint check", "finite difference", "gradient verification"],
        "prereq_keywords": ["adjoint influence signature", "QoI J"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical verification"],
        "named_tools": [],
        "result_category": "verification",
    },

    {
        "title": "Trivial Partition Generator Check (Coarse Level 1)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\max_{k,l,t}|\bar{Q}_{kl}(t) - Q^{\text{fine}}_{kl}(t)| < 10^{-8}"
            r"\quad\text{at }K=N\text{ (singleton clusters)}\]"
        ),
        "assumptions": "$K=N$: one node per cluster, singleton partition.",
        "conclusion": (
            "Strongest single check for coarse operator builder. "
            "Initial failure at $3.19\\times10^{12}$ (missing density weighting). "
            "After fix: $2.27\\times10^{-13}$."
        ),
        "interpretation": "Gate check for all downstream coarse validation.",
        "canonical_keywords": ["trivial partition", "generator check", "coarse operator", "verification"],
        "prereq_keywords": ["coarse generator", "discrete grid"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical verification"],
        "named_tools": [],
        "result_category": "verification",
    },

    {
        "title": "Template Lifting Identity (Coarse Level 5)",
        "type": THEOREM,
        "statement_latex": (
            r"\[\max_{t,x}|\tilde{m}^{\text{template}}(t,x) - m^*(t,x)| < 10^{-8}"
            r"\quad\text{when }\bar{m}_k(t) = \int_{C_k}m^*(t,x)\,dx\]"
        ),
        "assumptions": "$\\bar{m}$ is the aggregated reference trajectory.",
        "conclusion": (
            "Lifting operator invertible on image of aggregation: recovers $m^*$ exactly. "
            "Result: $4.44\\times10^{-16}$ (machine precision)."
        ),
        "interpretation": "Confirms lifting implementation is exact.",
        "canonical_keywords": ["template lifting identity", "invertibility", "machine precision", "verification"],
        "prereq_keywords": ["template lifting", "right-inverse property"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical verification"],
        "named_tools": [],
        "result_category": "verification",
    },

    {
        "title": "PI Uniqueness Check (Coarse Level 4)",
        "type": DEFINITION,
        "statement_latex": (
            r"\[\|\bar{m}^{(\text{zero})}-\bar{m}^{(\text{unif})}\|_\infty < 10^{-4},"
            r"\quad\|\bar{m}^{(\text{zero})}-\bar{m}^{(\text{agg})}\|_\infty < 10^{-4}\]"
            r"\[\text{Three initialisations: }\bar{u}^0=0,\;\bar{u}^0=1/K,\;\bar{u}^0=\bar{u}^*\]"
        ),
        "assumptions": "All three from same $\\bar{m}^0 = \\text{agg}(m_0^{\\text{ref}})$.",
        "conclusion": (
            "Verifies Lasry-Lions uniqueness at coarse level. Failure indicates multiple coarse equilibria "
            "or degenerate PI initialisation. DoubleWell: all three converge to same equilibrium."
        ),
        "interpretation": "Warm-starting from aggregated fine solution recommended in practice.",
        "canonical_keywords": ["PI uniqueness", "multiple equilibria", "Lasry-Lions coarse", "verification"],
        "prereq_keywords": ["Picard iteration", "Lasry-Lions monotonicity", "coarse PI residual"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical verification"],
        "named_tools": [],
        "result_category": "verification",
    },

    # ── 12. Scenarios ─────────────────────────────────────────────────────────

    {
        "title": "Hard:DoubleWell Scenario",
        "type": DEFINITION,
        "statement_latex": (
            r"\[V(x,m)=10(x-0.25)^2(x-0.75)^2,\quad\nu=0.05,\;T=1.0,\;N=64,\;M=80\]"
            r"\[m_0=\tfrac{1}{2}\mathcal{N}(0.25,0.06^2)+\tfrac{1}{2}\mathcal{N}(0.75,0.06^2)\]"
        ),
        "assumptions": "Zero mean-field coupling ($D_m F=0$).",
        "conclusion": (
            "Two stable attractors. Strong trapping ($\\beta=10$) causes $s_\\eta$ to be flat "
            "(range $\\approx 3\\times10^{-3}$). Correct verification scenario, wrong demonstration scenario."
        ),
        "interpretation": "Use asymmetrically-placed narrow $\\ell$ and bimodal weight-varying query family for demonstration.",
        "canonical_keywords": ["DoubleWell", "scenario", "zero coupling", "verification"],
        "prereq_keywords": ["DoubleWell coupling", "bimodal IC"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical experiments"],
        "named_tools": [],
        "result_category": "scenario",
    },

    {
        "title": "DefaultHerding Scenario",
        "type": DEFINITION,
        "statement_latex": (
            r"\[V(x,m)=\tfrac{1}{2}(x-\bar{x}_m)^2,\quad\nu=0.1,\;T=1.0,\;N=64,\;M=80\]"
            r"\[m_0=\mathcal{N}(0.3,0.06^2)\text{ (normalised)}\]"
        ),
        "assumptions": "Nonzero mean-field coupling.",
        "conclusion": (
            "Tests full adjoint pipeline including $D_m F$. Produces meaningful $s_\\eta$ "
            "(peaked near QoI location) suitable for method demonstration. "
            "Lasry-Lions monotonicity satisfied at fine level, may not hold under coarse aggregation."
        ),
        "interpretation": "Standard LQ-MFG benchmark. Primary demonstration scenario.",
        "canonical_keywords": ["herding scenario", "nonlocal coupling", "demonstration", "benchmark"],
        "prereq_keywords": ["herding coupling", "Gaussian IC"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical experiments"],
        "named_tools": [],
        "result_category": "scenario",
    },

    {
        "title": "Bimodal Initial Condition",
        "type": DEFINITION,
        "statement_latex": (
            r"\[m_0^{(w)}(x) = w\,\mathcal{N}(x;a,\sigma^2)+(1-w)\mathcal{N}(x;b,\sigma^2),"
            r"\quad a=0.25,\;b=0.75,\;\sigma=0.06\]"
            r"\[\text{Reference: }w=0.5,\quad\text{Query family: }w\sim\text{Uniform}(0.35,0.65)\]"
        ),
        "assumptions": "Natural IC for DoubleWell.",
        "conclusion": (
            "Query weight $w$ controls mass between two wells — the one DOF surviving $K=4$ aggregation. "
            "Correct query family for DoubleWell under Assumption A3."
        ),
        "interpretation": "$L^1$ from reference: $w=0.35\\to L^1\\approx0.28$; $w=0.45\\to L^1\\approx0.09$.",
        "canonical_keywords": ["bimodal IC", "initial condition", "two-well", "query family"],
        "prereq_keywords": ["DoubleWell scenario", "query family assumption A3"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical experiments"],
        "named_tools": [],
        "result_category": "scenario",
    },

    {
        "title": "Gaussian Initial Condition",
        "type": DEFINITION,
        "statement_latex": (
            r"\[m_0(x) = \mathcal{N}(x;x_0,\sigma^2),\quad"
            r"x_0\in[0,1],\;\sigma\in\{0.06,0.1\}\]"
        ),
        "assumptions": "Normalised to unit mass. $x_0$ not at periodic boundary.",
        "conclusion": (
            "Standard unimodal IC for herding scenario. At $\\sigma=0.06$, $N=64$: "
            "support on ~8 grid points ($\\pm 2\\sigma/h$)."
        ),
        "interpretation": "Smooth, positive, easy to perturb for query generation.",
        "canonical_keywords": ["Gaussian IC", "unimodal", "initial condition"],
        "prereq_keywords": ["herding scenario"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical experiments"],
        "named_tools": [],
        "result_category": "scenario",
    },

    {
        "title": "Convex Mixture Query IC",
        "type": DEFINITION,
        "statement_latex": (
            r"\[m_0^q = (1-\alpha)m_0^{\text{ref}} + \alpha\rho,"
            r"\quad\alpha\sim\text{Uniform}(0,0.08)\]"
            r"\[\|m_0^q - m_0^{\text{ref}}\|_{L^1} \leq 2\alpha < 0.16\]"
        ),
        "assumptions": "$\\rho$ is a Gaussian noise IC at randomly chosen centre.",
        "conclusion": (
            "Generates query ICs with controlled $L^1$ distance. "
            "Structurally problematic for DoubleWell: noise component places mass in "
            "high-$s_\\eta$-gradient regions unresolvable at $K=4$."
        ),
        "interpretation": "Use bimodal weight-variation family for DoubleWell instead.",
        "canonical_keywords": ["convex mixture", "query IC", "controlled perturbation", "L1 ball"],
        "prereq_keywords": ["query family assumption A3", "Gaussian IC"],
        "downstream_keywords": [],
        "suggested_hub": HUB,
        "setting": ["numerical experiments"],
        "named_tools": [],
        "result_category": "scenario",
    },
]


# ── Link ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 + 3 edge linking for existing KI concepts of a paper"
    )
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Notion page ID of the Paper Tracker entry whose KI concepts to link",
    )
    parser.add_argument(
        "--relink",
        action="store_true",
        help="Re-run Stage 2/3 even for pages already marked linked-ai",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pages found, do not call OpenAI or write to Notion",
    )
    args = parser.parse_args()

    # ── Initialise engines ────────────────────────────────────────────────────
    vector_index: VectorIndexEngine | None = None
    if os.environ.get("VECTOR_INDEX_ENABLED", "").lower() in ("1", "true"):
        try:
            vector_index = VectorIndexEngine()
            logger.info("VectorIndexEngine initialised (available=%s)", vector_index.available)
        except Exception:
            logger.warning("VectorIndexEngine init failed — falling back to TF-IDF retrieval.")

    engine = IngestionEngine(vector_index=vector_index)

    # ── Fetch all KI pages for this paper ─────────────────────────────────────
    logger.info("Querying Knowledge Inbox for paper %s ...", args.paper_id)
    ki_pages = engine.notion.query_database(
        engine.knowledge_inbox_db,
        filter={"property": "Source Paper", "relation": {"contains": args.paper_id}},
    )
    logger.info("Found %d KI page(s) for paper.", len(ki_pages))

    if not ki_pages:
        logger.warning("No KI pages found for paper %s. Nothing to link.", args.paper_id)
        return

    # Keep the full list for use as candidates regardless of link status.
    ki_pages_all = ki_pages

    # ── Filter to unlinked pages unless --relink ──────────────────────────────
    def _get_select(props: dict, key: str) -> str:
        try:
            return props[key]["select"]["name"] or ""
        except (KeyError, TypeError):
            return ""

    if not args.relink:
        before = len(ki_pages)
        ki_pages = [
            p for p in ki_pages
            if _get_select(p["properties"], "Graph Link Status") != "linked-ai"
        ]
        skipped = before - len(ki_pages)
        if skipped:
            logger.info(
                "Skipping %d already linked-ai page(s). Pass --relink to force re-linking.",
                skipped,
            )

    if not ki_pages:
        logger.info("All pages already linked. Pass --relink to force re-linking. Exiting.")
        return

    if args.dry_run:
        logger.info("[DRY-RUN] Would run Stage 2+3 on %d page(s):", len(ki_pages))
        for p in ki_pages:
            title = ""
            try:
                for v in p["properties"].values():
                    if v.get("type") == "title":
                        title = v["title"][0]["plain_text"]
                        break
            except Exception:
                pass
            link_status = _get_select(p["properties"], "Graph Link Status")
            logger.info("  [DRY-RUN] '%s' (link_status=%s) — %s", title, link_status, p["id"])
        return

    # ── Build candidate pool ──────────────────────────────────────────────────
    # Reconstruct (MathObject, page_id) pairs from all KI pages for this paper.
    # These are injected into the search index so concepts within this paper
    # can link to each other even before Qdrant is running.
    ki_pairs: list[tuple] = []
    for page in ki_pages_all:
        mo = VectorIndexEngine._math_object_from_ki_page(page)
        if mo is not None:
            ki_pairs.append((mo, page["id"]))

    sb_index: list[dict] = []
    if not (vector_index and vector_index.available):
        logger.info("Vector search unavailable — building TF-IDF candidate index ...")
        sb_index = engine._build_second_brain_index()
        # Add all KI pages for this paper so concepts can link to each other.
        engine._inject_ki_pages_into_index(ki_pairs, sb_index)
        logger.info(
            "Candidate pool ready: %d concept(s) (Second Brain + %d from this paper).",
            len(sb_index),
            len(ki_pairs),
        )
    else:
        # Index any KI pages not yet in Qdrant so vector search can find them.
        logger.info(
            "Indexing %d KI page(s) into Qdrant before linking ...", len(ki_pairs)
        )
        for mo, page_id in ki_pairs:
            try:
                vector_index.index_concept(mo, page_id, verified=False)
            except Exception:
                logger.warning(
                    "Qdrant indexing failed for page %s ('%s') — will still attempt linking.",
                    page_id, mo.title,
                )

    # ── Stage 2 + 3 for each KI page ─────────────────────────────────────────
    success, failed = 0, 0
    total = len(ki_pages)

    for i, page in enumerate(ki_pages, 1):
        ki_page_id = page["id"]
        try:
            concept = VectorIndexEngine._math_object_from_ki_page(page)
            if concept is None:
                logger.warning(
                    "[%02d/%02d] Page %s has no title — skipping.", i, total, ki_page_id
                )
                continue

            # Stage 2: retrieve candidates
            logger.info(
                "[%02d/%02d] Stage 2: retrieving candidates for '%s' ...",
                i, total, concept.title,
            )
            candidates = engine._retrieve_candidates_for_concept(
                concept, sb_index, current_page_id=ki_page_id
            )
            logger.info(
                "[%02d/%02d] %d candidate(s) found for '%s'.",
                i, total, len(candidates), concept.title,
            )

            # Stage 3: GPT edge linking
            logger.info(
                "[%02d/%02d] Stage 3: linking '%s' via LLM ...", i, total, concept.title
            )
            link_result = engine._run_stage_link(concept, candidates, run_id=ki_page_id)

            # Write Edge Suggestions + set Graph Link Status = linked-ai
            engine._update_knowledge_item_graph_data(ki_page_id, link_result)
            logger.info(
                "[%02d/%02d] Edges written for '%s' → %s.",
                i, total, concept.title, ki_page_id,
            )
            success += 1

        except Exception:
            logger.exception("[%02d/%02d] Failed to link page %s.", i, total, ki_page_id)
            failed += 1

        # Throttle to avoid hitting OpenAI RPM limits across 50+ concepts.
        if i < total:
            time.sleep(4)

    logger.info("Done. Linked: %d / %d. Failed: %d.", success, total, failed)
    if failed:
        logger.warning(
            "Some pages failed — re-run to retry "
            "(already-linked pages will be skipped unless --relink)."
        )


if __name__ == "__main__":
    main()
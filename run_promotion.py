"""
run_promotion.py — run PromotionEngine locally without Docker.

Polls Paper Tracker for papers at Status = s2-read and promotes all verified
Knowledge Inbox concepts to Second Brain + Edges DB.

Usage:
    python run_promotion.py
"""

import logging
import sys
from pathlib import Path

# ── Load .env before any module import ───────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Make `orchestrator/` importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from modules.promotion import PromotionEngine  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Run (no Qdrant — pass None for vector_index) ──────────────────────────────
PromotionEngine(vector_index=None).run()

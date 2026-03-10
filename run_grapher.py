"""
run_grapher.py — run DependencyGrapher locally without Docker.

Usage:
    python run_grapher.py [--out <dir>]

Outputs:
    <dir>/graph_verified.html
    <dir>/graph_inbox.html
    <dir>/index.html         ← open this in a browser

Defaults to ./graph_output/ if --out is not given.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Load .env before any module import so all env vars are available ──────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Make `orchestrator/` importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from orchestrator.modules.dependency_grapher import DependencyGrapher  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate knowledge-graph HTML locally.")
parser.add_argument(
    "--out",
    default="graph_output",
    help="Directory to write HTML files into (default: ./graph_output)",
)
args = parser.parse_args()

out_dir = Path(args.out).resolve()
os.environ["GRAPH_STATIC_DIR"] = str(out_dir)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Run ───────────────────────────────────────────────────────────────────────
DependencyGrapher().run()

print()
print(f"Done.  Open in browser:")
print(f"  {out_dir / 'index.html'}")

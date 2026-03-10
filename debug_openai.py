"""
debug_claude.py — minimal Anthropic connectivity + structured-output debug script.

Run from the project root:
    python debug_openai.py

Checks:
  1. ANTHROPIC_API_KEY is set and non-empty.
  2. A raw messages.create() call succeeds (auth + connectivity).
  3. An instructor-wrapped structured call succeeds (same path used by ingestion.py).
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv("C:\\Users\\blkv0u\\Desktop\\paper_pipeline\\.env")

import anthropic
import instructor
from pydantic import BaseModel

# ── 1. Key present? ───────────────────────────────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY is not set in .env")
    sys.exit(1)
print(f"OK  ANTHROPIC_API_KEY present (starts with '{api_key[:16]}...')")

# ── 2. Raw client ─────────────────────────────────────────────────────────────
raw_client = anthropic.Anthropic(api_key=api_key)

# ── 3. Raw messages.create() — auth + connectivity check ─────────────────────
print("\nSending raw messages.create() call ...")
try:
    t0 = time.monotonic()
    resp = raw_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say exactly: hello"}],
    )
    elapsed = time.monotonic() - t0
    print(f"OK  messages.create() succeeded in {elapsed:.2f}s")
    print(f"    Response: {resp.content[0].text!r}")
    print(f"    Stop reason: {resp.stop_reason}")
    print(f"    Usage: input={resp.usage.input_tokens} output={resp.usage.output_tokens} tokens")
except anthropic.AuthenticationError as e:
    print(f"FAIL  Authentication error — check ANTHROPIC_API_KEY: {e}")
    sys.exit(1)
except anthropic.RateLimitError as e:
    print(f"FAIL  Rate limit: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL  Unexpected error: {type(e).__name__}: {e}")
    sys.exit(1)

# ── 4. Instructor structured call — same path as ingestion.py ─────────────────
print("\nSending instructor structured call ...")

class _Tiny(BaseModel):
    answer: str
    confidence: float

instructor_client = instructor.from_anthropic(raw_client)

try:
    t0 = time.monotonic()
    result = instructor_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": 'Reply with {"answer": "hello", "confidence": 0.99}'}],
        response_model=_Tiny,
    )
    elapsed = time.monotonic() - t0
    print(f"OK  instructor call succeeded in {elapsed:.2f}s")
    print(f"    Parsed: answer={result.answer!r}  confidence={result.confidence}")
except anthropic.AuthenticationError as e:
    print(f"FAIL  Auth error: {e}")
    sys.exit(1)
except anthropic.RateLimitError as e:
    print(f"FAIL  Rate limit: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL  {type(e).__name__}: {e}")
    sys.exit(1)

print("\nAll checks passed — Claude is ready.")

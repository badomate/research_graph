"""
modules/latex_notion_parser.py
──────────────────────────────
Parses strings containing LaTeX math delimiters into Notion API block/
rich_text structures.

Supported delimiter forms
─────────────────────────
  Block-level  : $$...$$   \[...\]
  Inline        : $...$     \(...\)

Rules
─────
  • Block math   → standalone `equation` block  (no surrounding paragraph)
  • Inline math  → `equation` segment inside a `rich_text` array
  • Plain text   → `text` segment inside a `rich_text` array
  • A paragraph that contains ONLY block math is emitted as an equation
    block, not wrapped in a paragraph block.
  • A paragraph that mixes text + inline math is emitted as a single
    paragraph block with a rich_text array of alternating text/equation
    segments.
  • Empty segments are dropped.

Public API
──────────
  parse_to_blocks(text: str) -> list[dict]
      Top-level entry point.  Splits `text` on newlines, processes each
      line, and returns a flat list of Notion block dicts ready to pass
      to `blocks.children.append`.

  rich_text_segments(text: str) -> list[dict]
      Returns the rich_text array for a single line (inline math only).
      Useful when you need to embed math inside a property value rather
      than a block body.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ── Token types ───────────────────────────────────────────────────────────────

class _Seg(NamedTuple):
    kind: str   # "text" | "inline_eq" | "block_eq"
    content: str


# ── Regex patterns ────────────────────────────────────────────────────────────

# Order matters: longer / more specific patterns must come first.
_BLOCK_PATTERNS = [
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),          # $$...$$
    re.compile(r'\\\[(.+?)\\\]', re.DOTALL),           # \[...\]
]

_INLINE_PATTERNS = [
    re.compile(r'\\\((.+?)\\\)'),                      # \(...\)
    re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'),  # $...$ (non-$$)
]

# Combined scanner: block first, then inline, then plain text.
# Each group name encodes the kind.
_SCANNER = re.compile(
    r'(?P<block_dollar>\$\$(?P<bd_expr>.+?)\$\$)'
    r'|(?P<block_bracket>\\\[(?P<bb_expr>.+?)\\\])'
    r'|(?P<inline_paren>\\\((?P<ip_expr>.+?)\\\))'
    r'|(?P<inline_dollar>(?<!\$)\$(?!\$)(?P<id_expr>.+?)(?<!\$)\$(?!\$))',
    re.DOTALL,
)


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[_Seg]:
    """
    Split *text* into a sequence of _Seg tokens (text / inline_eq / block_eq).
    """
    segments: list[_Seg] = []
    cursor = 0

    for m in _SCANNER.finditer(text):
        # Plain text before this match
        if m.start() > cursor:
            raw = text[cursor:m.start()]
            if raw:
                segments.append(_Seg("text", raw))

        if m.group("block_dollar"):
            expr = (m.group("bd_expr") or "").strip()
            segments.append(_Seg("block_eq", expr))
        elif m.group("block_bracket"):
            expr = (m.group("bb_expr") or "").strip()
            segments.append(_Seg("block_eq", expr))
        elif m.group("inline_paren"):
            expr = (m.group("ip_expr") or "").strip()
            segments.append(_Seg("inline_eq", expr))
        elif m.group("inline_dollar"):
            expr = (m.group("id_expr") or "").strip()
            segments.append(_Seg("inline_eq", expr))

        cursor = m.end()

    # Trailing plain text
    if cursor < len(text):
        tail = text[cursor:]
        if tail:
            segments.append(_Seg("text", tail))

    return segments


# ── Notion block / rich_text builders ────────────────────────────────────────

def _text_rt(content: str) -> dict:
    """Plain-text rich_text segment (2000-char Notion limit per segment)."""
    return {"type": "text", "text": {"content": content[:2000]}}


def _eq_rt(expression: str) -> dict:
    """Inline equation rich_text segment."""
    return {"type": "equation", "equation": {"expression": expression}}


def _paragraph_block(rich_text: list[dict]) -> dict:
    return {"type": "paragraph", "paragraph": {"rich_text": rich_text}}


def _equation_block(expression: str) -> dict:
    return {"type": "equation", "equation": {"expression": expression}}


def _heading_block(text: str, level: int = 2) -> dict:
    key = f"heading_{level}"
    return {
        "type": key,
        key: {"rich_text": [_text_rt(text)]},
    }


# ── Line-level processing ─────────────────────────────────────────────────────

def _line_to_blocks(line: str) -> list[dict]:
    """
    Convert a single line of text (possibly containing LaTeX) into one or
    more Notion blocks.

    Decision tree
    ─────────────
    1. Tokenise the line.
    2. If the line is PURELY a single block_eq token → emit equation block.
    3. If the line contains block_eq tokens mixed with other content →
       split into sub-lines at each block_eq, emit equation blocks for
       the math and paragraph blocks for any surrounding text.
    4. Otherwise (inline math + text only) → emit a single paragraph
       block with a mixed rich_text array.
    """
    line = line.strip()
    if not line:
        return []

    segments = _tokenise(line)

    # Fast path: pure block equation
    if len(segments) == 1 and segments[0].kind == "block_eq":
        return [_equation_block(segments[0].content)]

    # Check whether any block_eq tokens are present
    has_block = any(s.kind == "block_eq" for s in segments)

    if not has_block:
        # Inline-only: build a rich_text paragraph
        rt = _segs_to_rich_text(segments)
        return [_paragraph_block(rt)] if rt else []

    # Mixed: flush pending inline/text segments as a paragraph, then emit
    # each block_eq as a standalone equation block.
    blocks: list[dict] = []
    pending: list[_Seg] = []

    def _flush_pending():
        if pending:
            rt = _segs_to_rich_text(pending)
            if rt:
                blocks.append(_paragraph_block(rt))
            pending.clear()

    for seg in segments:
        if seg.kind == "block_eq":
            _flush_pending()
            blocks.append(_equation_block(seg.content))
        else:
            pending.append(seg)

    _flush_pending()
    return blocks


def _segs_to_rich_text(segments: list[_Seg]) -> list[dict]:
    """Convert a list of _Seg tokens (text/inline_eq only) to rich_text."""
    rt: list[dict] = []
    for seg in segments:
        if seg.kind == "text":
            # Split long text into ≤2000-char chunks
            content = seg.content
            while content:
                rt.append(_text_rt(content[:2000]))
                content = content[2000:]
        elif seg.kind == "inline_eq":
            rt.append(_eq_rt(seg.content))
        # block_eq should not appear here; skip defensively
    return rt


# ── Public API ────────────────────────────────────────────────────────────────

def parse_to_blocks(text: str) -> list[dict]:
    """
    Parse a (possibly multi-line) string containing LaTeX delimiters into a
    flat list of Notion block dicts.

    Parameters
    ----------
    text : str
        Raw string, e.g. from concept.statement_latex or concept.interpretation.
        May contain $...$, $$...$$, \\(...\\), \\[...\\] math spans.

    Returns
    -------
    list[dict]
        Ready to pass to NotionClientWrapper.append_block_children.
    """
    if not text:
        return []
    text = _normalize_multiline_delimiters(text)
    blocks: list[dict] = []
    for line in text.splitlines():
        blocks.extend(_line_to_blocks(line))
    return blocks


def _normalize_multiline_delimiters(text: str) -> str:
    """
    Collapse multi-line block math expressions onto a single line so the
    line-by-line tokeniser can match them.

    Handles:
        \[          \[...\]  (possibly spanning N lines)
        \]
        $$          $$...$$  (possibly spanning N lines)
        $$
    """
    # \[...\] spanning multiple lines
    text = re.sub(r'\\\[\s*\n(.*?)\n\s*\\\]', 
                  lambda m: r'\[' + ' '.join(m.group(1).splitlines()) + r'\]',
                  text, flags=re.DOTALL)
    # $$...$$ spanning multiple lines
    text = re.sub(r'\$\$\s*\n(.*?)\n\s*\$\$',
                  lambda m: '$$' + ' '.join(m.group(1).splitlines()) + '$$',
                  text, flags=re.DOTALL)
    return text

def rich_text_segments(text: str) -> list[dict]:
    """
    Parse *text* (single line, inline math only) into a Notion rich_text array.

    Block-level delimiters ($$, \\[) are treated as inline equations here —
    use this only for property values, not block bodies.

    Parameters
    ----------
    text : str
        E.g. a concept title or short description field.

    Returns
    -------
    list[dict]
        A rich_text array suitable for title_prop, rich_text property values,
        or heading block rich_text fields.
    """
    if not text:
        return []
    segments = _tokenise(text)
    # Demote any block_eq to inline_eq for property context
    demoted = [
        _Seg("inline_eq", s.content) if s.kind == "block_eq" else s
        for s in segments
    ]
    return _segs_to_rich_text(demoted)


# ── Integration helpers (drop-in replacements for _paragraph_blocks) ──────────

def paragraph_blocks_from_latex(text: str) -> list[dict]:
    """
    Alias for parse_to_blocks.  Drop-in replacement for the existing
    `_paragraph_blocks` calls in ingestion.py:

        body_blocks.extend(paragraph_blocks_from_latex(concept.statement_latex))
    """
    return parse_to_blocks(text)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    cases = [
        # Pure block eq
        r"$$\rho_N^{-1} W_N - W \|_\square \to 0$$",
        # \[...\] block
        r"\[\rho_N^{-1} W_N - W\|_{L_\infty} \to L_1\]",
        # Inline only
        r"The sequence $W_N$ converges in cut norm $\|\cdot\|_\square$.",
        # Mixed inline
        r"Let $\alpha > 0$ and $W(x,y) = (1-\alpha)^2 (xy)^{-\alpha}$.",
        # Multi-line with block eq embedded
        "Assumption: $W \\in \\mathcal{W}_0$.\n"
        "$$\\rho_N^{-1} W_N - W\\|_\\square \\to 0$$\n"
        "This holds for all $N \\to \\infty$.",
        # \(...\) inline
        r"Define \(\mu_t\) as the marginal of \(X_t\).",
        # Plain text, no math
        "Not applicable (example family).",
    ]

    for i, c in enumerate(cases):
        print(f"\n── Case {i+1} ──────────────────────────────")
        print(f"Input : {c!r}")
        result = parse_to_blocks(c)
        print(f"Output: {json.dumps(result, indent=2)}")
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
    
    # NEW: if line starts with a LaTeX command and has no delimiters,
    # treat entire line as an equation block
    _BARE_LATEX_RE = re.compile(
        r'^(\\text\{|\\begin\{|\\partial|\\frac|\\sum|\\int|\\prod)'
    )
    if _BARE_LATEX_RE.match(line) and '$' not in line and r'\[' not in line:
        return [_equation_block(line)]
    
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
    # 1. Collapse multiline \[...\] onto one line (consumes \begin{} inside)
    text = re.sub(
        r'\\\[(.*?)\\\]',
        lambda m: r'\[' + ' '.join(m.group(1).splitlines()) + r'\]',
        text, flags=re.DOTALL
    )
    # 2. Collapse multiline $$...$$
    text = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: '$$' + ' '.join(m.group(1).splitlines()) + '$$',
        text, flags=re.DOTALL
    )
    # 3. Wrap bare \begin{env}...\end{env} not already inside \[...\]
    # Split on existing \[...\] tokens — only process segments between them.
    parts = re.split(r'(\\\[.*?\\\])', text, flags=re.DOTALL)
    processed = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside an existing \[...\] block — leave untouched
            processed.append(part)
        else:
            # Outside any \[...\] — wrap bare environments
            part = re.sub(
                r'(\\begin\{[a-z*]+\}.*?\\end\{[a-z*]+\})',
                lambda m: r'\[' + ' '.join(m.group(1).splitlines()) + r'\]',
                part, flags=re.DOTALL | re.IGNORECASE
            )
            processed.append(part)
    return ''.join(processed)

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

def sanitize_statement_latex(text: str) -> str:
    text = text.strip()

    # $$\begin{align}...\end{align}$$ → \[\begin{aligned}...\end{aligned}\]
    text = re.sub(
        r'^\$\$\s*(\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\})\s*\$\$$',
        r'\\[\1\\]',
        text
    )

    # $$\[...\]$$ → \[...\]  (redundant outer wrapper)
    text = re.sub(
        r'^\$\$\s*(\\[\s\S]*?\\])\s*\$\$$',
        r'\1',
        text
    )

    # bare $$...$$ with no \[ inside → \[...\]
    if text.startswith('$$') and text.endswith('$$') and r'\[' not in text:
        text = r'\[' + text[2:-2].strip() + r'\]'

    return text
# ── Self-test ─────────────────────────────────────────────────────────────────

# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    results = []

    def check(name, got, expect_fn, show=True):
        ok = expect_fn(got)
        status = PASS if ok else FAIL
        results.append((name, ok))
        if show or not ok:
            print(f"\n{status}  {name}")
            if not ok:
                print(f"       Got: {json.dumps(got, indent=2)}")
        return ok

    # ── 1. Pure $$...$$ → single equation block ───────────────────────────────
    check(
        "1. Pure $$...$$ → equation block",
        parse_to_blocks(r"$$\alpha \in [0,1]$$"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            r[0]["equation"]["expression"] == r"\alpha \in [0,1]"
        ),
    )

    # ── 2. Pure \[...\] → single equation block ───────────────────────────────
    check(
        r"2. Pure \[...\] → equation block",
        parse_to_blocks(r"\[\alpha \in [0,1]\]"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation"
        ),
    )

    # ── 3. Inline $...$ in prose → paragraph with mixed rich_text ────────────
    check(
        "3. Inline $...$ → paragraph with mixed rich_text",
        parse_to_blocks(r"Let $\alpha > 0$ be the node index."),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "paragraph" and
            any(s["type"] == "equation" for s in r[0]["paragraph"]["rich_text"]) and
            any(s["type"] == "text"     for s in r[0]["paragraph"]["rich_text"])
        ),
    )

    # ── 4. \(...\) inline → paragraph with equation segment ──────────────────
    check(
        r"4. \(...\) inline → equation segment in paragraph",
        parse_to_blocks(r"Define \(\mu_t\) as the flow measure."),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "paragraph" and
            any(s["type"] == "equation" for s in r[0]["paragraph"]["rich_text"])
        ),
    )

    # ── 5. Plain text, no math → paragraph block ─────────────────────────────
    check(
        "5. Plain text → paragraph block",
        parse_to_blocks("Not applicable (example family)."),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "paragraph" and
            r[0]["paragraph"]["rich_text"][0]["text"]["content"] == "Not applicable (example family)."
        ),
    )

    # ── 6. Multi-line \[...\] → single equation block (no newlines in expr) ──
    check(
        r"6. Multi-line \[...\] → collapsed to single equation block",
        parse_to_blocks("\\[\n\\alpha \\in [0,1]\n\\]"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "\n" not in r[0]["equation"]["expression"]
        ),
    )

    # ── 7. Multi-line $$...$$ → single equation block ────────────────────────
    check(
        "7. Multi-line $$...$$ → collapsed to single equation block",
        parse_to_blocks("$$\n\\alpha \\in [0,1]\n$$"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "\n" not in r[0]["equation"]["expression"]
        ),
    )

    # ── 8. \[\begin{aligned}...\end{aligned}\] → equation block ──────────────
    check(
        r"8. \[\begin{aligned}...\end{aligned}\] → equation block (not wrapped in $$)",
        parse_to_blocks(r"\[\begin{aligned} f(x) &= 0 \\ g(x) &= 1 \end{aligned}\]"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "begin{aligned}" in r[0]["equation"]["expression"]
        ),
    )

    # ── 9. Multi-line \[\begin{aligned}...\end{aligned}\] → collapsed ─────────
    check(
        r"9. Multi-line \[\begin{aligned}...\end{aligned}\] → single equation block",
        parse_to_blocks(
            "\\[\n\\begin{aligned}\nf(x) &= 0 \\\\\ng(x) &= 1\n\\end{aligned}\n\\]"
        ),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "begin{aligned}" in r[0]["equation"]["expression"] and
            "\n" not in r[0]["equation"]["expression"]
        ),
    )

    # ── 10. $$\begin{aligned}...\end{aligned}$$ → sanitized to \[...\] ────────
    check(
        "10. sanitize_statement_latex: $$\\begin{aligned}...\\end{aligned}$$ → \\[...\\]",
        parse_to_blocks(
            sanitize_statement_latex(
                r"$$\begin{aligned} f(x) &= 0 \\ g(x) &= 1 \end{aligned}$$"
            )
        ),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "begin{aligned}" in r[0]["equation"]["expression"]
        ),
    )

    # ── 11. $$\[...\]$$ → stripped to \[...\] ────────────────────────────────
    check(
        r"11. sanitize_statement_latex: $$\[...\]$$ → \[...\]",
        parse_to_blocks(sanitize_statement_latex(r"$$\[\alpha \in [0,1]\]$$")),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation"
        ),
    )

    # ── 12. bare $$...$$ (no \[) → \[...\] ───────────────────────────────────
    check(
        r"12. sanitize_statement_latex: bare $$...$$ → \[...\]",
        list(map(lambda s: s(sanitize_statement_latex(r"$$\alpha \in [0,1]$$")), [
            lambda t: t.startswith(r'\['),
            lambda t: t.endswith(r'\]'),
        ])),
        lambda r: all(r),
    )

    # ── 13. Bare \begin{aligned} (no outer delimiter) → wrapped in \[...\] ───
    check(
        r"13. Bare \begin{aligned} → wrapped in \[...\]",
        parse_to_blocks(
            "\\begin{aligned}\nf(x) &= 0 \\\\\ng(x) &= 1\n\\end{aligned}"
        ),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "begin{aligned}" in r[0]["equation"]["expression"]
        ),
    )

    # ── 14. Bare \partial... line → equation block ────────────────────────────
    check(
        r"14. Bare \partial... → equation block",
        parse_to_blocks(r"\partial_\alpha f_\ell(\alpha^*)=0"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation"
        ),
    )

    # ── 15. Bare \text{...} line → equation block ─────────────────────────────
    check(
        r"15. Bare \text{...} → equation block",
        parse_to_blocks(r"\text{If (A1)--(A6) hold and there exists }\alpha^*\in(0,1)"),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation"
        ),
    )

    # ── 16. Mixed: prose + block eq + prose → 3 blocks ───────────────────────
    check(
        "16. Mixed multi-line: paragraph + equation + paragraph",
        parse_to_blocks(
            "Assumption: $W \\in \\mathcal{W}_0$.\n"
            "$$\\rho_N^{-1} W_N \\to 0$$\n"
            "This holds for all $N$."
        ),
        lambda r: (
            len(r) == 3 and
            r[0]["type"] == "paragraph" and
            r[1]["type"] == "equation" and
            r[2]["type"] == "paragraph"
        ),
    )

    # ── 17. rich_text_segments: block_eq demoted to inline ───────────────────
    check(
        r"17. rich_text_segments: \[...\] demoted to inline equation segment",
        rich_text_segments(r"Value: \[\alpha\]"),
        lambda r: (
            len(r) >= 1 and
            any(s["type"] == "equation" for s in r)
        ),
    )

    # ── 18. Empty string → empty list ────────────────────────────────────────
    check(
        "18. Empty string → empty list",
        parse_to_blocks(""),
        lambda r: r == [],
    )

    # ── 19. Multiple inline equations in one line ─────────────────────────────
    check(
        "19. Multiple inline equations → single paragraph with 3 equation segments",
        parse_to_blocks(r"Let $\alpha \in [0,1]$ and $\beta > 0$ with $\alpha + \beta = 1$."),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "paragraph" and
            sum(1 for s in r[0]["paragraph"]["rich_text"] if s["type"] == "equation") == 3
        ),
    )

    # ── 20. Real-world: graphon statement with \begin{aligned} inside \[...\] ─
    check(
        "20. Real-world: \\[\\begin{aligned}...\\end{aligned}\\] → single equation block",
        parse_to_blocks(
            "\\[\\begin{aligned}\n"
            "n\\|W\\|_p &= \\left(\\int_{[0,1]^2} W(x,y)^p\\,dx\\,dy\\right)^{\\frac{1}{p}} \\\\\n"
            "\\|W\\|_\\infty &:= \\sup_{(x,y)\\in[0,1]^2} |W(x,y)|\n"
            "\\end{aligned}\\]"
        ),
        lambda r: (
            len(r) == 1 and
            r[0]["type"] == "equation" and
            "begin{aligned}" in r[0]["equation"]["expression"] and
            "\n" not in r[0]["equation"]["expression"]
        ),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'═'*52}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("\n  Failed:")
        for name, ok in results:
            if not ok:
                print(f"    ✗ {name}")
    print(f"{'═'*52}")
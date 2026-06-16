"""
modules/parsing/skip_proof.py — propose a parse scope that skips long proofs.

Pure, offline-testable. Given a paper's sections (heading + page span), it
classifies each as *keep* (abstract, intro, problem formulation, theorem/
assumption statements, numerical experiments, conclusion) or *skip* (proofs,
technical lemmas, appendices, references), and returns a proposed page selection
with reasons. It never executes — the UI shows the proposal for edit/accept.

Sections can be supplied directly, parsed from simple "pages heading" lines a
user types from the PDF's table of contents, or derived from the chunks of a
prior parse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# heading keyword → human reason. Order matters: SKIP is checked before KEEP so
# "Proof of the main theorem" skips even though it contains "theorem".
_SKIP_RULES = [
    (re.compile(r"\bproof(s)?\b", re.I), "proof section"),
    (re.compile(r"\bderivation of\b", re.I), "long derivation"),
    (re.compile(r"\bappendix|appendices\b", re.I), "appendix"),
    (re.compile(r"\bsupplement(ary|al)?\b", re.I), "supplementary material"),
    (re.compile(r"\btechnical lemma|auxiliary (results|lemmas)\b", re.I), "technical lemmas"),
    (re.compile(r"\breferences|bibliography\b", re.I), "references"),
    (re.compile(r"\backnowledg", re.I), "acknowledgements"),
]

_KEEP_RULES = [
    (re.compile(r"\babstract\b", re.I), "abstract"),
    (re.compile(r"\bintroduction\b", re.I), "introduction"),
    (re.compile(r"\b(problem|model)\s+(formulation|setup|set-up|setting)\b", re.I), "problem formulation"),
    (re.compile(r"\bpreliminaries|notation\b", re.I), "preliminaries/notation"),
    (re.compile(r"\bassumptions?\b", re.I), "assumptions"),
    (re.compile(r"\b(main\s+)?(theorem|result|results|theory)\b", re.I), "main results"),
    (re.compile(r"\b(numerical|experiment|experiments|simulation|results)\b", re.I), "numerical experiments"),
    (re.compile(r"\b(conclusion|discussion|outlook|future work)\b", re.I), "conclusion"),
]


@dataclass
class Section:
    heading: str
    page_start: int
    page_end: int


def classify_heading(heading: str) -> tuple[str, str]:
    """Return ("keep"|"skip", reason). Unknown headings default to keep (safe)."""
    h = heading or ""
    for rule, reason in _SKIP_RULES:
        if rule.search(h):
            return "skip", reason
    for rule, reason in _KEEP_RULES:
        if rule.search(h):
            return "keep", reason
    return "keep", "unclassified (kept by default)"


def parse_section_lines(text: str) -> list[Section]:
    """Parse user-typed lines like '1-8 Introduction' or '35 Proof of Theorem 2'.

    Accepts 'START[-END] heading' or 'START[-END]: heading'. Lines without a
    leading page spec are ignored.
    """
    sections: list[Section] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*(?:-\s*(\d+))?\s*[:.\)]?\s+(.*)$", line)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        heading = m.group(3).strip()
        if heading:
            sections.append(Section(heading, min(start, end), max(start, end)))
    return sections


def sections_from_chunks(chunks: list, total_pages: int | None = None) -> list[Section]:
    """Build sections from prior-parse chunks (heading + page_from/page_to)."""
    sections: list[Section] = []
    for ch in chunks:
        heading = getattr(ch, "heading", "") or ""
        ps = getattr(ch, "page_from", None)
        pe = getattr(ch, "page_to", None)
        if heading and ps:
            sections.append(Section(heading, int(ps), int(pe or ps)))
    return sections


def propose_scope(sections: list[Section], total_pages: int | None = None) -> dict:
    """Classify sections → a proposed page selection + reasons.

    Returns a dict with selected/skipped page lists, the per-section decisions,
    and a ready-to-use ``scope_json`` (page_range kind).
    """
    keep_pages: set[int] = set()
    skip_pages: set[int] = set()
    decisions: list[dict] = []
    for sec in sections:
        decision, reason = classify_heading(sec.heading)
        pages = list(range(sec.page_start, sec.page_end + 1))
        decisions.append({
            "heading": sec.heading, "pages": [sec.page_start, sec.page_end],
            "decision": decision, "reason": reason,
        })
        (keep_pages if decision == "keep" else skip_pages).update(pages)

    # A page kept by one section wins over a skip from another (keep is inclusive).
    selected = sorted(keep_pages)
    skipped = sorted(p for p in skip_pages if p not in keep_pages)

    return {
        "selected_pages": selected,
        "skipped_pages": skipped,
        "decisions": decisions,
        "scope_json": {
            "kind": "page_range",
            "page_ranges": _to_ranges(selected),
            "regions": [],
        },
        "total_pages": total_pages,
    }


def _to_ranges(pages: list[int]) -> list[list[int]]:
    """Collapse a sorted page list into [[start,end], ...] inclusive ranges."""
    if not pages:
        return []
    ranges: list[list[int]] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append([start, prev])
        start = prev = p
    ranges.append([start, prev])
    return ranges


def ranges_to_string(ranges: list[list[int]]) -> str:
    """Render [[1,8],[35,40]] as '1-8, 35-40' for the scope-builder input."""
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)

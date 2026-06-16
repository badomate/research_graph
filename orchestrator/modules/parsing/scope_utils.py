"""
modules/parsing/scope_utils.py — pure helpers over a ParseScope's ``scope_json``.

``scope_json`` shape (pages are 1-indexed, inclusive)::

    {"kind": "mixed",
     "page_ranges": [[1, 8], [35, 40]],
     "regions": [{"page": 3, "bbox": [x0, y0, x1, y1], "label": "theorem"}]}

Region bboxes are normalized to page size ([0,1] floats), never pixels.
"""
from __future__ import annotations

from typing import Iterable


def _clamp_pages(pages: Iterable[int], total_pages: int | None) -> list[int]:
    out = sorted({p for p in pages if p >= 1 and (total_pages is None or p <= total_pages)})
    return out


def selected_pages(scope_json: dict, total_pages: int | None = None) -> list[int]:
    """Return the sorted, de-duplicated 1-indexed pages a scope selects.

    ``full`` needs ``total_pages`` to enumerate; without it returns [].
    """
    kind = (scope_json or {}).get("kind", "full")
    if kind == "full":
        return list(range(1, total_pages + 1)) if total_pages else []

    pages: set[int] = set()
    for rng in (scope_json.get("page_ranges") or []):
        if not rng:
            continue
        lo, hi = int(rng[0]), int(rng[-1])
        if lo > hi:
            lo, hi = hi, lo
        pages.update(range(lo, hi + 1))
    for region in (scope_json.get("regions") or []):
        page = region.get("page")
        if isinstance(page, int):
            pages.add(page)
    return _clamp_pages(pages, total_pages)


def page_count(scope_json: dict, total_pages: int | None = None) -> int:
    return len(selected_pages(scope_json, total_pages))


def to_marker_page_range(pages: list[int]) -> str:
    """Render 1-indexed pages as Datalab's 0-indexed compact range string.

    [1,2,3,5,8,9] → "0-2,4,7-8".
    """
    zero = sorted({p - 1 for p in pages if p >= 1})
    if not zero:
        return ""
    parts: list[str] = []
    start = prev = zero[0]
    for p in zero[1:]:
        if p == prev + 1:
            prev = p
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = p
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def scope_summary(scope_json: dict, total_pages: int | None = None) -> dict:
    """Human-facing summary for the UI: selected/skipped pages + region count."""
    sel = selected_pages(scope_json, total_pages)
    skipped: list[int] = []
    if total_pages:
        sel_set = set(sel)
        skipped = [p for p in range(1, total_pages + 1) if p not in sel_set]
    return {
        "kind": (scope_json or {}).get("kind", "full"),
        "selected_pages": sel,
        "selected_count": len(sel),
        "skipped_pages": skipped,
        "skipped_count": len(skipped),
        "total_pages": total_pages,
        "region_count": len(scope_json.get("regions") or []) if scope_json else 0,
    }

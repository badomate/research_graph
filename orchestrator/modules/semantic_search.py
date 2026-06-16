"""
modules/semantic_search.py — semantic retrieval over parsed paper chunks.

The vector layer is a *derived* index, never the source of truth: every hit
resolves back to a SQLite `paper_chunks` row (and its paper). Two backends:

  * OpenAI embeddings (cosine) when ``OPENAI_API_KEY`` is set — embeddings are
    cached in-process by chunk ``content_hash`` so repeated searches are cheap.
  * a pure-Python TF-IDF cosine fallback otherwise — no deps, fully offline,
    unit-testable.

This keeps the web app dependency-light while giving real semantic ranking when
configured. (The richer Qdrant concept index in ``vector_index.py`` is separate;
this is chunk-level retrieval for the unified search box.)
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# In-process embedding cache: content_hash → vector.
_EMBED_CACHE: dict[str, list[float]] = {}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ── Pure TF-IDF cosine (offline-testable) ────────────────────────────────────────


def tfidf_rank(query: str, docs: list[str]) -> list[tuple[int, float]]:
    """Rank ``docs`` against ``query`` by TF-IDF cosine. Returns (index, score)
    sorted desc, dropping zero-score docs."""
    q_tokens = tokenize(query)
    if not q_tokens or not docs:
        return []
    doc_tokens = [tokenize(d) for d in docs]
    n = len(docs)
    df: Counter = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1
    idf = {t: math.log((n + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    def vec(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {t: (c / len(tokens)) * idf.get(t, math.log(n + 1) + 1.0) for t, c in tf.items()}

    qv = vec(q_tokens)
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    scored: list[tuple[int, float]] = []
    for i, toks in enumerate(doc_tokens):
        if not toks:
            continue
        dv = vec(toks)
        dot = sum(qv.get(t, 0.0) * v for t, v in dv.items())
        dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        score = dot / (qn * dn)
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── OpenAI embeddings (optional) ─────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embed_openai(texts: list[str], model: str) -> list[list[float]] | None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=model, input=[t[:8000] for t in texts])
        return [d.embedding for d in resp.data]
    except Exception:
        logger.warning("semantic_search: OpenAI embedding failed; falling back to TF-IDF", exc_info=True)
        return None


def embed_rank(query: str, docs: list[str], hashes: list[str], model: str) -> list[tuple[int, float]] | None:
    """Embedding cosine ranking, using the per-hash cache. None → backend off/failed."""
    to_embed = [(i, d) for i, (d, h) in enumerate(zip(docs, hashes)) if h not in _EMBED_CACHE]
    if to_embed:
        vecs = _embed_openai([d for _, d in to_embed], model)
        if vecs is None:
            return None
        for (i, _), v in zip(to_embed, vecs):
            _EMBED_CACHE[hashes[i]] = v
    qvec = _embed_openai([query], model)
    if not qvec:
        return None
    q = qvec[0]
    scored = [(i, _cosine(q, _EMBED_CACHE[h])) for i, h in enumerate(hashes) if h in _EMBED_CACHE]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Public entry point ───────────────────────────────────────────────────────────


def semantic_search(store, query: str, k: int = 20, model: str = "text-embedding-3-small") -> list[dict]:
    """Rank parsed chunks against ``query`` and resolve to paper/chunk rows.

    Returns up to ``k`` dicts: {paper, chunk, score, snippet, backend}. One result
    per paper (best chunk), so the search list stays paper-centric.
    """
    query = (query or "").strip()
    if not query:
        return []
    # Gather all parsed chunks across papers (single-user scale).
    chunks = _all_chunks(store)
    if not chunks:
        return []
    docs = [(c.heading + " " + c.text) if c.heading else c.text for c in chunks]
    hashes = [c.content_hash or str(i) for i, c in enumerate(chunks)]

    ranked = embed_rank(query, docs, hashes, model)
    backend = "embedding"
    if ranked is None:
        ranked = tfidf_rank(query, docs)
        backend = "tfidf"

    best_per_paper: dict[str, dict] = {}
    for idx, score in ranked:
        ch = chunks[idx]
        if ch.paper_id in best_per_paper:
            continue
        paper = store.get_paper(ch.paper_id)
        if paper is None:
            continue
        text = ch.text or ""
        best_per_paper[ch.paper_id] = {
            "paper": paper, "chunk": ch, "score": round(float(score), 4),
            "snippet": (text[:200] + ("…" if len(text) > 200 else "")),
            "backend": backend,
        }
        if len(best_per_paper) >= k:
            break
    return list(best_per_paper.values())


def _all_chunks(store) -> list:
    """All parsed chunks across papers (via the Store's paper list)."""
    chunks = []
    for paper in store.list_papers():
        chunks.extend(store.chunks_for_paper(paper.id))
    return chunks

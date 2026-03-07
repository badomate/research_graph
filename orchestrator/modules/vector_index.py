"""
modules/vector_index.py — Module 7: Semantic Retrieval via Qdrant
─────────────────────────────────────────────────────────────────
Maintains three Qdrant collections with role-specific embeddings per concept
and replaces the TF-IDF Stage 2 retrieval with proper semantic search.

Three-collection design
───────────────────────
  concept_full
      Embeds: title + conclusion + interpretation + canonical_keywords
      Matches: related | generalizes | special_case_of
      Rationale: full semantic overlap → broad relatedness signal

  concept_assumptions
      Embeds: assumptions + prereq_keywords + named_tools
      Matches: depends_on
      Rationale: query's assumptions close to concept B → B is a prerequisite

  concept_conclusions
      Embeds: conclusion + statement_latex_stripped + downstream_keywords
      Matches: enables
      Rationale: query's conclusions close to B's assumptions → query unlocks B

Point ID scheme
───────────────
  uuid5(NAMESPACE_URL, notion_page_id + ":" + role)
  → deterministic, idempotent upserts.

Embedding backends
──────────────────
  openai  — text-embedding-3-small (1536-d, default)
  local   — allenai/specter2 (768-d), requires sentence-transformers

All Qdrant operations use the synchronous qdrant_client API.
Never imported as a hard dependency — if Qdrant is unreachable, every
method is a silent no-op and the pipeline falls back to TF-IDF.

References
──────────
  Qdrant concepts:    https://qdrant.tech/documentation/concepts/
  SPECTER2:           https://arxiv.org/abs/2305.14722
  DPR retrieval:      https://arxiv.org/abs/2004.04906
  Math text prep:     https://arxiv.org/abs/2208.05051
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ── Environment-driven defaults ───────────────────────────────────────────────

_QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://qdrant:6333")
_EMBEDDING_BACKEND: str = os.environ.get("VECTOR_EMBEDDING_BACKEND", "openai").lower()
_EMBEDDING_MODEL: str = os.environ.get(
    "VECTOR_EMBEDDING_MODEL", "text-embedding-3-small"
)
_EMBEDDING_DIM: int = int(os.environ.get("VECTOR_EMBEDDING_DIM", "1536"))
_SB_CONCEPT_LEVEL: str = os.environ.get("SB_CONCEPT_LEVEL", "Concept")

# Retrieval cap applied after deduplication.
RETRIEVE_CANDIDATES_K: int = int(os.environ.get("RETRIEVE_CANDIDATES_K", "30"))

# Batch size for Qdrant upserts during rebuild.
_UPSERT_BATCH: int = 64

# Max characters for text fed to the embedding model (~512 tokens @ 4 chars/tok).
_EMBED_TEXT_MAX_CHARS: int = 2048

# Roles — used as point-ID suffixes and collection mapping.
_ROLES = ("full", "assumptions", "conclusions")


# ── LaTeX macro expansion table ───────────────────────────────────────────────

_MACRO_TABLE: list[tuple[str, str]] = [
    # Greek letters
    (r"\\varepsilon", "epsilon"),
    (r"\\epsilon", "epsilon"),
    (r"\\delta", "delta"),
    (r"\\alpha", "alpha"),
    (r"\\beta", "beta"),
    (r"\\gamma", "gamma"),
    (r"\\Gamma", "Gamma"),
    (r"\\lambda", "lambda"),
    (r"\\Lambda", "Lambda"),
    (r"\\mu", "mu"),
    (r"\\nu", "nu"),
    (r"\\sigma", "sigma"),
    (r"\\Sigma", "Sigma"),
    (r"\\pi\b", "pi"),
    (r"\\Pi\b", "Pi"),
    (r"\\omega", "omega"),
    (r"\\Omega", "Omega"),
    (r"\\phi", "phi"),
    (r"\\Phi", "Phi"),
    (r"\\psi", "psi"),
    (r"\\Psi", "Psi"),
    (r"\\rho", "rho"),
    (r"\\tau", "tau"),
    (r"\\theta", "theta"),
    (r"\\Theta", "Theta"),
    (r"\\xi\b", "xi"),
    (r"\\zeta", "zeta"),
    (r"\\eta\b", "eta"),
    (r"\\kappa", "kappa"),
    # Arrows and relations
    (r"\\to\b", "to"),
    (r"\\rightarrow", "to"),
    (r"\\Rightarrow", "implies"),
    (r"\\leftarrow", "from"),
    (r"\\leftrightarrow", "if and only if"),
    (r"\\iff\b", "if and only if"),
    (r"\\in\b", "in"),
    (r"\\notin\b", "not in"),
    (r"\\subseteq", "subset of"),
    (r"\\subset", "subset of"),
    (r"\\supseteq", "superset of"),
    (r"\\leq", "less than or equal"),
    (r"\\geq", "greater than or equal"),
    (r"\\le\b", "less than or equal"),
    (r"\\ge\b", "greater than or equal"),
    (r"\\neq", "not equal to"),
    (r"\\equiv", "equivalent to"),
    (r"\\approx", "approximately"),
    (r"\\sim\b", "similar to"),
    (r"\\propto", "proportional to"),
    # Set / logic
    (r"\\forall", "for all"),
    (r"\\exists", "there exists"),
    (r"\\cup\b", "union"),
    (r"\\cap\b", "intersection"),
    (r"\\setminus", "minus"),
    (r"\\emptyset", "empty set"),
    (r"\\varnothing", "empty set"),
    (r"\\infty", "infinity"),
    # Calculus / operators
    (r"\\nabla", "nabla"),
    (r"\\partial", "partial"),
    (r"\\cdot\b", "times"),
    (r"\\times\b", "times"),
    (r"\\otimes", "tensor product"),
    (r"\\oplus", "direct sum"),
    (r"\\circ\b", "composed with"),
    # Misc
    (r"\\ldots", "..."),
    (r"\\cdots", "..."),
    (r"\\vdots", "..."),
    (r"\\pm\b", "plus or minus"),
    (r"\\mp\b", "minus or plus"),
]

# Compiled once at module load.
_MACRO_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat), repl) for pat, repl in _MACRO_TABLE
]

# \mathbb letter → English
_MATHBB_MAP: dict[str, str] = {
    "R": "real numbers",
    "N": "natural numbers",
    "Z": "integers",
    "Q": "rational numbers",
    "C": "complex numbers",
    "E": "expected value",
    "P": "probability",
    "H": "Hilbert space",
    "L": "Lebesgue space",
}


# ── Candidate output type ─────────────────────────────────────────────────────


@dataclass
class CandidateWithHint:
    """
    A single candidate concept returned by Stage 2 vector retrieval.

    ``edge_type_hint`` is populated from which collection produced the hit:
      "related"    — concept_full (query with full embedding)
      "depends_on" — concept_full (query with assumption embedding)
      "enables"    — concept_assumptions (query with conclusion embedding)

    If the same concept appears in multiple collections, hints are
    comma-joined and the highest score is kept.
    """

    notion_page_id: str
    title: str
    score: float
    edge_type_hint: str  # "related" | "depends_on" | "enables" | comma-joined
    hub: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict compatible with Stage 3 (TF-IDF-style) callers."""
        return {
            "id": self.notion_page_id,
            "title": self.title,
            "score": self.score,
            "hub": self.hub,
            "edge_type_hint": self.edge_type_hint,
            "verified": self.verified,
            # summary intentionally absent — not available from Qdrant payload
        }


# ── Main engine ───────────────────────────────────────────────────────────────


class VectorIndexEngine:
    """
    Module 7: Semantic concept retrieval via Qdrant vector database.

    Maintains three role-specific collections:
      - concept_full:         full semantic similarity
      - concept_assumptions:  prerequisite signal
      - concept_conclusions:  enables signal

    Thread-safe. All Qdrant operations use the synchronous qdrant_client API.
    Embedding calls are batched where possible to minimise API round-trips.

    Graceful degradation
    ────────────────────
    If Qdrant is unreachable at startup ``_available`` is set to False and
    every public method becomes a no-op.  The pipeline then falls back to
    TF-IDF retrieval silently.
    """

    COLLECTIONS: list[str] = [
        "concept_full",
        "concept_assumptions",
        "concept_conclusions",
    ]

    def __init__(self) -> None:
        self._available: bool = False
        self._vector_dim: int = _EMBEDDING_DIM
        self._backend: str = _EMBEDDING_BACKEND
        self._openai_client: Any = None
        self._local_model: Any = None

        # ── Connect to Qdrant ─────────────────────────────────────────────────
        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=_QDRANT_URL, timeout=10)
            # Connectivity smoke-test.
            self._client.get_collections()
            logger.info("VectorIndexEngine: connected to Qdrant at %s.", _QDRANT_URL)
        except Exception as exc:
            logger.warning(
                "VectorIndexEngine: Qdrant unavailable (%s). "
                "Falling back to TF-IDF retrieval.",
                exc,
            )
            return

        # ── Initialise embedding backend ──────────────────────────────────────
        try:
            if self._backend == "local":
                self._init_local_model()
            else:
                self._init_openai_client()
        except Exception as exc:
            logger.warning(
                "VectorIndexEngine: embedding backend init failed (%s). "
                "Falling back to TF-IDF retrieval.",
                exc,
            )
            return

        # ── Ensure collections exist ──────────────────────────────────────────
        try:
            self._ensure_collections()
        except Exception as exc:
            logger.warning(
                "VectorIndexEngine: collection setup failed (%s). "
                "Falling back to TF-IDF retrieval.",
                exc,
            )
            return

        self._available = True
        logger.info(
            "VectorIndexEngine: ready (backend=%s, dim=%d).",
            self._backend,
            self._vector_dim,
        )

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_openai_client(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — cannot use OpenAI embeddings.")
        import openai

        self._openai_client = openai.OpenAI(api_key=api_key)
        logger.debug("VectorIndexEngine: OpenAI embedding client ready.")

    def _init_local_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embedding backend. "
                "Install with: pip install sentence-transformers>=2.7.0"
            ) from exc

        model_name = _EMBEDDING_MODEL if _EMBEDDING_MODEL else "sentence-transformers/allenai-specter"
        logger.info("VectorIndexEngine: loading local model '%s' ...", model_name)
        self._local_model = SentenceTransformer(model_name)
        logger.info("VectorIndexEngine: local model loaded.")

    # ── Public property ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if Qdrant is reachable and all collections exist."""
        return self._available

    # ── Public API ────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Return True if Qdrant is reachable and all collections exist.

        Safe to call repeatedly; does not raise.
        """
        if not self._available:
            return False
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            return all(c in existing for c in self.COLLECTIONS)
        except Exception as exc:
            logger.warning("VectorIndexEngine.health_check: %s", exc)
            return False

    def index_concept(
        self,
        concept: Any,  # MathObject — avoid circular import
        notion_page_id: str,
        verified: bool,
    ) -> None:
        """
        Index a single concept across all three collections. Idempotent.

        Uses upsert with deterministic point IDs so re-indexing the same
        concept overwrites the previous entry without creating duplicates.

        Partial failure policy: if one collection fails, logs an error and
        continues with the remaining two.
        """
        if not self._available:
            return

        payload = self._build_payload(concept, notion_page_id, verified)

        role_collection_text: list[tuple[str, str, str]] = [
            ("full", "concept_full", self._build_full_text(concept)),
            ("assumptions", "concept_assumptions", self._build_assumption_text(concept)),
            ("conclusions", "concept_conclusions", self._build_conclusion_text(concept)),
        ]

        for role, collection, text in role_collection_text:
            try:
                embedding = self._embed(text)
                point_id = self._point_id(notion_page_id, role)
                self._safe_upsert(
                    collection,
                    [self._make_point(point_id, embedding, payload)],
                )
                logger.debug(
                    "VectorIndexEngine.index_concept: upserted '%s' → %s in '%s'.",
                    concept.title,
                    point_id,
                    collection,
                )
            except Exception:
                logger.error(
                    "VectorIndexEngine.index_concept: failed for '%s' in collection '%s'.",
                    concept.title,
                    collection,
                    exc_info=True,
                )

    def promote_concept(self, ki_page_id: str, sb_page_id: str) -> None:
        """
        Migrate all three collection entries from KI page ID to SB page ID.

        Steps per collection:
          1. Retrieve old point (with vector) by deterministic point ID.
          2. Upsert new point at new deterministic ID with updated payload
             (notion_page_id → sb_page_id, verified → True).
          3. Delete old point.

        Partial failure policy: log error per collection, continue with rest.
        """
        if not self._available:
            return

        from qdrant_client.models import PointIdsList

        now_iso = datetime.now(timezone.utc).isoformat()

        for role, collection in zip(_ROLES, self.COLLECTIONS):
            old_id = self._point_id(ki_page_id, role)
            new_id = self._point_id(sb_page_id, role)
            try:
                # Retrieve old point.
                points = self._client.retrieve(
                    collection_name=collection,
                    ids=[old_id],
                    with_vectors=True,
                    with_payload=True,
                )
                if not points:
                    logger.warning(
                        "VectorIndexEngine.promote_concept: point %s not found in '%s'. "
                        "Concept may not have been indexed at KI stage.",
                        old_id,
                        collection,
                    )
                    continue

                old_point = points[0]
                new_payload = dict(old_point.payload or {})
                new_payload["notion_page_id"] = sb_page_id
                new_payload["verified"] = True
                new_payload["indexed_at"] = now_iso

                # Upsert under new ID.
                self._safe_upsert(
                    collection,
                    [self._make_point(new_id, old_point.vector, new_payload)],
                )
                # Delete old point.
                self._client.delete(
                    collection_name=collection,
                    points_selector=PointIdsList(points=[old_id]),
                )
                logger.debug(
                    "VectorIndexEngine.promote_concept: migrated %s → %s in '%s'.",
                    old_id,
                    new_id,
                    collection,
                )
            except Exception:
                logger.error(
                    "VectorIndexEngine.promote_concept: failed for collection '%s' "
                    "(ki=%s, sb=%s).",
                    collection,
                    ki_page_id,
                    sb_page_id,
                    exc_info=True,
                )

    def retrieve_candidates(
        self,
        concept: Any,  # MathObject
        k_per_collection: int = 15,
        verified_only: bool = False,
    ) -> list[CandidateWithHint]:
        """
        Run three sequential ANN queries and merge results.

        Query plan
        ──────────
        1. concept_full        ← full embedding       → hint: "related"
        2. concept_full        ← assumption embedding → hint: "depends_on"
        3. concept_assumptions ← conclusion embedding → hint: "enables"

        Returns a deduplicated list sorted by score descending.
        If the same notion_page_id appears in multiple hits, keeps the highest
        score and joins the hints with a comma.
        """
        if not self._available:
            return []

        # ── Build the three query embeddings ─────────────────────────────────
        full_text       = self._build_full_text(concept)
        assumption_text = self._build_assumption_text(concept)
        conclusion_text = self._build_conclusion_text(concept)

        try:
            full_emb, assumption_emb, conclusion_emb = self._embed_batch(
                [full_text, assumption_text, conclusion_text]
            )
        except Exception:
            logger.error(
                "VectorIndexEngine.retrieve_candidates: embedding failed for '%s'.",
                concept.title,
                exc_info=True,
            )
            return []

        qdrant_filter = self._build_filter(verified_only)

        # ── Three sequential queries ──────────────────────────────────────────
        # (collection_name, embedding, edge_type_hint)
        queries: list[tuple[str, list[float], str]] = [
            ("concept_full",        full_emb,        "related"),
            ("concept_full",        assumption_emb,  "depends_on"),
            ("concept_assumptions", conclusion_emb,  "enables"),
        ]

        all_hits: list[tuple[str, Any]] = []  # (hint, ScoredPoint)

        for collection, emb, hint in queries:
            try:
                response = self._client.query_points(
                    collection_name=collection,
                    query=emb,
                    limit=k_per_collection,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                for point in response.points:
                    all_hits.append((hint, point))
            except Exception:
                logger.error(
                    "VectorIndexEngine.retrieve_candidates: query failed "
                    "for collection '%s' hint '%s'.",
                    collection,
                    hint,
                    exc_info=True,
                )
                # Continue with remaining queries — partial results are
                # better than no results.

        return self._merge_hits(all_hits, RETRIEVE_CANDIDATES_K)

    def rebuild_index_from_second_brain(
        self,
        notion: Any,  # NotionClientWrapper — avoid circular import
        sb_db_id: str,
    ) -> None:
        """
        Drop and recreate all three collections, then re-index every concept
        in the Second Brain.

        Idempotent: running twice produces the same result.
        Use for: initial setup, embedding model changes, schema migrations.
        """
        if not self._available:
            logger.warning(
                "VectorIndexEngine.rebuild_index_from_second_brain: "
                "Qdrant not available — aborting rebuild."
            )
            return

        logger.info("VectorIndexEngine: starting full index rebuild ...")
        self._drop_collections()
        self._ensure_collections()

        # Fetch all Second Brain Concept pages.
        pages = notion.query_database(
            sb_db_id,
            filter={
                "property": "Note Level",
                "select": {"equals": _SB_CONCEPT_LEVEL},
            },
        )
        logger.info(
            "VectorIndexEngine rebuild: found %d Second Brain concept(s).", len(pages)
        )
        if not pages:
            return

        # Build (pseudo_MathObject, sb_page_id) pairs from SB page data.
        triples: list[tuple[Any, str]] = []
        for page in pages:
            mo = self._math_object_from_sb_page(page)
            if mo is not None:
                triples.append((mo, page["id"]))

        # Collect all texts for batch embedding.
        texts_full = [self._build_full_text(mo) for mo, _ in triples]
        texts_assm = [self._build_assumption_text(mo) for mo, _ in triples]
        texts_conc = [self._build_conclusion_text(mo) for mo, _ in triples]
        all_texts = texts_full + texts_assm + texts_conc

        logger.info(
            "VectorIndexEngine rebuild: embedding %d texts ...", len(all_texts)
        )
        try:
            all_embeddings = self._embed_batch(all_texts)
        except Exception:
            logger.error(
                "VectorIndexEngine rebuild: batch embedding failed.", exc_info=True
            )
            return

        n = len(triples)
        embs_full = all_embeddings[:n]
        embs_assm = all_embeddings[n : 2 * n]
        embs_conc = all_embeddings[2 * n :]

        # Build PointStruct lists.
        from qdrant_client.models import PointStruct

        now_iso = datetime.now(timezone.utc).isoformat()
        points_full: list[PointStruct] = []
        points_assm: list[PointStruct] = []
        points_conc: list[PointStruct] = []

        for j, (mo, page_id) in enumerate(triples):
            payload = self._build_payload(mo, page_id, verified=True)
            payload["indexed_at"] = now_iso
            points_full.append(
                PointStruct(
                    id=self._point_id(page_id, "full"),
                    vector=embs_full[j],
                    payload=payload,
                )
            )
            points_assm.append(
                PointStruct(
                    id=self._point_id(page_id, "assumptions"),
                    vector=embs_assm[j],
                    payload=payload,
                )
            )
            points_conc.append(
                PointStruct(
                    id=self._point_id(page_id, "conclusions"),
                    vector=embs_conc[j],
                    payload=payload,
                )
            )

        # Upsert in batches of _UPSERT_BATCH points per collection.
        for col_name, points_list in [
            ("concept_full", points_full),
            ("concept_assumptions", points_assm),
            ("concept_conclusions", points_conc),
        ]:
            total = len(points_list)
            for i in range(0, total, _UPSERT_BATCH):
                batch = points_list[i : i + _UPSERT_BATCH]
                try:
                    self._client.upsert(collection_name=col_name, points=batch)
                    logger.info(
                        "VectorIndexEngine rebuild: upserted %d/%d to '%s'.",
                        min(i + _UPSERT_BATCH, total),
                        total,
                        col_name,
                    )
                except Exception:
                    logger.error(
                        "VectorIndexEngine rebuild: upsert failed for '%s' batch %d.",
                        col_name,
                        i // _UPSERT_BATCH,
                        exc_info=True,
                    )

        logger.info(
            "VectorIndexEngine rebuild: complete — %d concept(s) indexed.", n
        )

    # ── Private: collection management ───────────────────────────────────────

    def _ensure_collections(self) -> None:
        """Create collections if they don't exist. Called in __init__."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        for name in self.COLLECTIONS:
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "VectorIndexEngine: created collection '%s' (dim=%d).",
                    name,
                    self._vector_dim,
                )
            else:
                logger.debug(
                    "VectorIndexEngine: collection '%s' already exists.", name
                )

    def _drop_collections(self) -> None:
        """Delete all three collections. ONLY called from rebuild_index_from_second_brain."""
        for name in self.COLLECTIONS:
            try:
                self._client.delete_collection(collection_name=name)
                logger.info("VectorIndexEngine: dropped collection '%s'.", name)
            except Exception:
                logger.debug(
                    "VectorIndexEngine: could not drop '%s' (may not exist).", name
                )

    def _safe_upsert(self, collection_name: str, points: list[Any]) -> None:
        """
        Upsert points, recreating the collection if it was deleted.

        Retries once after recreating missing collections.
        """
        try:
            self._client.upsert(collection_name=collection_name, points=points)
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg or "does not exist" in msg:
                logger.warning(
                    "VectorIndexEngine: collection '%s' missing, recreating.",
                    collection_name,
                )
                self._ensure_collections()
                self._client.upsert(collection_name=collection_name, points=points)
            else:
                raise

    # ── Private: embedding ────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed a single string. Dispatches to the configured backend."""
        return self._embed_batch([text])[0]

    def _embed_batch(
        self, texts: list[str], _sub_batch: int = 256
    ) -> list[list[float]]:
        """
        Batch embed a list of strings.

        Splits into sub-batches of ``_sub_batch`` to stay within API limits.
        More efficient than calling ``_embed`` repeatedly for large inputs.
        """
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), _sub_batch):
            chunk = texts[i : i + _sub_batch]
            if self._backend == "local":
                batch_emb = self._embed_local_batch(chunk)
            else:
                batch_emb = self._embed_openai_batch(chunk)
            results.extend(batch_emb)
        return results

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI Embeddings API with retry/back-off."""
        response = self._openai_client.embeddings.create(
            input=texts,
            model=_EMBEDDING_MODEL,
        )
        # API preserves order but we sort defensively.
        ordered = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]

    def _embed_local_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed using the local SentenceTransformer model."""
        embeddings = self._local_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    # ── Private: text construction ────────────────────────────────────────────

    def _build_full_text(self, concept: Any) -> str:
        """
        Build embedding text for ``concept_full``.

        Encodes: title + conclusion + interpretation + canonical_keywords.
        """
        parts = [
            concept.title or "",
            concept.conclusion or "",
            concept.interpretation or "",
            " ".join(getattr(concept, "canonical_keywords", None) or []),
        ]
        raw = " ".join(p for p in parts if p)
        return self._strip_latex(raw)

    def _build_assumption_text(self, concept: Any) -> str:
        """
        Build embedding text for ``concept_assumptions``.

        Encodes: assumptions + prereq_keywords + named_tools.
        """
        parts = [
            concept.assumptions or "",
            " ".join(getattr(concept, "prereq_keywords", None) or []),
            " ".join(getattr(concept, "named_tools", None) or []),
        ]
        raw = " ".join(p for p in parts if p)
        return self._strip_latex(raw)

    def _build_conclusion_text(self, concept: Any) -> str:
        """
        Build embedding text for ``concept_conclusions``.

        Encodes: conclusion + statement_latex_stripped + downstream_keywords.
        """
        parts = [
            concept.conclusion or "",
            getattr(concept, "statement_latex", None) or "",
            " ".join(getattr(concept, "downstream_keywords", None) or []),
        ]
        raw = " ".join(p for p in parts if p)
        return self._strip_latex(raw)

    # ── Private: payload helpers ──────────────────────────────────────────────

    def _build_payload(
        self, concept: Any, notion_page_id: str, verified: bool
    ) -> dict[str, Any]:
        """Build the Qdrant point payload dict from a MathObject."""
        title_clean = concept.title or ""
        title_raw = f"[{concept.type}] {title_clean}"
        return {
            "notion_page_id": notion_page_id,
            "title": title_raw,
            "title_clean": title_clean,
            "verified": verified,
            "suggested_hub": getattr(concept, "suggested_hub", None) or "",
            "concept_type": concept.type or "",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _make_point(point_id: str, vector: list[float], payload: dict) -> Any:
        """Construct a qdrant_client PointStruct."""
        from qdrant_client.models import PointStruct

        return PointStruct(id=point_id, vector=vector, payload=payload)

    # ── Private: retrieval helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_filter(verified_only: bool) -> Any | None:
        """Build a Qdrant filter dict for verified_only queries."""
        if not verified_only:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[FieldCondition(key="verified", match=MatchValue(value=True))]
        )

    @staticmethod
    def _merge_hits(
        all_hits: list[tuple[str, Any]],
        top_k: int,
    ) -> list[CandidateWithHint]:
        """
        Deduplicate hits by notion_page_id.

        For duplicates: keep highest score, join hints with comma.
        """
        best: dict[str, CandidateWithHint] = {}

        for hint, hit in all_hits:
            payload = hit.payload or {}
            page_id: str = payload.get("notion_page_id", "")
            if not page_id:
                continue
            score: float = float(hit.score)
            title: str = payload.get("title_clean", "") or payload.get("title", "")
            hub: str = payload.get("suggested_hub", "")
            verified: bool = bool(payload.get("verified", False))

            if page_id not in best:
                best[page_id] = CandidateWithHint(
                    notion_page_id=page_id,
                    title=title,
                    score=score,
                    edge_type_hint=hint,
                    hub=hub,
                    verified=verified,
                )
            else:
                existing = best[page_id]
                # Merge hints (deduplicated, comma-joined).
                existing_hints = set(existing.edge_type_hint.split(","))
                existing_hints.add(hint)
                existing.edge_type_hint = ",".join(sorted(existing_hints))
                # Keep highest score.
                if score > existing.score:
                    existing.score = score

        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    # ── Private: text preprocessing ──────────────────────────────────────────

    @staticmethod
    def _strip_latex(text: str) -> str:
        """
        Preprocess a string by expanding LaTeX macros to natural language
        and stripping remaining LaTeX commands.

        Processing pipeline
        ────────────────────
        1. Strip display-math delimiters \\[...\\] and $$...$$
           (content removed — display math is the full theorem statement,
           embedded separately via statement_latex)
        2. Strip inline-math delimiters $...$ — keep content
        3. Strip \\(...\\) — keep content
        4. Expand \\mathbb{X} → English ("real numbers", etc.)
        5. Expand \\mathcal{X} → bare letter X
        6. Expand named macros (Greek letters, arrows, relations)
        7. Expand \\frac{a}{b} → "a over b"
        8. Expand \\sqrt{x} → "square root of x"
        9. Expand \\|...\\| → "norm of ..."
        10. Strip remaining \\command{content} → content only (3 passes)
        11. Strip standalone \\command
        12. Strip remaining LaTeX delimiters { } leaving subscript _ intact
        13. Collapse superscripts: L^2 → L2
        14. Collapse whitespace; truncate to _EMBED_TEXT_MAX_CHARS
        """
        if not text:
            return text

        # 1. Strip display math (remove content too — pure LaTeX equations
        #    add noise rather than semantic signal for embedding).
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
        text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)

        # 2. Strip inline math delimiters, keep content.
        text = re.sub(r"\$(.+?)\$", r" \1 ", text, flags=re.DOTALL)

        # 3. Strip \(...\) delimiters, keep content.
        text = re.sub(r"\\\((.+?)\\\)", r" \1 ", text, flags=re.DOTALL)

        # 4. Expand \mathbb{X} → English.
        def _expand_mathbb(m: re.Match) -> str:
            letter = m.group(1).strip()
            return _MATHBB_MAP.get(letter, letter)

        text = re.sub(r"\\mathbb\{([^}]+)\}", _expand_mathbb, text)

        # 5. Expand \mathcal{X} → just the letter(s).
        text = re.sub(r"\\mathcal\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\mathit\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\operatorname\{([^}]+)\}", r"\1", text)

        # 6. Expand named macros (Greek, arrows, relations, …).
        for pattern, replacement in _MACRO_RE:
            text = pattern.sub(replacement, text)

        # 7. Expand \frac{a}{b} → "a over b" (non-nested first, then nested).
        for _ in range(3):
            text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1 over \2", text)

        # 8. Expand \sqrt{x} → "square root of x".
        text = re.sub(r"\\sqrt\{([^}]*)\}", r"square root of \1", text)

        # 9. Expand \|...\| → "norm of ...".
        text = re.sub(r"\\\|(.+?)\\\|", r"norm of \1", text, flags=re.DOTALL)

        # 10. Strip remaining \command{content} → content (3 passes for nesting).
        for _ in range(3):
            text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", text)

        # 11. Strip standalone \command.
        text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

        # 12. Strip { and }.
        text = re.sub(r"[{}]", " ", text)

        # 13. Collapse superscripts: L^2 → L2.
        text = re.sub(r"\^", "", text)

        # 14. Collapse whitespace and truncate.
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > _EMBED_TEXT_MAX_CHARS:
            text = text[:_EMBED_TEXT_MAX_CHARS].rsplit(" ", 1)[0]

        return text

    # ── Private: point ID ─────────────────────────────────────────────────────

    @staticmethod
    def _point_id(notion_page_id: str, role: str) -> str:
        """
        Deterministic UUID5 from notion_page_id + role.

        Using NAMESPACE_URL ensures the ID space is well-defined and
        collision-resistant across different notion_page_id values.

        The same inputs always produce the same ID, enabling idempotent upserts.
        """
        seed = f"{notion_page_id}:{role}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    # ── Private: SB page → pseudo-MathObject for rebuild ─────────────────────

    @staticmethod
    def _math_object_from_sb_page(page: dict) -> Any | None:
        """
        Extract available fields from an SB Concept page and return a
        lightweight object suitable for embedding via the _build_*_text methods.

        Returns None if the page has no usable title.
        """
        props = page.get("properties", {})

        def _title() -> str:
            for v in props.values():
                if v.get("type") == "title":
                    try:
                        return v["title"][0]["plain_text"]
                    except (KeyError, IndexError):
                        return ""
            return ""

        def _text(key: str) -> str:
            try:
                segs = props[key]["rich_text"]
                return "".join(s.get("plain_text", "") for s in segs)
            except (KeyError, TypeError):
                return ""

        def _select(key: str) -> str:
            try:
                return props[key]["select"]["name"] or ""
            except (KeyError, TypeError):
                return ""

        def _multi(key: str) -> list[str]:
            try:
                return [o["name"] for o in props[key]["multi_select"]]
            except (KeyError, TypeError):
                return []

        title = _title()
        if not title:
            return None

        # Strip [Type] prefix if present.
        title_clean = re.sub(r"^\[[^\]]+\]\s*", "", title)

        # Build a simple namespace object — no Pydantic overhead needed here.
        class _SBConcept:
            pass

        obj = _SBConcept()
        obj.title = title_clean
        obj.type = _select("Type") or "Definition"
        obj.statement_latex = _text("Statement LaTeX")
        obj.assumptions = _text("Assumptions")
        obj.conclusion = _text("One Liner") or _text("Interpretation")
        obj.interpretation = _text("Interpretation")
        obj.proof_idea = _text("Proof Idea")
        obj.canonical_keywords = _multi("Keywords")
        obj.prereq_keywords = _multi("Prereq Keywords")
        obj.downstream_keywords = _multi("Downstream Keywords")
        obj.named_tools = _multi("Named Tools")
        obj.suggested_hub = _text("Suggested Hub")
        obj.setting = _multi("Setting")
        return obj


# ── CLI entry point ───────────────────────────────────────────────────────────


def _cli_rebuild() -> None:
    """Rebuild the vector index from the Second Brain (CLI entry point)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="VectorIndexEngine CLI — paper_pipeline Module 7"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate all Qdrant collections from Second Brain.",
    )
    args = parser.parse_args()

    if not args.rebuild:
        parser.print_help()
        sys.exit(0)

    # Load .env if present.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    sb_db_id = os.environ.get("NOTION_SECOND_BRAIN_DB_ID")
    if not sb_db_id:
        print("ERROR: NOTION_SECOND_BRAIN_DB_ID not set.", file=sys.stderr)
        sys.exit(1)

    # Import here to get fresh env vars after load_dotenv.
    try:
        from modules.notion_client_wrapper import NotionClientWrapper
    except ModuleNotFoundError:
        # Running from orchestrator/ directory.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from notion_client_wrapper import NotionClientWrapper  # type: ignore[no-redef]

    notion = NotionClientWrapper()
    engine = VectorIndexEngine()

    if not engine.available:
        print("ERROR: VectorIndexEngine is not available (Qdrant unreachable?).", file=sys.stderr)
        sys.exit(1)

    engine.rebuild_index_from_second_brain(notion, sb_db_id)
    print("Rebuild complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _cli_rebuild()

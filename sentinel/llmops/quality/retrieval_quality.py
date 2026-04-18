"""Retrieval quality monitoring for RAG pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from sentinel.config.schema import LLMOpsConfig, RetrievalQualityConfig

log = structlog.get_logger(__name__)

# Blending weights when semantic embeddings are available.
_TOKEN_WEIGHT = 0.4
_SEMANTIC_WEIGHT = 0.6


@dataclass
class RetrievalQualityResult:
    """Per-call retrieval quality metrics."""

    relevance: float = 0.0
    chunk_utilisation: float = 0.0
    answer_coverage: float = 0.0
    faithfulness: float = 0.0
    chunks_retrieved: int = 0
    chunks_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        return self.relevance >= 0.5 and self.faithfulness >= 0.7


class RetrievalQualityMonitor:
    """Track RAG-specific quality metrics for every retrieval + answer pair.

    When ``sentence-transformers`` is available the monitor blends
    token-overlap scores (weight 0.4) with cosine-similarity scores
    (weight 0.6) for the *relevance* and *faithfulness* metrics.  If the
    library cannot be imported the monitor falls back to pure
    token-overlap scoring.
    """

    def __init__(self, config: RetrievalQualityConfig | None = None):
        self.config = config or RetrievalQualityConfig()
        self.tracked = set(self.config.track or ["relevance", "chunk_utilisation", "faithfulness"])
        self._embedder: Any | None = None
        self._embedder_tried: bool = False

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Any) -> RetrievalQualityMonitor:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.llmops.quality.retrieval_quality)
        if isinstance(config, LLMOpsConfig):
            return cls(config.quality.retrieval_quality)
        return cls(config.llmops.quality.retrieval_quality)  # type: ignore[union-attr]

    # ── Embedding helpers ─────────────────────────────────────────

    def _get_embedder(self) -> Any | None:
        """Lazily load sentence-transformers; return *None* if unavailable."""
        if self._embedder_tried:
            return self._embedder
        self._embedder_tried = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("retrieval_quality.embedder_loaded", model="all-MiniLM-L6-v2")
        except Exception:
            log.info("retrieval_quality.embedder_unavailable", fallback="token_overlap")
        return self._embedder

    def _semantic_score(self, text_a: str, texts_b: list[str]) -> float:
        """Compute max cosine similarity between *text_a* and each text in *texts_b*.

        Args:
            text_a: The anchor text (query or response sentence).
            texts_b: Candidate texts (chunk texts).

        Returns:
            Maximum cosine similarity in ``[0, 1]``, or ``0.0`` on error.
        """
        embedder = self._get_embedder()
        if embedder is None or not texts_b:
            return 0.0
        try:
            import numpy as np

            emb_a = embedder.encode([text_a])[0]
            emb_b = embedder.encode(texts_b)
            norms = np.linalg.norm(emb_b, axis=1) * np.linalg.norm(emb_a) + 1e-9
            sims = np.dot(emb_b, emb_a) / norms
            return float(np.clip(np.max(sims), 0.0, 1.0))
        except Exception as exc:
            log.warning("retrieval_quality.semantic_score_failed", error=str(exc))
            return 0.0

    @staticmethod
    def _blend(token_score: float, semantic_score: float | None) -> float:
        """Blend token and semantic scores when semantic is available."""
        if semantic_score is None:
            return token_score
        return _TOKEN_WEIGHT * token_score + _SEMANTIC_WEIGHT * semantic_score

    # ── Evaluate ──────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        response: str,
        chunks: list[str | dict[str, Any]],
    ) -> RetrievalQualityResult:
        """Evaluate retrieval quality for a single RAG call.

        Args:
            query: The user query.
            response: The LLM response text.
            chunks: Retrieved context chunks (strings or dicts with a ``text`` key).

        Returns:
            A :class:`RetrievalQualityResult` with per-metric scores.
        """
        chunk_texts = [c if isinstance(c, str) else str(c.get("text", c)) for c in chunks]
        result = RetrievalQualityResult(chunks_retrieved=len(chunk_texts))
        if not chunk_texts:
            return result

        has_embedder = self._get_embedder() is not None

        q_tokens = set(_tokenise(query))
        r_tokens = set(_tokenise(response))

        # ── Relevance ─────────────────────────────────────────────
        token_relevances = [
            len(q_tokens & set(_tokenise(c))) / max(1, len(q_tokens)) for c in chunk_texts
        ]
        token_relevance = sum(token_relevances) / len(token_relevances)

        sem_relevance: float | None = None
        if has_embedder:
            sem_relevance = self._semantic_score(query, chunk_texts)

        result.relevance = self._blend(token_relevance, sem_relevance)

        # ── Chunk utilisation ─────────────────────────────────────
        used = sum(1 for c in chunk_texts if set(_tokenise(c)) & r_tokens)
        result.chunks_used = used
        result.chunk_utilisation = used / max(1, len(chunk_texts))

        # ── Answer coverage ───────────────────────────────────────
        result.answer_coverage = len(q_tokens & r_tokens) / max(1, len(q_tokens))

        # ── Faithfulness ──────────────────────────────────────────
        chunk_tokens: set[str] = set()
        for c in chunk_texts:
            chunk_tokens.update(_tokenise(c))
        token_faith = len(r_tokens & chunk_tokens) / max(1, len(r_tokens))

        sem_faith: float | None = None
        if has_embedder:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
            if sentences:
                sem_scores = [self._semantic_score(s, chunk_texts) for s in sentences]
                sem_faith = sum(sem_scores) / len(sem_scores)

        result.faithfulness = self._blend(token_faith, sem_faith)

        result.metadata["scoring_mode"] = "blended" if has_embedder else "token_overlap"

        if result.relevance < self.config.min_relevance:
            log.warning("retrieval.low_relevance", score=result.relevance)
        if result.faithfulness < self.config.min_faithfulness:
            log.warning("retrieval.low_faithfulness", score=result.faithfulness)
        return result


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 2]

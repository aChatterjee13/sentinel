"""Embedding-based semantic drift monitor for LLM outputs."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from sentinel.config.schema import LLMOpsConfig, SemanticDriftConfig
from sentinel.core.types import AlertSeverity, DriftReport

log = structlog.get_logger(__name__)


class SemanticDriftMonitor:
    """Tracks the centroid of LLM output embeddings over time.

    Drift is computed as the cosine distance between the reference
    centroid (fitted on a baseline corpus) and the centroid of a rolling
    window of recent outputs. When the distance exceeds the configured
    threshold, a :class:`DriftReport` is produced and routed through the
    standard notification + audit pipeline.

    All public methods are thread-safe. A :class:`threading.Lock` serialises
    access to the mutable window, reference centroid, and lazy-loaded embedder.
    """

    def __init__(
        self,
        config: SemanticDriftConfig | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        window_size: int = 500,
    ):
        self.config = config or SemanticDriftConfig()
        effective_window_size = getattr(self.config, "window_size", None) or window_size
        self._embed_fn = embed_fn
        self._embedder = None
        self._reference_centroid: np.ndarray | None = None
        self._reference_n: int = 0
        self._window: deque[np.ndarray] = deque(maxlen=effective_window_size)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Any) -> SemanticDriftMonitor:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.llmops.quality.semantic_drift)
        if isinstance(config, LLMOpsConfig):
            return cls(config.quality.semantic_drift)
        return cls(config.llmops.quality.semantic_drift)  # type: ignore[union-attr]

    # ── Embedding ─────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(texts), dtype=float)
        if self._embedder is None:
            try:
                from sentence_transformers import (
                    SentenceTransformer,  # type: ignore[import-not-found]
                )

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                raise RuntimeError(
                    "no embedder available — install sentence-transformers or pass embed_fn"
                ) from e
        return np.asarray(self._embedder.encode(texts), dtype=float)

    # ── Fit / observe / detect ────────────────────────────────────

    def fit(self, reference_outputs: list[str]) -> None:
        """Compute the reference centroid from a baseline corpus."""
        if not reference_outputs:
            raise ValueError("reference_outputs is empty")
        embeddings = self._embed(reference_outputs)
        with self._lock:
            self._reference_centroid = embeddings.mean(axis=0)
            self._reference_n = len(reference_outputs)
        log.info("semantic_drift.fitted", n=self._reference_n, dim=embeddings.shape[1])

    def observe(self, response: str) -> None:
        """Add a single response to the rolling window."""
        emb = self._embed([response])[0]
        with self._lock:
            self._window.append(emb)

    def detect(self, model_name: str = "llm") -> DriftReport:
        """Compute drift between the rolling window centroid and the reference."""
        with self._lock:
            if self._reference_centroid is None:
                raise RuntimeError("call fit() with a reference corpus first")
            if not self._window:
                return DriftReport(
                    model_name=model_name,
                    method="semantic_drift",
                    is_drifted=False,
                    severity=AlertSeverity.INFO,
                    test_statistic=0.0,
                    window=self.config.window,
                    metadata={"reason": "no observations"},
                )

            current = np.asarray(self._window).mean(axis=0)
            ref = self._reference_centroid
            ref_n = self._reference_n

        distance = float(_cosine_distance(ref, current))
        is_drifted = distance > self.config.threshold
        severity = AlertSeverity.from_score(
            distance,
            thresholds={
                "warning": self.config.threshold * 0.5,
                "high": self.config.threshold,
                "critical": self.config.threshold * 2,
            },
        )

        return DriftReport(
            model_name=model_name,
            method="semantic_drift",
            is_drifted=is_drifted,
            severity=severity,
            test_statistic=distance,
            window=self.config.window,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "reference_n": ref_n,
                "window_n": len(self._window),
                "embedding_model": self.config.embedding_model,
            },
        )

    def reset_window(self) -> None:
        """Clear the rolling observation window."""
        with self._lock:
            self._window.clear()


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 1.0
    return 1.0 - float(np.dot(a, b) / norm)

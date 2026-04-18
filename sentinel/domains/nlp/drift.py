"""NLP-specific drift detectors — vocabulary, embedding, label distribution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from sentinel.core.types import DriftReport
from sentinel.domains.nlp.text_stats import TextStatsMonitor
from sentinel.observability.drift.base import BaseDriftDetector


class VocabularyDriftDetector(BaseDriftDetector):
    """Detect vocabulary drift via OOV rate and new token frequency."""

    method_name = "vocabulary_drift"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.05,
        top_new_tokens_k: int = 50,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self._monitor = TextStatsMonitor(oov_threshold=threshold, top_new_tokens_k=top_new_tokens_k)

    def fit(self, reference: Any) -> None:
        texts = list(reference) if not isinstance(reference, str) else [reference]
        self._monitor.fit(texts)
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted:
            raise RuntimeError("VocabularyDriftDetector not fitted")
        texts = list(current) if not isinstance(current, str) else [current]
        stats = self._monitor.evaluate(texts)
        is_drifted = stats.oov_rate > self.threshold
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=is_drifted,
            severity=self._severity_from_score(stats.oov_rate),
            test_statistic=stats.oov_rate,
            feature_scores={
                "oov_rate": stats.oov_rate,
                "unique_tokens": float(stats.unique_tokens),
            },
            drifted_features=stats.new_tokens[:10] if is_drifted else [],
            metadata={
                "n_documents": stats.n_documents,
                "avg_length": stats.avg_length,
                "avg_token_count": stats.avg_token_count,
                "top_new_tokens": stats.new_tokens,
            },
        )


class EmbeddingDriftDetector(BaseDriftDetector):
    """Detect drift in the embedding space using centroid cosine distance.

    The detector accepts pre-computed embeddings (an ``(n, d)`` matrix).
    For raw text inputs, callers should embed them with their preferred
    model first — the SDK does not pull in heavy embedding dependencies
    by default.
    """

    method_name = "embedding_drift"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.1,
        method: str = "cosine_centroid",
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self.method = method
        self._reference_centroid: np.ndarray | None = None
        self._reference_cov: np.ndarray | None = None

    def fit(self, reference: Any) -> None:
        arr = np.asarray(reference, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("EmbeddingDriftDetector expects an (n, d) embedding matrix")
        self._reference_centroid = arr.mean(axis=0)
        if self.method == "mahalanobis":
            self._reference_cov = np.cov(arr.T) + np.eye(arr.shape[1]) * 1e-6
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted or self._reference_centroid is None:
            raise RuntimeError("EmbeddingDriftDetector not fitted")
        arr = np.asarray(current, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("expected (n, d) embedding matrix")
        current_centroid = arr.mean(axis=0)
        if self.method == "mahalanobis" and self._reference_cov is not None:
            diff = current_centroid - self._reference_centroid
            try:
                inv = np.linalg.pinv(self._reference_cov)
                score = float(np.sqrt(diff @ inv @ diff))
            except np.linalg.LinAlgError:
                score = float(np.linalg.norm(diff))
        elif self.method == "mmd":
            score = _rbf_mmd(arr, self._reference_centroid)
        else:
            score = _cosine_distance(self._reference_centroid, current_centroid)
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=score >= self.threshold,
            severity=self._severity_from_score(score),
            test_statistic=score,
            feature_scores={"distance": score},
            drifted_features=["embedding_centroid"] if score >= self.threshold else [],
            metadata={"method": self.method},
        )


class LabelDistributionDriftDetector(BaseDriftDetector):
    """Chi-squared drift detector for predicted label distributions."""

    method_name = "label_distribution"

    def __init__(self, model_name: str, threshold: float = 0.05, **kwargs: Any):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self._reference: dict[str, float] | None = None

    def fit(self, reference: Iterable[str]) -> None:
        counter = Counter(str(x) for x in reference)
        total = sum(counter.values()) or 1
        self._reference = {k: v / total for k, v in counter.items()}
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise RuntimeError("LabelDistributionDriftDetector not fitted")
        counter = Counter(str(x) for x in current)
        total = sum(counter.values()) or 1
        current_dist = {k: v / total for k, v in counter.items()}
        all_labels = set(self._reference) | set(current_dist)
        chi2 = 0.0
        for label in all_labels:
            expected = self._reference.get(label, 1e-9)
            observed = current_dist.get(label, 0.0)
            chi2 += (observed - expected) ** 2 / max(expected, 1e-9)
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=chi2 >= self.threshold,
            severity=self._severity_from_score(chi2),
            test_statistic=chi2,
            feature_scores={k: current_dist.get(k, 0.0) for k in all_labels},
            drifted_features=sorted(all_labels)[:10] if chi2 >= self.threshold else [],
            metadata={"reference": self._reference, "current": current_dist},
        )


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def _rbf_mmd(samples: np.ndarray, reference_centroid: np.ndarray, sigma: float = 1.0) -> float:
    """A cheap MMD proxy: distance between centroid and samples in RBF kernel space."""
    diffs = samples - reference_centroid
    sq = np.sum(diffs * diffs, axis=1)
    return float(1.0 - np.mean(np.exp(-sq / (2 * sigma * sigma))))

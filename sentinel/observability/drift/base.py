"""Abstract base class for all drift detectors."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog

from sentinel.core.types import AlertSeverity, DriftReport

log = structlog.get_logger(__name__)

ArrayLike = Any  # numpy arrays, pandas DataFrames, lists — duck-typed


class BaseDriftDetector(ABC):
    """Abstract drift detector.

    All detectors must implement `fit` (capture a reference) and `detect`
    (compare current data to the reference and return a `DriftReport`).

    A ``threading.Lock`` is provided on every instance as ``self._lock``
    so that subclasses can guard ``fit`` / ``detect`` / ``reset`` against
    concurrent mutation.
    """

    method_name: str = "base"
    requires_actuals: bool = False

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.2,
        feature_names: list[str] | None = None,
        **_: Any,
    ) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        self.model_name = model_name
        self.threshold = threshold
        self.feature_names = feature_names
        self._reference: np.ndarray | None = None
        self._fitted = False
        self._lock = threading.Lock()

    @abstractmethod
    def fit(self, reference: ArrayLike) -> None:
        """Store the reference distribution / state."""

    @abstractmethod
    def detect(self, current: ArrayLike, **kwargs: Any) -> DriftReport:
        """Compare `current` to the reference and return a `DriftReport`."""

    def reset(self) -> None:
        """Forget the current reference. Use after a retrain."""
        self._reference = None
        self._fitted = False

    def is_fitted(self) -> bool:
        """True once `fit` has been called."""
        return self._fitted

    # ── helpers shared by subclasses ──────────────────────────────

    @staticmethod
    def _to_2d_array(x: ArrayLike) -> np.ndarray:
        """Coerce input to a 2-D float array."""
        if hasattr(x, "values"):  # pandas DataFrame / Series
            x = x.values
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _drop_nan_rows(arr: np.ndarray, context: str = "data") -> np.ndarray:
        """Drop rows containing any NaN and log a warning if rows were removed.

        Args:
            arr: 2-D float array.
            context: Label for log messages (e.g. ``"reference"`` or ``"current"``).

        Returns:
            The array with NaN-containing rows removed.
        """
        nan_mask = np.isnan(arr).any(axis=1)
        n_dropped = int(nan_mask.sum())
        if n_dropped > 0:
            log.warning(
                "drift.nan_rows_dropped",
                context=context,
                rows_dropped=n_dropped,
                rows_remaining=int(arr.shape[0] - n_dropped),
            )
            arr = arr[~nan_mask]
        return arr

    @staticmethod
    def _resolve_feature_names(arr: np.ndarray, names: list[str] | None) -> list[str]:
        if names is not None and len(names) == arr.shape[1]:
            return list(names)
        return [f"feature_{i}" for i in range(arr.shape[1])]

    def _severity_from_score(self, score: float) -> AlertSeverity:
        """Map a drift score to a severity using the configured threshold.

        Anything below the threshold is INFO, then 1x = WARNING, 1.5x = HIGH,
        2x = CRITICAL.
        """
        if score < self.threshold:
            return AlertSeverity.INFO
        if score < self.threshold * 1.5:
            return AlertSeverity.WARNING
        if score < self.threshold * 2:
            return AlertSeverity.HIGH
        return AlertSeverity.CRITICAL

    def _empty_report(self, note: str) -> DriftReport:
        """Return a safe no-drift report when all data was dropped."""
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=False,
            severity=AlertSeverity.INFO,
            test_statistic=0.0,
            feature_scores={},
            drifted_features=[],
            metadata={"warning": note},
        )

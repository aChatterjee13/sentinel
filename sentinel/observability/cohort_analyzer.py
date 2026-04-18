"""Cohort-based performance analysis for production prediction monitoring."""

from __future__ import annotations

import threading
from collections import defaultdict, deque

import numpy as np
import structlog

from sentinel.config.schema import CohortAnalysisConfig
from sentinel.core.types import (
    CohortComparativeReport,
    CohortMetrics,
    CohortPerformanceReport,
)

log = structlog.get_logger(__name__)

# ── Internal buffer entry ──────────────────────────────────────────

_PredEntry = tuple[dict[str, float], float, float | None]
"""(features_dict, prediction, actual_or_None)."""


class CohortAnalyzer:
    """Segments predictions into cohorts and computes per-cohort health.

    Thread-safe: uses a lock around the prediction buffers.  The analyser
    is deliberately lightweight — it stores scalar features as dicts and
    computes statistics on demand.

    Example:
        >>> analyzer = CohortAnalyzer(config, model_name="fraud_v2")
        >>> analyzer.add_prediction({"age": 35}, 0.9, 1.0, cohort_id="age_30_40")
        >>> report = analyzer.get_cohort_report("age_30_40")
    """

    def __init__(self, config: CohortAnalysisConfig, model_name: str) -> None:
        self.config = config
        self.model_name = model_name
        self._lock = threading.Lock()
        # cohort_id → deque of (features_dict, prediction, actual|None)
        self._buffers: dict[str, deque[_PredEntry]] = defaultdict(
            lambda: deque(maxlen=config.buffer_size)
        )

    # ── Ingestion ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all accumulated prediction data across all cohorts."""
        with self._lock:
            self._buffers.clear()

    def add_prediction(
        self,
        features: dict[str, float],
        prediction: float,
        actual: float | None = None,
        cohort_id: str | None = None,
    ) -> None:
        """Buffer a prediction under its cohort.

        Args:
            features: Scalar feature dict for this prediction.
            prediction: Model prediction value.
            actual: Ground-truth label (optional; may arrive later).
            cohort_id: Explicit cohort ID.  If ``None`` and
                ``config.cohort_column`` is set, the cohort is derived
                from ``features[cohort_column]``.  If neither is available
                the prediction is silently dropped.
        """
        cid = cohort_id
        if cid is None and self.config.cohort_column:
            val = features.get(self.config.cohort_column)
            if val is not None:
                cid = str(val)
        if cid is None:
            return

        with self._lock:
            if len(self._buffers) >= self.config.max_cohorts and cid not in self._buffers:
                log.warning(
                    "cohort_analyzer.max_cohorts_reached",
                    max_cohorts=self.config.max_cohorts,
                    cohort_id=cid,
                )
                return
            self._buffers[cid].append((features, prediction, actual))

    # ── Query API ──────────────────────────────────────────────────

    @property
    def cohort_ids(self) -> list[str]:
        """Return all tracked cohort IDs."""
        with self._lock:
            return list(self._buffers.keys())

    def cohort_count(self, cohort_id: str) -> int:
        """Number of buffered predictions for a cohort."""
        with self._lock:
            buf = self._buffers.get(cohort_id)
            return len(buf) if buf else 0

    # ── Analysis ───────────────────────────────────────────────────

    def _compute_metrics(self, cohort_id: str, entries: list[_PredEntry]) -> CohortMetrics:
        """Build a ``CohortMetrics`` from raw entries."""
        preds = np.array([e[1] for e in entries], dtype=float)
        actuals_raw = [e[2] for e in entries]
        has_actuals = any(a is not None for a in actuals_raw)

        mean_actual: float | None = None
        accuracy: float | None = None
        if has_actuals:
            valid = [(p, a) for p, a in zip(preds, actuals_raw, strict=False) if a is not None]
            if valid:
                va = np.array([a for _, a in valid], dtype=float)
                vp = np.array([p for p, _ in valid], dtype=float)
                mean_actual = float(va.mean())
                # Binary classification accuracy heuristic
                if set(va.tolist()).issubset({0.0, 1.0}):
                    binary_preds = (vp >= 0.5).astype(float)
                    accuracy = float((binary_preds == va).mean())

        return CohortMetrics(
            cohort_id=cohort_id,
            count=len(entries),
            mean_prediction=float(preds.mean()) if len(preds) else None,
            mean_actual=mean_actual,
            accuracy=accuracy,
        )

    def get_cohort_report(self, cohort_id: str) -> CohortPerformanceReport | None:
        """Generate a performance report for a single cohort.

        Returns ``None`` if the cohort does not exist.
        """
        with self._lock:
            buf = self._buffers.get(cohort_id)
            if buf is None:
                return None
            entries = list(buf)

        metrics = self._compute_metrics(cohort_id, entries)
        return CohortPerformanceReport(
            model_name=self.model_name,
            cohort_id=cohort_id,
            metrics=metrics,
        )

    def compare_cohorts(self) -> CohortComparativeReport:
        """Compare all tracked cohorts and flag performance disparities.

        A cohort is flagged when its accuracy (if available) deviates from
        the global mean by more than ``config.disparity_threshold`` (relative).
        """
        with self._lock:
            snapshot = {cid: list(buf) for cid, buf in self._buffers.items()}

        cohort_metrics: list[CohortMetrics] = []
        for cid, entries in snapshot.items():
            if len(entries) < self.config.min_samples_per_cohort:
                continue
            cohort_metrics.append(self._compute_metrics(cid, entries))

        # Global aggregates
        all_preds: list[float] = []
        all_actuals: list[float] = []
        for entries in snapshot.values():
            for _, pred, actual in entries:
                all_preds.append(pred)
                if actual is not None:
                    all_actuals.append(actual)

        global_mean_pred = float(np.mean(all_preds)) if all_preds else None
        global_acc: float | None = None
        if all_actuals and all_preds:
            all_entries = []
            for entries in snapshot.values():
                all_entries.extend(entries)
            valid = [(p, a) for _, p, a in all_entries if a is not None]
            if valid:
                va = np.array([a for _, a in valid], dtype=float)
                vp = np.array([p for p, _ in valid], dtype=float)
                if set(va.tolist()).issubset({0.0, 1.0}):
                    global_acc = float(((vp >= 0.5).astype(float) == va).mean())

        # Disparity detection
        flags: list[str] = []
        if global_acc is not None:
            for m in cohort_metrics:
                if m.accuracy is not None:
                    if global_acc == 0.0:
                        gap = abs(m.accuracy)
                    else:
                        gap = abs(m.accuracy - global_acc) / global_acc
                    if gap > self.config.disparity_threshold:
                        flags.append(m.cohort_id)

        return CohortComparativeReport(
            model_name=self.model_name,
            cohorts=cohort_metrics,
            disparity_flags=flags,
            global_mean_prediction=global_mean_pred,
            global_accuracy=global_acc,
        )

    def get_disparity_alerts(self) -> list[str]:
        """Convenience: return cohort IDs flagged for performance disparity."""
        return self.compare_cohorts().disparity_flags

    def reset(self) -> None:
        """Clear all prediction buffers."""
        with self._lock:
            self._buffers.clear()

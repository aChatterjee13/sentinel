"""Concept drift detectors — DDM, EDDM, ADWIN, Page-Hinkley.

These are streaming detectors: you push observations one at a time and the
detector raises a warning or drift signal when the error rate shifts.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np

from sentinel.core.types import AlertSeverity, DriftReport
from sentinel.observability.drift.base import ArrayLike, BaseDriftDetector


class DDMConceptDriftDetector(BaseDriftDetector):
    """Drift Detection Method (Gama et al. 2004).

    Tracks the running mean error rate and standard deviation. When the
    current rate exceeds (mean + warning_level*std) it warns; when it exceeds
    (mean + drift_level*std) it signals drift.
    """

    method_name = "ddm"
    requires_actuals = True

    def __init__(
        self,
        model_name: str,
        threshold: float = 3.0,  # alias for drift_level
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 30,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, feature_names=feature_names)
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self._n = 0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._error_rate = 0.0
        self._std = 0.0
        self._lock = threading.Lock()

    def fit(self, reference: ArrayLike) -> None:
        """DDM is online — `fit` initialises the running stats from a reference window."""
        ref = np.asarray(reference, dtype=float).ravel()
        with self._lock:
            for err in ref:
                self._update(float(err))
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        cur = np.asarray(current, dtype=float).ravel()
        warning = False
        drifted = False
        with self._lock:
            for err in cur:
                self._update(float(err))
                if self._n < self.min_samples:
                    continue
                sum_p_s = self._error_rate + self._std
                if sum_p_s < self._p_min + self._s_min:
                    self._p_min = self._error_rate
                    self._s_min = self._std
                if sum_p_s > self._p_min + self.drift_level * self._s_min:
                    drifted = True
                    break
                if sum_p_s > self._p_min + self.warning_level * self._s_min:
                    warning = True

            severity = AlertSeverity.INFO
            if drifted:
                severity = AlertSeverity.HIGH
            elif warning:
                severity = AlertSeverity.WARNING

            report = DriftReport(
                model_name=self.model_name,
                method=self.method_name,
                is_drifted=drifted,
                severity=severity,
                test_statistic=float(self._error_rate),
                feature_scores={"error_rate": self._error_rate, "std": self._std},
                drifted_features=["target"] if drifted else [],
                metadata={
                    "n_samples": self._n,
                    "warning_state": warning,
                    "p_min": self._p_min if self._p_min != float("inf") else None,
                },
            )
        return report

    def reset(self) -> None:
        with self._lock:
            super().reset()
            self._n = 0
            self._p_min = float("inf")
            self._s_min = float("inf")
            self._error_rate = 0.0
            self._std = 0.0

    def _update(self, err: float) -> None:
        self._n += 1
        # Welford-style running mean of binary errors
        self._error_rate += (err - self._error_rate) / self._n
        if self._n >= 2 and 0 < self._error_rate < 1:
            self._std = float(np.sqrt(self._error_rate * (1 - self._error_rate) / self._n))
        else:
            self._std = 0.0


class EDDMConceptDriftDetector(BaseDriftDetector):
    """Early Drift Detection Method — distance-between-errors variant.

    Better suited for gradual drift than DDM.
    """

    method_name = "eddm"
    requires_actuals = True

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.9,
        warning_level: float = 0.95,
        min_samples: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold)
        self.warning_level = warning_level
        self.min_samples = min_samples
        self._n_errors = 0
        self._last_err_pos = 0
        self._distance_mean = 0.0
        self._distance_std = 0.0
        self._max_dist_mean = 0.0
        self._max_dist_std = 0.0
        self._sample_idx = 0
        self._lock = threading.Lock()

    def fit(self, reference: ArrayLike) -> None:
        with self._lock:
            for err in np.asarray(reference, dtype=float).ravel():
                self._update(float(err))
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        warning = False
        drifted = False
        with self._lock:
            for err in np.asarray(current, dtype=float).ravel():
                self._update(float(err))
                if self._n_errors < self.min_samples:
                    continue
                ratio = (self._distance_mean + 2 * self._distance_std) / max(
                    self._max_dist_mean + 2 * self._max_dist_std, 1e-9
                )
                if ratio < self.threshold:
                    drifted = True
                    break
                if ratio < self.warning_level:
                    warning = True

            severity = (
                AlertSeverity.HIGH
                if drifted
                else (AlertSeverity.WARNING if warning else AlertSeverity.INFO)
            )
            report = DriftReport(
                model_name=self.model_name,
                method=self.method_name,
                is_drifted=drifted,
                severity=severity,
                test_statistic=float(self._distance_mean),
                feature_scores={
                    "distance_mean": self._distance_mean,
                    "max_dist": self._max_dist_mean,
                },
                drifted_features=["target"] if drifted else [],
            )
        return report

    def reset(self) -> None:
        with self._lock:
            super().reset()
            self._n_errors = 0
            self._last_err_pos = 0
            self._distance_mean = 0.0
            self._distance_std = 0.0
            self._max_dist_mean = 0.0
            self._max_dist_std = 0.0
            self._sample_idx = 0

    def _update(self, err: float) -> None:
        self._sample_idx += 1
        if err > 0.5:  # treat as error
            self._n_errors += 1
            distance = self._sample_idx - self._last_err_pos
            self._last_err_pos = self._sample_idx
            old_mean = self._distance_mean
            self._distance_mean += (distance - self._distance_mean) / self._n_errors
            # Welford online variance — clamp to zero instead of abs()
            variance = self._distance_std**2 + (distance - old_mean) * (
                distance - self._distance_mean
            )
            self._distance_std = float(np.sqrt(max(0.0, variance)))
            running = self._distance_mean + 2 * self._distance_std
            if running > self._max_dist_mean + 2 * self._max_dist_std:
                self._max_dist_mean = self._distance_mean
                self._max_dist_std = self._distance_std


class ADWINConceptDriftDetector(BaseDriftDetector):
    """ADaptive WINdowing (Bifet & Gavaldà 2007).

    Maintains a variable-length window of recent observations. When the
    window can be split into two sub-windows whose means differ by more
    than a Hoeffding bound, the older half is dropped.

    This implementation is a simplified version sufficient for production
    drift alerting; the full algorithm uses exponential histograms.
    """

    method_name = "adwin"
    requires_actuals = True

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.002,  # delta — confidence level
        max_window: int = 5000,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold)
        self.delta = threshold
        self.max_window = max_window
        self._window: deque[float] = deque(maxlen=max_window)
        self._lock = threading.Lock()

    def fit(self, reference: ArrayLike) -> None:
        with self._lock:
            for x in np.asarray(reference, dtype=float).ravel():
                self._window.append(float(x))
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        with self._lock:
            for x in np.asarray(current, dtype=float).ravel():
                self._window.append(float(x))

            n = len(self._window)
            drifted = False
            cut_point = -1
            for split in range(1, n):
                w0 = np.fromiter((self._window[i] for i in range(split)), dtype=float)
                w1 = np.fromiter((self._window[i] for i in range(split, n)), dtype=float)
                if len(w0) < 5 or len(w1) < 5:
                    continue
                mean_diff = abs(w0.mean() - w1.mean())
                m = 1 / (1 / len(w0) + 1 / len(w1))
                eps_cut = float(np.sqrt((1 / (2 * m)) * np.log(4 * n / self.delta)))
                if mean_diff > eps_cut:
                    drifted = True
                    cut_point = split
                    break

            if drifted and cut_point >= 0:
                # Drop older half
                for _ in range(cut_point):
                    self._window.popleft()

            report = DriftReport(
                model_name=self.model_name,
                method=self.method_name,
                is_drifted=drifted,
                severity=AlertSeverity.HIGH if drifted else AlertSeverity.INFO,
                test_statistic=float(np.mean(self._window) if self._window else 0.0),
                feature_scores={"window_size": float(len(self._window))},
                drifted_features=["target"] if drifted else [],
            )
        return report

    def reset(self) -> None:
        with self._lock:
            super().reset()
            self._window.clear()


class PageHinkleyDriftDetector(BaseDriftDetector):
    """Page-Hinkley test — CUSUM variant for detecting mean shifts."""

    method_name = "page_hinkley"
    requires_actuals = True

    def __init__(
        self,
        model_name: str,
        threshold: float = 50.0,
        delta: float = 0.005,
        alpha: float = 1 - 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold)
        self.delta = delta
        self.alpha = alpha
        self._mean = 0.0
        self._n = 0
        self._cum_sum = 0.0
        self._min_cum_sum = 0.0
        self._lock = threading.Lock()

    def fit(self, reference: ArrayLike) -> None:
        with self._lock:
            for x in np.asarray(reference, dtype=float).ravel():
                self._update(float(x))
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        with self._lock:
            drifted = False
            for x in np.asarray(current, dtype=float).ravel():
                self._update(float(x))
                ph = self._cum_sum - self._min_cum_sum
                if ph > self.threshold:
                    drifted = True
                    break

            report = DriftReport(
                model_name=self.model_name,
                method=self.method_name,
                is_drifted=drifted,
                severity=AlertSeverity.HIGH if drifted else AlertSeverity.INFO,
                test_statistic=float(self._cum_sum - self._min_cum_sum),
                feature_scores={"mean": self._mean, "cumsum": self._cum_sum},
                drifted_features=["target"] if drifted else [],
            )
            if drifted:
                self._cum_sum = 0.0
                self._min_cum_sum = 0.0
        return report

    def reset(self) -> None:
        with self._lock:
            super().reset()
            self._mean = 0.0
            self._n = 0
            self._cum_sum = 0.0
            self._min_cum_sum = 0.0

    def _update(self, x: float) -> None:
        self._n += 1
        self._mean = self.alpha * self._mean + (1 - self.alpha) * x if self._n > 1 else x
        self._cum_sum = self._cum_sum + (x - self._mean - self.delta)
        self._min_cum_sum = min(self._min_cum_sum, self._cum_sum)

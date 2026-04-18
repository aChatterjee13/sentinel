"""Data drift detectors — PSI, KS, JS divergence, chi-squared, Wasserstein.

All detectors implement `BaseDriftDetector` and produce `DriftReport` objects.
They are pure NumPy with no SciPy dependency by default; SciPy is used opportunistically
when available for faster / more accurate p-value computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.core.exceptions import DriftDetectionError
from sentinel.core.types import AlertSeverity, DriftReport
from sentinel.observability.drift.base import ArrayLike, BaseDriftDetector

_EPS = 1e-8


def _safe_log(x: np.ndarray) -> np.ndarray:
    """log(x) that survives zeros."""
    return np.log(np.clip(x, _EPS, None))


def _bin_distribution(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin two arrays into a shared histogram and return normalized densities."""
    if reference.size == 0 or current.size == 0:
        raise DriftDetectionError("cannot bin empty arrays")
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]
    if ref_clean.size == 0 or cur_clean.size == 0:
        raise DriftDetectionError("no finite values in data after NaN filtering")
    lo = float(min(ref_clean.min(), cur_clean.min()))
    hi = float(max(ref_clean.max(), cur_clean.max()))
    if lo == hi:
        # Constant feature — no drift signal possible
        return np.array([1.0]), np.array([1.0])
    edges = np.linspace(lo, hi, n_bins + 1)
    ref_hist, _ = np.histogram(ref_clean, bins=edges)
    cur_hist, _ = np.histogram(cur_clean, bins=edges)
    ref_dist = ref_hist / max(ref_hist.sum(), 1)
    cur_dist = cur_hist / max(cur_hist.sum(), 1)
    return ref_dist + _EPS, cur_dist + _EPS


# ── PSI ────────────────────────────────────────────────────────────


class PSIDriftDetector(BaseDriftDetector):
    """Population Stability Index drift detector.

    PSI is a symmetric KL-divergence variant widely used in BFSI for
    monitoring credit score and risk model inputs.

    Threshold guidance:
        < 0.1   stable
        0.1-0.2 moderate shift
        > 0.2   significant shift
    """

    method_name = "psi"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.2,
        feature_names: list[str] | None = None,
        n_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, feature_names=feature_names)
        self.n_bins = n_bins

    def fit(self, reference: ArrayLike) -> None:
        ref = self._to_2d_array(reference)
        ref = self._drop_nan_rows(ref, context="reference")
        self._reference = ref
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise DriftDetectionError("PSI detector must be fit() before detect()")
        cur = self._to_2d_array(current)
        cur = self._drop_nan_rows(cur, context="current")
        if cur.shape[0] == 0:
            return self._empty_report("all rows contained NaN — no data for drift check")
        if cur.shape[1] != self._reference.shape[1]:
            raise DriftDetectionError(
                f"feature count mismatch: ref={self._reference.shape[1]}, cur={cur.shape[1]}"
            )
        names = self._resolve_feature_names(cur, self.feature_names)

        scores: dict[str, float] = {}
        for i, name in enumerate(names):
            ref_dist, cur_dist = _bin_distribution(
                self._reference[:, i], cur[:, i], n_bins=self.n_bins
            )
            psi = float(np.sum((cur_dist - ref_dist) * _safe_log(cur_dist / ref_dist)))
            scores[name] = psi

        max_psi = max(scores.values()) if scores else 0.0
        drifted = [name for name, s in scores.items() if s >= self.threshold]
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=bool(drifted),
            severity=self._severity_from_score(max_psi),
            test_statistic=max_psi,
            feature_scores=scores,
            drifted_features=drifted,
            metadata={"n_bins": self.n_bins, "ref_size": int(self._reference.shape[0])},
        )


# ── Kolmogorov-Smirnov ─────────────────────────────────────────────


class KSDriftDetector(BaseDriftDetector):
    """Two-sample Kolmogorov-Smirnov drift detector.

    Pure-NumPy implementation that computes the max difference between
    empirical CDFs. p-value is approximated using the Smirnov asymptotic formula.
    Best for continuous features.
    """

    method_name = "ks"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.05,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # threshold here is interpreted as a p-value cutoff
        super().__init__(model_name=model_name, threshold=threshold, feature_names=feature_names)

    def fit(self, reference: ArrayLike) -> None:
        ref = self._to_2d_array(reference)
        ref = self._drop_nan_rows(ref, context="reference")
        self._reference = ref
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise DriftDetectionError("KS detector must be fit() before detect()")
        cur = self._to_2d_array(current)
        cur = self._drop_nan_rows(cur, context="current")
        if cur.shape[0] == 0:
            return self._empty_report("all rows contained NaN — no data for drift check")
        names = self._resolve_feature_names(cur, self.feature_names)

        scores: dict[str, float] = {}
        pvalues: dict[str, float] = {}
        for i, name in enumerate(names):
            d, p = self._ks_statistic(self._reference[:, i], cur[:, i])
            scores[name] = d
            pvalues[name] = p

        drifted = [n for n, p in pvalues.items() if p < self.threshold]
        max_d = max(scores.values()) if scores else 0.0
        min_p = min(pvalues.values()) if pvalues else 1.0

        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=bool(drifted),
            severity=self._severity_from_score(max_d),
            test_statistic=max_d,
            p_value=min_p,
            feature_scores=scores,
            drifted_features=drifted,
            metadata={"per_feature_pvalues": pvalues},
        )

    @staticmethod
    def _ks_statistic(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
        """Two-sample KS statistic and asymptotic p-value."""
        ref = np.sort(ref[np.isfinite(ref)])
        cur = np.sort(cur[np.isfinite(cur)])
        n1, n2 = len(ref), len(cur)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        all_vals = np.concatenate([ref, cur])
        cdf_ref = np.searchsorted(ref, all_vals, side="right") / n1
        cdf_cur = np.searchsorted(cur, all_vals, side="right") / n2
        d = float(np.max(np.abs(cdf_ref - cdf_cur)))
        # Smirnov asymptotic p-value
        en = np.sqrt(n1 * n2 / (n1 + n2))
        # Series expansion of the Kolmogorov distribution
        lam = (en + 0.12 + 0.11 / en) * d
        j = np.arange(1, 101)
        p = 2.0 * float(np.sum((-1) ** (j - 1) * np.exp(-2.0 * lam**2 * j**2)))
        p = max(0.0, min(1.0, p))
        return d, p


# ── Jensen-Shannon divergence ──────────────────────────────────────


class JSDivergenceDriftDetector(BaseDriftDetector):
    """Symmetric Jensen-Shannon divergence drift detector. Bounded in [0,1]."""

    method_name = "js_divergence"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.1,
        feature_names: list[str] | None = None,
        n_bins: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, feature_names=feature_names)
        self.n_bins = n_bins

    def fit(self, reference: ArrayLike) -> None:
        ref = self._to_2d_array(reference)
        ref = self._drop_nan_rows(ref, context="reference")
        self._reference = ref
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise DriftDetectionError("JS detector must be fit() before detect()")
        cur = self._to_2d_array(current)
        cur = self._drop_nan_rows(cur, context="current")
        if cur.shape[0] == 0:
            return self._empty_report("all rows contained NaN — no data for drift check")
        names = self._resolve_feature_names(cur, self.feature_names)

        scores: dict[str, float] = {}
        for i, name in enumerate(names):
            p, q = _bin_distribution(self._reference[:, i], cur[:, i], n_bins=self.n_bins)
            m = 0.5 * (p + q)
            kl_pm = float(np.sum(p * _safe_log(p / m)))
            kl_qm = float(np.sum(q * _safe_log(q / m)))
            jsd = 0.5 * (kl_pm + kl_qm)
            # Normalise by ln(2) so the result is in [0, 1]
            scores[name] = jsd / np.log(2)

        max_jsd = max(scores.values()) if scores else 0.0
        drifted = [n for n, s in scores.items() if s >= self.threshold]
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=bool(drifted),
            severity=self._severity_from_score(max_jsd),
            test_statistic=max_jsd,
            feature_scores=scores,
            drifted_features=drifted,
        )


# ── Chi-squared ────────────────────────────────────────────────────


class ChiSquaredDriftDetector(BaseDriftDetector):
    """Pearson chi-squared test for categorical / discretised features."""

    method_name = "chi_squared"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.05,
        feature_names: list[str] | None = None,
        n_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        # threshold = p-value cutoff
        super().__init__(model_name=model_name, threshold=threshold, feature_names=feature_names)
        self.n_bins = n_bins

    def fit(self, reference: ArrayLike) -> None:
        ref = self._to_2d_array(reference)
        ref = self._drop_nan_rows(ref, context="reference")
        self._reference = ref
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise DriftDetectionError("Chi-squared detector must be fit() before detect()")
        cur = self._to_2d_array(current)
        cur = self._drop_nan_rows(cur, context="current")
        if cur.shape[0] == 0:
            return DriftReport(
                model_name=self.model_name,
                method=self.method_name,
                is_drifted=False,
                severity=AlertSeverity.INFO,
                test_statistic=0.0,
                p_value=1.0,
                feature_scores={},
                drifted_features=[],
                metadata={"warning": "empty current data"},
            )
        names = self._resolve_feature_names(cur, self.feature_names)

        scores: dict[str, float] = {}
        pvalues: dict[str, float] = {}
        for i, name in enumerate(names):
            ref_dist, cur_dist = _bin_distribution(
                self._reference[:, i], cur[:, i], n_bins=self.n_bins
            )
            n_cur = max(int(cur.shape[0]), 1)
            expected = ref_dist * n_cur
            observed = cur_dist * n_cur
            chi2 = float(np.sum((observed - expected) ** 2 / np.clip(expected, _EPS, None)))
            df = max(len(ref_dist) - 1, 1)
            scores[name] = chi2
            pvalues[name] = self._chi2_pvalue(chi2, df)

        drifted = [n for n, p in pvalues.items() if p < self.threshold]
        max_chi2 = max(scores.values()) if scores else 0.0
        min_p = min(pvalues.values()) if pvalues else 1.0
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=bool(drifted),
            severity=self._severity_from_score(max_chi2),
            test_statistic=max_chi2,
            p_value=min_p,
            feature_scores=scores,
            drifted_features=drifted,
            metadata={"per_feature_pvalues": pvalues},
        )

    @staticmethod
    def _chi2_pvalue(chi2: float, df: int) -> float:
        """Survival function of the chi-squared distribution.

        Falls back to a Wilson-Hilferty normal approximation when SciPy is
        unavailable. Accurate to ~2 decimal places, which is fine for alerting.
        """
        try:
            from scipy.stats import chi2 as _scipy_chi2  # type: ignore[import-not-found]

            return float(_scipy_chi2.sf(chi2, df))
        except ImportError:
            # Wilson-Hilferty
            if df <= 0:
                return 1.0
            z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
            # Standard normal SF via erfc
            from math import erfc, sqrt

            return float(0.5 * erfc(z / sqrt(2)))


# ── Wasserstein ────────────────────────────────────────────────────


class WassersteinDriftDetector(BaseDriftDetector):
    """1-Wasserstein (earth mover's) distance for continuous features."""

    method_name = "wasserstein"

    def fit(self, reference: ArrayLike) -> None:
        ref = self._to_2d_array(reference)
        ref = self._drop_nan_rows(ref, context="reference")
        self._reference = ref
        self._fitted = True

    def detect(self, current: ArrayLike, **_: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise DriftDetectionError("Wasserstein detector must be fit() before detect()")
        cur = self._to_2d_array(current)
        cur = self._drop_nan_rows(cur, context="current")
        if cur.shape[0] == 0:
            return self._empty_report("all rows contained NaN — no data for drift check")
        names = self._resolve_feature_names(cur, self.feature_names)

        scores: dict[str, float] = {}
        for i, name in enumerate(names):
            scores[name] = self._wasserstein_1d(self._reference[:, i], cur[:, i])

        max_w = max(scores.values()) if scores else 0.0
        drifted = [n for n, s in scores.items() if s >= self.threshold]
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=bool(drifted),
            severity=self._severity_from_score(max_w),
            test_statistic=max_w,
            feature_scores=scores,
            drifted_features=drifted,
        )

    @staticmethod
    def _wasserstein_1d(ref: np.ndarray, cur: np.ndarray) -> float:
        """Closed-form 1-D Wasserstein via sorted CDF integration."""
        ref = np.sort(ref[np.isfinite(ref)])
        cur = np.sort(cur[np.isfinite(cur)])
        if len(ref) == 0 or len(cur) == 0:
            return 0.0
        all_vals = np.sort(np.concatenate([ref, cur]))
        deltas = np.diff(all_vals)
        cdf_ref = np.searchsorted(ref, all_vals[:-1], side="right") / len(ref)
        cdf_cur = np.searchsorted(cur, all_vals[:-1], side="right") / len(cur)
        return float(np.sum(np.abs(cdf_ref - cdf_cur) * deltas))

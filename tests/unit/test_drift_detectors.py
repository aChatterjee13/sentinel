"""Unit tests for the core drift detectors."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.exceptions import DriftDetectionError
from sentinel.core.types import AlertSeverity
from sentinel.observability.drift.data_drift import (
    ChiSquaredDriftDetector,
    JSDivergenceDriftDetector,
    KSDriftDetector,
    PSIDriftDetector,
    WassersteinDriftDetector,
)


class TestPSIDriftDetector:
    def test_no_drift_on_identical_distribution(self, stable_features: np.ndarray) -> None:
        d = PSIDriftDetector(model_name="m", threshold=0.2)
        d.fit(stable_features)
        report = d.detect(stable_features)
        assert not report.is_drifted
        assert report.test_statistic < 0.1
        assert report.severity == AlertSeverity.INFO

    def test_detects_obvious_drift(
        self, stable_features: np.ndarray, drifted_features: np.ndarray
    ) -> None:
        d = PSIDriftDetector(model_name="m", threshold=0.2)
        d.fit(stable_features)
        report = d.detect(drifted_features)
        assert report.is_drifted
        assert report.test_statistic > 0.2
        assert len(report.drifted_features) > 0
        assert report.severity in (
            AlertSeverity.WARNING,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL,
        )

    def test_raises_when_not_fitted(self, stable_features: np.ndarray) -> None:
        d = PSIDriftDetector(model_name="m")
        with pytest.raises(DriftDetectionError):
            d.detect(stable_features)

    def test_feature_count_mismatch(self) -> None:
        d = PSIDriftDetector(model_name="m")
        d.fit(np.ones((10, 3)))
        with pytest.raises(DriftDetectionError):
            d.detect(np.ones((10, 4)))

    def test_reset_clears_state(self, stable_features: np.ndarray) -> None:
        d = PSIDriftDetector(model_name="m")
        d.fit(stable_features)
        assert d.is_fitted()
        d.reset()
        assert not d.is_fitted()


class TestKSDriftDetector:
    def test_no_drift_returns_high_pvalue(self, rng: np.random.Generator) -> None:
        # Two independent samples from the SAME distribution should not drift
        ref = rng.normal(loc=0.0, scale=1.0, size=(500, 4))
        cur = rng.normal(loc=0.0, scale=1.0, size=(500, 4))
        d = KSDriftDetector(model_name="m", threshold=0.05)
        d.fit(ref)
        report = d.detect(cur)
        assert not report.is_drifted
        assert report.p_value is not None
        assert report.p_value > 0.05

    def test_detects_drift(self, stable_features: np.ndarray, drifted_features: np.ndarray) -> None:
        d = KSDriftDetector(model_name="m", threshold=0.05)
        d.fit(stable_features)
        report = d.detect(drifted_features)
        assert report.is_drifted
        assert report.p_value is not None and report.p_value < 0.05


class TestJSDivergenceDriftDetector:
    def test_zero_for_identical(self, stable_features: np.ndarray) -> None:
        d = JSDivergenceDriftDetector(model_name="m", threshold=0.1)
        d.fit(stable_features)
        report = d.detect(stable_features)
        assert report.test_statistic < 0.05

    def test_bounded_in_zero_one(
        self, stable_features: np.ndarray, drifted_features: np.ndarray
    ) -> None:
        d = JSDivergenceDriftDetector(model_name="m", threshold=0.1)
        d.fit(stable_features)
        report = d.detect(drifted_features)
        for score in report.feature_scores.values():
            assert 0.0 <= score <= 1.0


class TestChiSquaredDriftDetector:
    def test_handles_categorical_features(
        self, categorical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, cur = categorical_data
        d = ChiSquaredDriftDetector(model_name="m", threshold=0.05)
        d.fit(ref)
        report = d.detect(cur)
        assert report.method == "chi_squared"
        assert report.p_value is not None


class TestWassersteinDriftDetector:
    def test_zero_for_identical(self, stable_features: np.ndarray) -> None:
        d = WassersteinDriftDetector(model_name="m", threshold=0.5)
        d.fit(stable_features)
        report = d.detect(stable_features)
        assert report.test_statistic < 0.1

    def test_positive_for_shifted(
        self, stable_features: np.ndarray, drifted_features: np.ndarray
    ) -> None:
        d = WassersteinDriftDetector(model_name="m", threshold=0.5)
        d.fit(stable_features)
        report = d.detect(drifted_features)
        assert report.test_statistic > 0.5

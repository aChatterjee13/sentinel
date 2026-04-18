"""Tests for Gap-E: streaming concept drift wiring in SentinelClient."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.config.schema import (
    AuditConfig,
    ConceptDriftConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)
from sentinel.core.client import SentinelClient


def _make_config(*, concept_method: str = "ddm") -> SentinelConfig:
    return SentinelConfig(
        model=ModelConfig(name="test_model"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2),
            concept=ConceptDriftConfig(
                method=concept_method,
                warning_level=2.0,
                drift_level=3.0,
                min_samples=10,
            ),
        ),
        audit=AuditConfig(storage="local"),
    )


class TestConceptDriftDetectorBuilt:
    def test_detector_built_when_concept_configured(self) -> None:
        config = _make_config(concept_method="ddm")
        client = SentinelClient(config)
        try:
            assert client._concept_drift_detector is not None
            assert client._concept_drift_detector.method_name == "ddm"
        finally:
            client.close()

    def test_detector_not_built_without_concept(self) -> None:
        config = SentinelConfig(
            model=ModelConfig(name="test_model"),
            drift=DriftConfig(data=DataDriftConfig(method="psi", threshold=0.2)),
            audit=AuditConfig(storage="local"),
        )
        client = SentinelClient(config)
        try:
            assert client._concept_drift_detector is None
        finally:
            client.close()

    @pytest.mark.parametrize("method", ["ddm", "eddm", "adwin", "page_hinkley"])
    def test_all_concept_methods_build(self, method: str) -> None:
        config = _make_config(concept_method=method)
        client = SentinelClient(config)
        try:
            assert client._concept_drift_detector is not None
            assert client._concept_drift_detector.method_name == method
        finally:
            client.close()


class TestConceptDriftFromLogPrediction:
    def test_concept_fed_when_actual_provided(self) -> None:
        config = _make_config()
        client = SentinelClient(config)
        try:
            # Log predictions with actuals
            for _ in range(20):
                client.log_prediction(features={"x": 1.0}, prediction=1, actual=1)
            assert client._concept_observations == 20
        finally:
            client.close()

    def test_concept_not_fed_without_actual(self) -> None:
        config = _make_config()
        client = SentinelClient(config)
        try:
            for _ in range(10):
                client.log_prediction(features={"x": 1.0}, prediction=1)
            assert client._concept_observations == 0
        finally:
            client.close()


class TestComputeErrorSignal:
    def test_binary_classification_correct(self) -> None:
        assert SentinelClient._compute_error_signal(1, 1) == 0.0

    def test_binary_classification_wrong(self) -> None:
        assert SentinelClient._compute_error_signal(0, 1) == 1.0

    def test_regression_difference(self) -> None:
        assert SentinelClient._compute_error_signal(1.5, 2.0) == pytest.approx(0.5)

    def test_categorical_match(self) -> None:
        assert SentinelClient._compute_error_signal("cat", "cat") == 0.0

    def test_categorical_mismatch(self) -> None:
        assert SentinelClient._compute_error_signal("cat", "dog") == 1.0


class TestConceptDriftInCheckDrift:
    def test_concept_drift_merged_into_report(self) -> None:
        config = _make_config()
        client = SentinelClient(config)
        try:
            rng = np.random.default_rng(42)
            baseline = rng.normal(size=(100, 1))
            client.fit_baseline(baseline)

            # Log correct predictions first (no drift)
            for _ in range(30):
                client.log_prediction(features={"x": 1.0}, prediction=1, actual=1)

            report = client.check_drift(baseline)
            assert "concept_drift" in report.metadata
        finally:
            client.close()

    def test_concept_drift_promotes_overall_drift(self) -> None:
        config = _make_config()
        client = SentinelClient(config)
        try:
            rng = np.random.default_rng(42)
            baseline = rng.normal(size=(100, 1))
            client.fit_baseline(baseline)

            # Log many errors to trigger concept drift
            for _ in range(200):
                client.log_prediction(features={"x": 1.0}, prediction=0, actual=1)

            report = client.check_drift(baseline)
            concept_data = report.metadata.get("concept_drift", {})
            if concept_data.get("is_drifted"):
                assert report.is_drifted is True
        finally:
            client.close()

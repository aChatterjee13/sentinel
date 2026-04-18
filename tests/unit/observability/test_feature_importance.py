"""Tests for SHAP, permutation, and builtin feature importance computation."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sentinel.config.schema import FeatureHealthConfig
from sentinel.core.types import AlertSeverity, DriftReport
from sentinel.observability.drift.base import BaseDriftDetector
from sentinel.observability.feature_health import FeatureHealthMonitor

# ── Helpers ────────────────────────────────────────────────────────


class _StubDetector(BaseDriftDetector):
    """Detector that returns preset drift results."""

    def __init__(self) -> None:
        super().__init__(model_name="test", threshold=0.2)
        self._fitted = True

    def fit(self, reference_data: Any) -> None:
        self._fitted = True

    def detect(self, current_data: Any) -> DriftReport:
        return DriftReport(
            model_name="test",
            method="stub",
            is_drifted=False,
            severity=AlertSeverity.INFO,
            test_statistic=0.0,
            feature_scores={"f0": 0.05, "f1": 0.03},
            drifted_features=[],
        )

    def reset(self) -> None:
        pass


def _make_tree_model() -> Any:
    """Train a minimal DecisionTreeClassifier for testing."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf, X, y


# ── compute_importance — builtin ──────────────────────────────────


class TestComputeImportanceBuiltin:
    def test_builtin_with_sklearn_model(self) -> None:
        clf, X, y = _make_tree_model()
        config = FeatureHealthConfig(importance_method="builtin")
        monitor = FeatureHealthMonitor(config, model_name="m")

        result = monitor.compute_importance(clf, X, y, feature_names=["a", "b"])

        assert set(result.keys()) == {"a", "b"}
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in result.values())
        # Also stored internally
        assert monitor._importances == result

    def test_builtin_generates_feature_names(self) -> None:
        clf, X, y = _make_tree_model()
        config = FeatureHealthConfig(importance_method="builtin")
        monitor = FeatureHealthMonitor(config, model_name="m")

        result = monitor.compute_importance(clf, X, y)

        assert set(result.keys()) == {"f0", "f1"}


# ── compute_importance — permutation ──────────────────────────────


class TestComputeImportancePermutation:
    def test_permutation_importance(self) -> None:
        clf, X, y = _make_tree_model()
        config = FeatureHealthConfig(importance_method="permutation")
        monitor = FeatureHealthMonitor(config, model_name="m")

        result = monitor.compute_importance(clf, X, y, feature_names=["a", "b"])

        assert set(result.keys()) == {"a", "b"}
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in result.values())

    def test_permutation_falls_back_when_y_is_none(self) -> None:
        clf, X, _ = _make_tree_model()
        config = FeatureHealthConfig(importance_method="permutation")
        monitor = FeatureHealthMonitor(config, model_name="m")

        result = monitor.compute_importance(clf, X, y=None, feature_names=["a", "b"])

        # Should fall back to builtin and still produce valid importances
        assert set(result.keys()) == {"a", "b"}
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_permutation_falls_back_when_sklearn_missing(self) -> None:
        clf, X, y = _make_tree_model()
        config = FeatureHealthConfig(importance_method="permutation")
        monitor = FeatureHealthMonitor(config, model_name="m")

        with patch.dict(sys.modules, {"sklearn.inspection": None, "sklearn": None}):
            # Force re-import to fail
            result = monitor._compute_permutation_importance(clf, X, y, ["a", "b"])

        # Falls back to builtin
        assert set(result.keys()) == {"a", "b"}


# ── compute_importance — SHAP ─────────────────────────────────────


class TestComputeImportanceShap:
    @pytest.mark.skipif(
        not importlib.util.find_spec("shap"),
        reason="shap not installed",
    )
    def test_shap_importance(self) -> None:
        clf, X, y = _make_tree_model()
        config = FeatureHealthConfig(importance_method="shap")
        monitor = FeatureHealthMonitor(config, model_name="m")

        result = monitor.compute_importance(clf, X, y, feature_names=["a", "b"])

        assert set(result.keys()) == {"a", "b"}
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in result.values())

    def test_shap_falls_back_when_not_installed(self) -> None:
        clf, X, _y = _make_tree_model()
        config = FeatureHealthConfig(importance_method="shap")
        monitor = FeatureHealthMonitor(config, model_name="m")

        with patch.dict(sys.modules, {"shap": None}):
            result = monitor._compute_shap_importance(clf, X, ["a", "b"])

        # Falls back to builtin
        assert set(result.keys()) == {"a", "b"}
        assert abs(sum(result.values()) - 1.0) < 1e-6


# ── set_importances still works ───────────────────────────────────


class TestSetImportancesStillWorks:
    def test_manual_override(self) -> None:
        config = FeatureHealthConfig(importance_method="builtin")
        monitor = FeatureHealthMonitor(config, model_name="m")

        monitor.set_importances({"x": 0.7, "y": 0.3})
        assert monitor._importances == {"x": 0.7, "y": 0.3}

        # compute_importance should overwrite
        clf, X, y_arr = _make_tree_model()
        monitor.compute_importance(clf, X, y_arr, feature_names=["a", "b"])
        assert "x" not in monitor._importances
        assert "a" in monitor._importances


# ── fit_baseline auto-computes importance ─────────────────────────


class TestFitBaselineAutoComputes:
    def test_importance_auto_computed_when_model_provided(self) -> None:
        from sentinel.config.schema import SentinelConfig

        config = SentinelConfig(
            model={"name": "test_model", "type": "classification"},
            feature_health={"importance_method": "builtin"},
        )

        from sentinel.core.client import SentinelClient

        client = SentinelClient(config)
        assert client.feature_health._importances == {}

        clf, X, y = _make_tree_model()
        client.fit_baseline(X, model=clf, y=y)

        assert len(client.feature_health._importances) > 0
        assert abs(sum(client.feature_health._importances.values()) - 1.0) < 1e-6
        assert client._model is clf
        assert client._last_importance_calc is not None

    def test_importance_not_computed_without_model(self) -> None:
        from sentinel.config.schema import SentinelConfig

        config = SentinelConfig(
            model={"name": "test_model", "type": "classification"},
        )

        from sentinel.core.client import SentinelClient

        client = SentinelClient(config)
        _, X, _ = _make_tree_model()
        client.fit_baseline(X)

        assert client.feature_health._importances == {}
        assert client._last_importance_calc is None

    def test_fit_baseline_uses_stored_model(self) -> None:
        from sentinel.config.schema import SentinelConfig
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model={"name": "test_model", "type": "classification"},
            feature_health={"importance_method": "builtin"},
        )
        client = SentinelClient(config)
        clf, X, y = _make_tree_model()

        client.set_model(clf)
        client.fit_baseline(X, y=y)

        assert len(client.feature_health._importances) > 0


# ── set_model ─────────────────────────────────────────────────────


class TestSetModel:
    def test_set_model_stores_reference(self) -> None:
        from sentinel.config.schema import SentinelConfig
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model={"name": "test_model", "type": "classification"},
        )
        client = SentinelClient(config)
        assert client._model is None

        clf, _, _ = _make_tree_model()
        client.set_model(clf)
        assert client._model is clf


# ── recalculate_importance scheduler ──────────────────────────────


class TestRecalculateImportanceScheduler:
    def _make_client_with_model(
        self,
        recalculate: str = "weekly",
    ) -> tuple[Any, Any]:
        from sentinel.config.schema import SentinelConfig
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model={"name": "test_model", "type": "classification"},
            feature_health={
                "importance_method": "builtin",
                "recalculate_importance": recalculate,
            },
        )
        client = SentinelClient(config)
        clf, X, y = _make_tree_model()
        client.set_model(clf)
        client.fit_baseline(X, y=y)

        # Seed the prediction buffer so get_feature_health doesn't error
        for i in range(5):
            client.log_prediction(
                features={"f0": float(X[i, 0]), "f1": float(X[i, 1])},
                prediction=int(y[i]),
            )
        return client, clf

    def test_no_recompute_within_interval(self) -> None:
        client, _ = self._make_client_with_model("weekly")

        original_time = client._last_importance_calc

        # Calling get_feature_health should NOT recompute (within interval)
        client.get_feature_health()

        assert client._last_importance_calc == original_time

    def test_recompute_after_interval_elapsed(self) -> None:
        client, _ = self._make_client_with_model("weekly")

        # Backdate the last computation to 8 days ago
        client._last_importance_calc = datetime.now(timezone.utc) - timedelta(days=8)
        old_time = client._last_importance_calc

        client.get_feature_health()

        # Should have recomputed
        assert client._last_importance_calc is not None
        assert client._last_importance_calc > old_time

    def test_never_does_not_recompute(self) -> None:
        client, _ = self._make_client_with_model("never")

        # Backdate way past any threshold
        client._last_importance_calc = datetime.now(timezone.utc) - timedelta(days=365)
        old_time = client._last_importance_calc

        client.get_feature_health()

        # Should NOT have recomputed
        assert client._last_importance_calc == old_time

    def test_daily_interval_triggers(self) -> None:
        client, _ = self._make_client_with_model("daily")

        client._last_importance_calc = datetime.now(timezone.utc) - timedelta(hours=25)
        old_time = client._last_importance_calc

        client.get_feature_health()

        assert client._last_importance_calc > old_time

    def test_no_recompute_without_model(self) -> None:
        from sentinel.config.schema import SentinelConfig
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model={"name": "test_model", "type": "classification"},
            feature_health={
                "importance_method": "builtin",
                "recalculate_importance": "daily",
            },
        )
        client = SentinelClient(config)
        _, X, y = _make_tree_model()
        client.fit_baseline(X)

        for i in range(5):
            client.log_prediction(
                features={"f0": float(X[i, 0]), "f1": float(X[i, 1])},
                prediction=int(y[i]),
            )

        # No model set → no recomputation even if interval elapsed
        client._last_importance_calc = datetime.now(timezone.utc) - timedelta(days=365)
        old_time = client._last_importance_calc

        client.get_feature_health()
        assert client._last_importance_calc == old_time

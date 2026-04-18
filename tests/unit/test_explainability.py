"""Tests for ExplainabilityEngine — row, global, and cohort explanations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentinel.intelligence.explainability import ExplainabilityEngine

# ── Helpers ────────────────────────────────────────────────────────


class _FakeModel:
    """Minimal model that returns sum of features as prediction."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X, axis=1)


@pytest.fixture()
def model() -> _FakeModel:
    return _FakeModel()


@pytest.fixture()
def features() -> list[str]:
    return ["a", "b", "c"]


@pytest.fixture()
def engine(model: _FakeModel, features: list[str]) -> ExplainabilityEngine:
    """Engine without SHAP (falls back to permutation)."""
    with patch.dict("sys.modules", {"shap": None}):
        return ExplainabilityEngine(model, feature_names=features, method="auto")


@pytest.fixture()
def X() -> np.ndarray:
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


# ── Row-level explain() ───────────────────────────────────────────


class TestExplainRow:
    def test_returns_list_of_dicts(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        result = engine.explain(X)
        assert isinstance(result, list)
        assert len(result) == 3
        for row in result:
            assert set(row.keys()) == {"a", "b", "c"}

    def test_single_row(self, engine: ExplainabilityEngine) -> None:
        result = engine.explain(np.array([[1.0, 2.0, 3.0]]))
        assert len(result) == 1

    def test_1d_input_reshaped(self, engine: ExplainabilityEngine) -> None:
        result = engine.explain(np.array([1.0, 2.0, 3.0]))
        assert len(result) == 1

    def test_explain_one(self, engine: ExplainabilityEngine) -> None:
        result = engine.explain_one(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, dict)
        assert "a" in result

    def test_top_features(self, engine: ExplainabilityEngine) -> None:
        result = engine.top_features(np.array([1.0, 2.0, 3.0]), n=2)
        assert len(result) == 2
        assert all(isinstance(t, tuple) for t in result)


# ── Global explain_global() ───────────────────────────────────────


class TestExplainGlobal:
    def test_returns_sorted_dict(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        result = engine.explain_global(X)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b", "c"}
        # Values should be sorted descending
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_all_non_negative(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        result = engine.explain_global(X)
        assert all(v >= 0 for v in result.values())

    def test_empty_input(self, engine: ExplainabilityEngine) -> None:
        result = engine.explain_global(np.zeros((0, 3)))
        assert all(v == 0.0 for v in result.values())

    def test_single_row(self, engine: ExplainabilityEngine) -> None:
        result = engine.explain_global(np.array([[1.0, 0.0, 0.0]]))
        assert isinstance(result, dict)
        assert len(result) == 3


# ── Cohort explain_cohorts() ──────────────────────────────────────


class TestExplainCohorts:
    def test_returns_per_cohort_dicts(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        labels = ["a_group", "a_group", "b_group"]
        result = engine.explain_cohorts(X, labels)
        assert set(result.keys()) == {"a_group", "b_group"}
        assert set(result["a_group"].keys()) == {"a", "b", "c"}

    def test_single_cohort(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        labels = ["all", "all", "all"]
        result = engine.explain_cohorts(X, labels)
        assert len(result) == 1
        assert "all" in result

    def test_mismatched_labels_raises(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        with pytest.raises(ValueError, match="cohort_labels length"):
            engine.explain_cohorts(X, ["only_two", "labels"])

    def test_all_values_non_negative(self, engine: ExplainabilityEngine, X: np.ndarray) -> None:
        labels = ["x", "y", "x"]
        result = engine.explain_cohorts(X, labels)
        for cohort_vals in result.values():
            assert all(v >= 0 for v in cohort_vals.values())

    def test_cohort_means_differ(self, engine: ExplainabilityEngine) -> None:
        # Two very different cohorts
        X = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
        labels = ["high_a", "high_a", "high_c", "high_c"]
        result = engine.explain_cohorts(X, labels)
        assert result["high_a"]["a"] != result["high_c"]["a"]


# ── Permutation fallback ──────────────────────────────────────────


class TestPermutationFallback:
    def test_broken_predict_returns_zeros(self, features: list[str]) -> None:
        model = MagicMock()
        model.predict.side_effect = RuntimeError("broken")
        with patch.dict("sys.modules", {"shap": None}):
            eng = ExplainabilityEngine(model, feature_names=features)
        result = eng.explain(np.array([[1.0, 2.0, 3.0]]))
        assert result == [{"a": 0.0, "b": 0.0, "c": 0.0}]

    def test_no_shap_uses_permutation(self, engine: ExplainabilityEngine) -> None:
        assert engine._explainer is None  # no SHAP loaded
        result = engine.explain(np.array([[1.0, 2.0, 3.0]]))
        assert isinstance(result, list)

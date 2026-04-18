"""SHAP/LIME wrapper providing per-prediction explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class ExplainabilityEngine:
    """Wraps SHAP/LIME with a uniform API and graceful fallbacks.

    The engine prefers SHAP TreeExplainer (fast for tree models) and falls
    back to KernelExplainer or a simple permutation-based approximation
    when SHAP is not installed.

    Example:
        >>> engine = ExplainabilityEngine(model, feature_names=["age", "amount"])
        >>> engine.explain(X[0:1])  # → {"age": 0.12, "amount": -0.04}
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: Any | None = None,
        method: str = "auto",
    ):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.method = method
        self._explainer: Any | None = None
        self._method_used: str = "unknown"
        self._init_explainer()

    def _init_explainer(self) -> None:
        try:
            import shap  # type: ignore[import-not-found]
        except ImportError:
            log.warning("explainability.shap_not_installed", message="falling back to permutation")
            self._explainer = None
            return

        try:
            if hasattr(self.model, "estimators_") or hasattr(self.model, "tree_"):
                self._explainer = shap.TreeExplainer(self.model)
            elif self.background_data is not None:
                self._explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            else:
                self._explainer = None
        except Exception as e:
            log.warning("explainability.init_failed", error=str(e))
            self._explainer = None

    # ── Public API ────────────────────────────────────────────────

    def explain(self, X: Any) -> list[dict[str, float]]:
        """Compute per-feature attributions for each row in X."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(X_arr)
                if isinstance(shap_values, list):
                    # Multi-class — sum across classes for an aggregate
                    shap_values = np.sum(np.abs(np.array(shap_values)), axis=0)
                shap_values = np.asarray(shap_values)
                self._method_used = "shap"
                return [
                    dict(zip(self.feature_names, row.tolist(), strict=False)) for row in shap_values
                ]
            except Exception as e:
                log.warning("explainability.shap_failed", error=str(e))

        self._method_used = "permutation"
        return self._permutation_explanations(X_arr)

    def explain_one(self, x: Any) -> dict[str, float]:
        """Convenience: explain a single prediction."""
        return self.explain(np.asarray(x).reshape(1, -1))[0]

    @property
    def method_used(self) -> str:
        """Return the method actually used by the last ``explain()`` call."""
        return self._method_used

    def top_features(self, x: Any, n: int = 5) -> list[tuple[str, float]]:
        """Return the top-N features by absolute attribution."""
        attributions = self.explain_one(x)
        ranked = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return ranked[:n]

    def explain_global(self, X: Any) -> dict[str, float]:
        """Compute global feature importance as mean |attribution| across all rows.

        Args:
            X: Feature matrix (n_samples x n_features).

        Returns:
            Dict mapping feature name → mean absolute attribution, sorted
            descending by importance.
        """
        attributions = self.explain(X)
        if not attributions:
            return dict.fromkeys(self.feature_names, 0.0)

        sums: dict[str, float] = dict.fromkeys(self.feature_names, 0.0)
        for row in attributions:
            for name, value in row.items():
                if isinstance(value, (list, tuple)):
                    sums[name] = sums.get(name, 0.0) + sum(abs(v) for v in value)
                else:
                    sums[name] = sums.get(name, 0.0) + abs(value)
        n = len(attributions)
        means = {name: sums[name] / n for name in self.feature_names}
        return dict(sorted(means.items(), key=lambda kv: kv[1], reverse=True))

    def explain_cohorts(
        self,
        X: Any,
        cohort_labels: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute per-cohort mean |attribution| for cross-cohort comparison.

        Args:
            X: Feature matrix (n_samples x n_features).
            cohort_labels: Length-n list assigning each row to a cohort.

        Returns:
            Dict mapping cohort_id → {feature → mean |attribution|}.

        Raises:
            ValueError: If ``len(cohort_labels) != X.shape[0]``.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if len(cohort_labels) != X_arr.shape[0]:
            msg = (
                f"cohort_labels length ({len(cohort_labels)}) must match X rows ({X_arr.shape[0]})"
            )
            raise ValueError(msg)

        attributions = self.explain(X_arr)

        # Bucket by cohort
        cohort_sums: dict[str, dict[str, float]] = {}
        cohort_counts: dict[str, int] = {}
        for label, row_attr in zip(cohort_labels, attributions, strict=False):
            if label not in cohort_sums:
                cohort_sums[label] = dict.fromkeys(self.feature_names, 0.0)
                cohort_counts[label] = 0
            cohort_counts[label] += 1
            for name, value in row_attr.items():
                if isinstance(value, (list, tuple)):
                    cohort_sums[label][name] += sum(abs(v) for v in value)
                else:
                    cohort_sums[label][name] += abs(value)

        result: dict[str, dict[str, float]] = {}
        for label in cohort_sums:
            n = cohort_counts[label]
            result[label] = {name: cohort_sums[label][name] / n for name in self.feature_names}
        return result

    # ── Permutation fallback ──────────────────────────────────────

    def _permutation_explanations(self, X: np.ndarray) -> list[dict[str, float]]:
        """Crude permutation importance per row — no SHAP required."""
        try:
            base_pred = self.model.predict(X).astype(float).ravel()
        except Exception:
            return [dict.fromkeys(self.feature_names, 0.0) for _ in range(X.shape[0])]

        explanations: list[dict[str, float]] = []
        for i in range(X.shape[0]):
            row = X[i].copy()
            attr: dict[str, float] = {}
            for j, name in enumerate(self.feature_names):
                perturbed = row.copy()
                perturbed[j] = 0.0
                try:
                    delta = float(base_pred[i]) - float(
                        self.model.predict(perturbed.reshape(1, -1)).ravel()[0]
                    )
                except Exception:
                    delta = 0.0
                attr[name] = delta
            explanations.append(attr)
        return explanations

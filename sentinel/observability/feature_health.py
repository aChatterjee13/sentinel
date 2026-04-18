"""Per-feature health monitoring with importance ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from sentinel.config.schema import FeatureHealthConfig
from sentinel.core.types import AlertSeverity, FeatureHealth, FeatureHealthReport
from sentinel.observability.drift.base import BaseDriftDetector

log = structlog.get_logger(__name__)


class FeatureHealthMonitor:
    """Combines drift scores with feature importance to prioritise alerts.

    The monitor consumes a drift report from any `BaseDriftDetector` and
    enriches it with importance and null-rate metadata.
    """

    def __init__(
        self,
        config: FeatureHealthConfig,
        model_name: str,
        importances: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.model_name = model_name
        self._importances: dict[str, float] = importances or {}
        self._max_importance_entries: int = 100

    def set_importances(self, importances: dict[str, float]) -> None:
        """Update feature importances (called by SentinelClient on a schedule)."""
        self._importances = dict(importances)
        # Enforce max size — keep the most important features
        if len(self._importances) > self._max_importance_entries:
            ranked = sorted(self._importances.items(), key=lambda kv: kv[1], reverse=True)
            self._importances = dict(ranked[: self._max_importance_entries])

    def compute_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute feature importances using the configured method.

        Checks ``self.config.importance_method`` and dispatches to the
        appropriate algorithm:

        * ``"builtin"`` — extracts ``feature_importances_`` or ``coef_``
          from the model.
        * ``"shap"`` — lazy-imports ``shap``, computes mean absolute SHAP
          values, normalises to sum to 1.0.  Falls back to ``"builtin"``
          with a warning when ``shap`` is not installed.
        * ``"permutation"`` — lazy-imports ``sklearn``, runs
          ``permutation_importance``, normalises.  Requires *y*; falls
          back to ``"builtin"`` with a warning when *y* is ``None`` or
          ``sklearn`` is not installed.

        The result is stored via :meth:`set_importances` so subsequent
        calls to :meth:`evaluate` use the updated ranking.

        Args:
            model: A fitted estimator (any object with a ``predict`` or
                ``predict_proba`` method).
            X: Feature matrix (n_samples x n_features).
            y: Target vector.  Required for ``"permutation"`` method.
            feature_names: Feature name list.  When ``None``, names are
                generated as ``f0, f1, …``.

        Returns:
            Dict mapping feature name → normalised importance (sums to 1.0).

        Example:
            >>> monitor.compute_importance(clf, X_train, y_train)
            {'age': 0.35, 'income': 0.45, 'tenure': 0.20}
        """
        n_features = X.shape[1] if X.ndim > 1 else 1
        names = feature_names or [f"f{i}" for i in range(n_features)]
        method = self.config.importance_method

        if method == "shap":
            result = self._compute_shap_importance(model, X, names)
        elif method == "permutation":
            result = self._compute_permutation_importance(model, X, y, names)
        else:
            result = self.compute_builtin_importance(model, names)

        self.set_importances(result)
        return result

    def _compute_shap_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Compute SHAP-based importance, falling back to builtin on error."""
        try:
            import shap
        except ImportError:
            log.warning(
                "feature_health.shap_not_installed",
                fallback="builtin",
                hint="pip install shap",
            )
            return self.compute_builtin_importance(model, feature_names)

        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            vals = np.abs(shap_values.values)
            if vals.ndim == 3:
                # Multi-output: average across outputs
                vals = vals.mean(axis=2)
            mean_abs = vals.mean(axis=0)
            total = mean_abs.sum()
            if total < 1e-9:
                return dict(zip(feature_names, [0.0] * len(feature_names), strict=False))
            normalised = mean_abs / total
            return dict(zip(feature_names, normalised.tolist(), strict=False))
        except Exception:
            log.warning(
                "feature_health.shap_computation_failed",
                fallback="builtin",
                exc_info=True,
            )
            return self.compute_builtin_importance(model, feature_names)

    def _compute_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray | None,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Compute permutation importance, falling back to builtin on error."""
        if y is None:
            log.warning(
                "feature_health.permutation_requires_y",
                fallback="builtin",
            )
            return self.compute_builtin_importance(model, feature_names)

        try:
            from sklearn.inspection import permutation_importance as _perm_imp
        except ImportError:
            log.warning(
                "feature_health.sklearn_not_installed",
                fallback="builtin",
                hint="pip install scikit-learn",
            )
            return self.compute_builtin_importance(model, feature_names)

        try:
            result = _perm_imp(model, X, y, n_repeats=10, random_state=42)
            raw = np.asarray(result.importances_mean, dtype=float)
            raw = np.clip(raw, 0.0, None)
            total = raw.sum()
            if total < 1e-9:
                return dict(zip(feature_names, [0.0] * len(feature_names), strict=False))
            normalised = raw / total
            return dict(zip(feature_names, normalised.tolist(), strict=False))
        except Exception:
            log.warning(
                "feature_health.permutation_computation_failed",
                fallback="builtin",
                exc_info=True,
            )
            return self.compute_builtin_importance(model, feature_names)

    def evaluate(
        self,
        detector: BaseDriftDetector,
        current: Any,
        null_rates: dict[str, float] | None = None,
    ) -> FeatureHealthReport:
        """Run drift detection and produce a feature-level health report."""
        report = detector.detect(current)
        null_rates = null_rates or {}
        feats: list[FeatureHealth] = []
        for name, score in report.feature_scores.items():
            importance = self._importances.get(name, 0.0)
            null_rate = null_rates.get(name, 0.0)
            is_drifted = name in report.drifted_features
            severity = self._severity(is_drifted, importance)
            feats.append(
                FeatureHealth(
                    name=name,
                    importance=importance,
                    drift_score=float(score),
                    null_rate=float(null_rate),
                    is_drifted=is_drifted,
                    severity=severity,
                )
            )

        # Top-N drifted features by drift_score * importance (weighted importance)
        ranked = sorted(
            feats,
            key=lambda f: f.drift_score * (f.importance + 1e-6),
            reverse=True,
        )
        top_n = [f.name for f in ranked[: self.config.alert_on_top_n_drift] if f.is_drifted]

        return FeatureHealthReport(
            model_name=self.model_name,
            features=feats,
            top_n_drifted=top_n,
        )

    def _severity(self, is_drifted: bool, importance: float) -> AlertSeverity:
        if not is_drifted:
            return AlertSeverity.INFO
        if importance > 0.5:
            return AlertSeverity.CRITICAL
        if importance > 0.2:
            return AlertSeverity.HIGH
        return AlertSeverity.WARNING

    @staticmethod
    def compute_null_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
        """Helper: compute per-key null rates from a row list."""
        if not rows:
            return {}
        keys = {k for r in rows for k in r}
        n = len(rows)
        return {k: sum(1 for r in rows if r.get(k) is None) / n for k in keys}

    @staticmethod
    def compute_builtin_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
        """Pull `feature_importances_` or `coef_` from a fitted estimator."""
        importances: np.ndarray | None = None
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importances = np.abs(coef).ravel() if coef.ndim > 1 else np.abs(coef)
        if importances is None:
            return {}
        importances = importances / max(importances.sum(), 1e-9)
        return dict(zip(feature_names, importances.tolist(), strict=False))

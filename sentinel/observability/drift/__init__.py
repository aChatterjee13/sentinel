"""Drift detection — data, concept, and model drift.

The package exposes a registry-based factory so configs like
``method: psi`` resolve to the right implementation at runtime.
"""

from __future__ import annotations

from typing import Any

from sentinel.observability.drift.base import BaseDriftDetector
from sentinel.observability.drift.concept_drift import (
    ADWINConceptDriftDetector,
    DDMConceptDriftDetector,
    EDDMConceptDriftDetector,
    PageHinkleyDriftDetector,
)
from sentinel.observability.drift.data_drift import (
    ChiSquaredDriftDetector,
    JSDivergenceDriftDetector,
    KSDriftDetector,
    PSIDriftDetector,
    WassersteinDriftDetector,
)
from sentinel.observability.drift.model_drift import ModelPerformanceDriftDetector

# ── Registry ───────────────────────────────────────────────────────

_DATA_DRIFT_REGISTRY: dict[str, type[BaseDriftDetector]] = {
    "psi": PSIDriftDetector,
    "ks": KSDriftDetector,
    "js_divergence": JSDivergenceDriftDetector,
    "chi_squared": ChiSquaredDriftDetector,
    "wasserstein": WassersteinDriftDetector,
}

_CONCEPT_DRIFT_REGISTRY: dict[str, type[BaseDriftDetector]] = {
    "ddm": DDMConceptDriftDetector,
    "eddm": EDDMConceptDriftDetector,
    "adwin": ADWINConceptDriftDetector,
    "page_hinkley": PageHinkleyDriftDetector,
}


def create_drift_detector(
    method: str,
    model_name: str,
    threshold: float = 0.2,
    **kwargs: Any,
) -> BaseDriftDetector:
    """Factory: build a drift detector by method name."""
    if method in _DATA_DRIFT_REGISTRY:
        cls = _DATA_DRIFT_REGISTRY[method]
    elif method in _CONCEPT_DRIFT_REGISTRY:
        cls = _CONCEPT_DRIFT_REGISTRY[method]
    else:
        raise ValueError(
            f"unknown drift method '{method}'. "
            f"Available: {sorted(_DATA_DRIFT_REGISTRY) + sorted(_CONCEPT_DRIFT_REGISTRY)}"
        )
    return cls(model_name=model_name, threshold=threshold, **kwargs)


def register_data_drift_detector(name: str, cls: type[BaseDriftDetector]) -> None:
    """Plug-in API: register a custom data drift detector."""
    _DATA_DRIFT_REGISTRY[name] = cls


def register_concept_drift_detector(name: str, cls: type[BaseDriftDetector]) -> None:
    """Plug-in API: register a custom concept drift detector."""
    _CONCEPT_DRIFT_REGISTRY[name] = cls


__all__ = [
    "ADWINConceptDriftDetector",
    "BaseDriftDetector",
    "ChiSquaredDriftDetector",
    "DDMConceptDriftDetector",
    "EDDMConceptDriftDetector",
    "JSDivergenceDriftDetector",
    "KSDriftDetector",
    "ModelPerformanceDriftDetector",
    "PSIDriftDetector",
    "PageHinkleyDriftDetector",
    "WassersteinDriftDetector",
    "create_drift_detector",
    "register_concept_drift_detector",
    "register_data_drift_detector",
]

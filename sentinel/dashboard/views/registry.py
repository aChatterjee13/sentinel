"""Model registry view — list models, versions, and detail."""

from __future__ import annotations

from typing import Any

from sentinel.core.exceptions import RegistryError
from sentinel.dashboard.state import DashboardState


def list_models(state: DashboardState) -> dict[str, Any]:
    """Build the registry list page payload."""
    client = state.client
    models: list[dict[str, Any]] = []
    try:
        names = client.registry.list_models()
    except Exception:
        names = []
    for name in names:
        try:
            versions = client.registry.list_versions(name)
        except Exception:
            versions = []
        models.append(
            {
                "name": name,
                "versions": versions,
                "latest": versions[-1] if versions else None,
            }
        )
    return {"models": models, "active_model": client.model_name}


def detail(state: DashboardState, model: str, version: str) -> dict[str, Any] | None:
    """Build the per-version detail payload."""
    client = state.client
    try:
        mv = client.registry.get(model, version)
    except RegistryError:
        return None
    baseline: dict[str, Any] | None
    try:
        baseline = client.registry.get_baseline(model, version)
    except Exception:
        baseline = None

    sibling_versions = client.registry.list_versions(model)
    return {
        "model": mv.model_dump(mode="json"),
        "baseline": baseline,
        "sibling_versions": sibling_versions,
    }


def compare(state: DashboardState, model: str, v1: str, v2: str) -> dict[str, Any] | None:
    """Compare two versions of the same model."""
    client = state.client
    try:
        return client.registry.compare(model, v1, v2)
    except RegistryError:
        return None

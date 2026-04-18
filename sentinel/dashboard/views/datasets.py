"""Dataset registry view — list datasets, versions, and detail."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def list_datasets(state: DashboardState) -> dict[str, Any]:
    """Build the dataset list page payload.

    Args:
        state: Current dashboard state.

    Returns:
        Dict with ``datasets`` list (grouped by name) and ``total`` count.
    """
    client = state.client
    datasets: list[dict[str, Any]] = []
    try:
        names = client.datasets.list_names()
    except Exception:
        names = []
    total = 0
    for name in names:
        try:
            versions = client.datasets.list_versions(name)
        except Exception:
            versions = []
        version_summaries = [
            {
                "version": v.version,
                "format": v.format,
                "split": v.split,
                "num_rows": v.num_rows,
                "created_at": v.created_at,
            }
            for v in versions
        ]
        total += len(versions)
        datasets.append(
            {
                "name": name,
                "versions": version_summaries,
                "latest": version_summaries[-1] if version_summaries else None,
                "count": len(version_summaries),
            }
        )
    return {"datasets": datasets, "total": total}


def detail(state: DashboardState, name: str, version: str) -> dict[str, Any] | None:
    """Build the per-version detail payload.

    Args:
        state: Current dashboard state.
        name: Dataset name.
        version: Version string.

    Returns:
        Dict with dataset detail and sibling versions, or ``None`` if
        the version is not found.
    """
    client = state.client
    try:
        ds = client.datasets.get(name, version)
    except KeyError:
        return None

    sibling_versions = [v.version for v in client.datasets.list_versions(name)]

    return {
        "dataset": ds.model_dump(mode="json", by_alias=True),
        "sibling_versions": sibling_versions,
    }

"""Lineage helpers for dataset version graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from sentinel.foundation.datasets.registry import DatasetRegistry

log = structlog.get_logger(__name__)


def resolve_dataset_ref(ref: str) -> tuple[str, str]:
    """Parse a ``'name@version'`` reference into *(name, version)*.

    Args:
        ref: Dataset reference string.

    Returns:
        Tuple of *(name, version)*.

    Raises:
        ValueError: If *ref* does not contain ``@``.

    Example:
        >>> resolve_dataset_ref("train_data@1.0")
        ('train_data', '1.0')
    """
    if "@" not in ref:
        raise ValueError(f"Dataset reference must use 'name@version' format, got: '{ref}'")
    name, version = ref.rsplit("@", 1)
    if not name or not version:
        raise ValueError(f"Dataset reference must use 'name@version' format, got: '{ref}'")
    return name, version


def get_lineage_graph(registry: DatasetRegistry, name: str, version: str) -> dict[str, Any]:
    """Build a lineage graph for a dataset version.

    Follows the ``derived_from`` chain up to the root ancestor and
    collects linked experiments and models at each step.

    Args:
        registry: The dataset registry to query.
        name: Dataset name.
        version: Dataset version.

    Returns:
        Dict with keys ``dataset``, ``derived_from``, ``linked_experiments``,
        ``linked_models``, and ``ancestry`` (list of ``'name@version'`` refs).

    Raises:
        KeyError: If the starting dataset version is not found.

    Example:
        >>> graph = get_lineage_graph(registry, "features", "2.0")
        >>> graph["ancestry"]
        ['features@1.0', 'raw_data@1.0']
    """
    ds = registry.get(name, version)
    graph: dict[str, Any] = {
        "dataset": f"{name}@{version}",
        "derived_from": ds.derived_from,
        "linked_experiments": list(ds.linked_experiments),
        "linked_models": list(ds.linked_models),
    }

    chain: list[str] = []
    current = ds.derived_from
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        chain.append(current)
        try:
            parent_name, parent_ver = resolve_dataset_ref(current)
            parent = registry.get(parent_name, parent_ver)
            current = parent.derived_from
        except (KeyError, ValueError):
            break

    graph["ancestry"] = chain
    return graph

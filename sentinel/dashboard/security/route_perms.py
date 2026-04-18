"""Route → permission lookup table.

Each dashboard URL group maps to a single namespaced permission. The
table is consulted by :func:`build_pages_router` /
:func:`build_api_router` so individual route handlers stay readable
and the routing → authorisation surface is reviewable in one place.

Adding a new route is a two-step process:

1. Add the route to ``pages.py`` or ``api.py`` with
   ``Depends(require_permission(permission_for_path("/your/path")))``.
2. Add the path → permission entry to :data:`_RULES` below.

The matcher is path-prefix based and longest-match wins, so
``/api/audit/stream`` correctly resolves to the same ``audit.read``
permission as ``/api/audit``.
"""

from __future__ import annotations

# Ordered most-specific → least-specific. The matcher walks this list
# in order and returns the first prefix match, so paths must be sorted
# longest first when they share a common ancestor.
_RULES: list[tuple[str, str]] = [
    # Auth-free routes (the auth middleware still gates these,
    # but no permission is required to load them).
    ("/api/health/live", ""),
    ("/api/health", ""),
    # Exports
    ("/api/export/audit.csv", "audit.read"),
    ("/api/export/drift.csv", "drift.read"),
    ("/api/export/metrics.csv", "registry.read"),
    # Drift
    ("/api/drift", "drift.read"),
    ("/drift", "drift.read"),
    # Features
    ("/api/features", "features.read"),
    ("/features", "features.read"),
    # Registry
    ("/api/registry", "registry.read"),
    ("/registry", "registry.read"),
    # Audit
    ("/api/audit", "audit.read"),
    ("/audit", "audit.read"),
    # LLMOps + token economics
    ("/api/llmops", "llmops.read"),
    ("/api/tokens", "llmops.read"),
    ("/llmops", "llmops.read"),
    # AgentOps + traces
    ("/api/agentops", "agentops.read"),
    ("/api/traces", "agentops.read"),
    ("/agentops", "agentops.read"),
    # Deployments
    ("/api/deployments/rollback", "deployments.write"),
    ("/api/deployments", "deployments.read"),
    ("/deployments", "deployments.read"),
    # Compliance
    ("/api/compliance", "compliance.read"),
    ("/compliance", "compliance.read"),
    # Overview last so it doesn't accidentally claim sub-paths.
    ("/api/overview", ""),
    ("/", ""),
]


def permission_for_path(path: str) -> str:
    """Return the permission required to access ``path``.

    The empty string means "no permission needed beyond
    authentication" and is the default when the path is not in the
    table — useful for one-off pages like ``/`` or ``/api/health``.
    """
    for prefix, permission in _RULES:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix):
            # Be strict about ``/`` so it doesn't match every path.
            if prefix == "/" and path != "/":
                continue
            return permission
    return ""

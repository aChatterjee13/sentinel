"""Pure view functions for the dashboard.

Each view function takes a :class:`DashboardState` (or just a
:class:`SentinelClient`) and returns a JSON-serialisable dict ready to be
fed into a Jinja2 template or returned from an ``/api/*`` endpoint. Views
own no FastAPI types so they can be unit-tested directly.
"""

from sentinel.dashboard.views import (
    agentops,
    audit,
    compliance,
    datasets,
    deployments,
    drift,
    features,
    llmops,
    overview,
    registry,
)

__all__ = [
    "agentops",
    "audit",
    "compliance",
    "datasets",
    "deployments",
    "drift",
    "features",
    "llmops",
    "overview",
    "registry",
]

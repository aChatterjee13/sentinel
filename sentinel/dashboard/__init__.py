"""Sentinel dashboard — local-first FastAPI UI over the SentinelClient.

This module is an optional extra. Install with:

    pip install sentinel-mlops[dashboard]

The dashboard is a single FastAPI application that reads everything from a
live :class:`~sentinel.SentinelClient` — there is no separate data store. It
ships in three flavours that all share the same code:

* **Local dev dashboard** — `sentinel dashboard --config sentinel.yaml`
* **Embeddable router** — :class:`SentinelDashboardRouter` returns a FastAPI
  ``APIRouter`` that customers can mount under their own apps.
* **Hosted (future)** — out of scope, but the :func:`create_dashboard_app`
  factory makes it straightforward to wrap.

Example:
    >>> from sentinel import SentinelClient
    >>> from sentinel.dashboard import create_dashboard_app
    >>> client = SentinelClient.from_config("sentinel.yaml")
    >>> app = create_dashboard_app(client)
    >>> # uvicorn.run(app, host="127.0.0.1", port=8000)
"""

from __future__ import annotations

from sentinel.dashboard.server import (
    SentinelDashboardRouter,
    create_dashboard_app,
    run,
)

__all__ = [
    "SentinelDashboardRouter",
    "create_dashboard_app",
    "run",
]

"""FastAPI dependency providers for the dashboard.

Routes call :func:`get_state` (or :func:`get_client`) via ``Depends`` to pull
the live :class:`DashboardState` out of the FastAPI app. The state is set on
``app.state.dashboard_state`` by :func:`create_dashboard_app`.
"""

from typing import TYPE_CHECKING

from fastapi import Request

from sentinel.core.exceptions import DashboardError
from sentinel.dashboard.state import DashboardState

if TYPE_CHECKING:
    from sentinel.core.client import SentinelClient


def get_state(request: Request) -> DashboardState:
    """Return the :class:`DashboardState` bound to the running app."""
    state = getattr(request.app.state, "dashboard_state", None)
    if state is None:
        raise DashboardError("dashboard state not initialised on app")
    return state


def get_client(request: Request) -> "SentinelClient":
    """Return the live :class:`SentinelClient` for the running app."""
    return get_state(request).client

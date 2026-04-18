"""FastAPI route modules for the dashboard."""

from sentinel.dashboard.routes.api import build_api_router
from sentinel.dashboard.routes.pages import build_pages_router

__all__ = ["build_api_router", "build_pages_router"]

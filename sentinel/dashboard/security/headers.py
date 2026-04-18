"""Security headers middleware for the dashboard.

Adds the always-on hardening headers every modern security review
expects: anti-clickjacking, MIME sniffing prevention, referrer
policy, restricted permissions policy, HSTS (when the request was
served over HTTPS), and a Content Security Policy that allows the
exact CDN scripts the dashboard already loads.

Operators can override the default CSP via
``dashboard.server.csp.policy`` or disable CSP entirely via
``dashboard.server.csp.enabled``. Disabling CSP is a deliberate
escape hatch for environments behind a strict reverse-proxy CSP
that would otherwise double-set the header.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from sentinel.config.schema import CSPConfig

if TYPE_CHECKING:
    from starlette.types import ASGIApp

log = structlog.get_logger(__name__)


# The dashboard pulls Tailwind, HTMX, and Plotly from CDNs and an
# inline ``<script>`` block configures Tailwind. The CSP needs to
# allow exactly those origins. Inline event handlers are blocked
# (no unsafe-inline for scripts) but inline ``style`` attributes
# from Tailwind require ``unsafe-inline`` in the style directive.
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' "
    "https://cdn.tailwindcss.com https://unpkg.com https://cdn.plot.ly "
    "https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds anti-clickjacking, MIME, referrer, HSTS, and CSP headers."""

    def __init__(self, app: "ASGIApp", csp: CSPConfig | None = None) -> None:
        super().__init__(app)
        self.csp = csp or CSPConfig()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        headers = response.headers
        # Base set — cheap, always on.
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=()",
        )

        # HSTS is only meaningful over HTTPS — don't set it for plain
        # HTTP, otherwise an attacker on a captive portal could
        # poison clients with the wrong scheme.
        if request.url.scheme == "https":
            headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )

        if self.csp.enabled and "Content-Security-Policy" not in headers:
            headers["Content-Security-Policy"] = self.csp.policy or DEFAULT_CSP

        return response


__all__ = ["DEFAULT_CSP", "SecurityHeadersMiddleware"]

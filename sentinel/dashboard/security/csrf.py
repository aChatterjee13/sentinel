"""Double-submit cookie CSRF protection for the dashboard.

The dashboard is read-only today (workstream #6 will add write
endpoints), but designing CSRF in advance keeps the security surface
reviewable in one place and means WS-6 ships safely without an
auth-stack rewrite.

Strategy: classic double-submit cookie. The middleware sets a
cookie containing a random token on every response (or reuses the
existing one), then verifies on unsafe HTTP verbs that the same
token also appears in a request header (default ``X-CSRF-Token``).
The same-origin policy prevents an attacker page from reading the
cookie value, so even though the token isn't bound to a server
session it's effectively unforgeable from another origin.

Bearer-token requests are exempt: the bearer token is itself the
auth credential and there is no cookie for an attacker to forge a
companion submit for. Static asset requests are exempt because they
never carry session state.
"""

import hmac
import secrets
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from sentinel.config.schema import CSRFConfig

if TYPE_CHECKING:
    from starlette.types import ASGIApp

log = structlog.get_logger(__name__)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_TOKEN_BYTES = 32


def _new_token() -> str:
    """Return a fresh URL-safe random CSRF token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def _is_static(path: str) -> bool:
    return path.startswith("/static/")


def _is_bearer(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    return auth.lower().startswith("bearer ")


class CSRFMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces double-submit CSRF tokens."""

    def __init__(self, app: "ASGIApp", cfg: CSRFConfig) -> None:
        super().__init__(app)
        self.cfg = cfg

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self.cfg.enabled:
            request.state.csrf_token = ""
            return await call_next(request)

        path = request.url.path
        method = request.method.upper()

        # Static assets are pure read traffic — never gated.
        if _is_static(path):
            return await call_next(request)

        # Bearer-token requests are exempt: the bearer token IS the
        # credential and is not stored in a cookie an attacker can
        # forge a companion submit for.
        bearer = _is_bearer(request)

        cookie_token = request.cookies.get(self.cfg.cookie_name)

        if method not in _SAFE_METHODS and not bearer:
            header_token = request.headers.get(self.cfg.header_name)
            if not cookie_token or not header_token:
                log.warning(
                    "dashboard.csrf.missing",
                    path=path,
                    method=method,
                    has_cookie=bool(cookie_token),
                    has_header=bool(header_token),
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "CSRF token missing"},
                )
            if not hmac.compare_digest(
                header_token.encode("utf-8"),
                cookie_token.encode("utf-8"),
            ):
                log.warning("dashboard.csrf.mismatch", path=path, method=method)
                return JSONResponse(
                    status_code=403,
                    content={"error": "CSRF token mismatch"},
                )

        # Make sure every page render has a token to embed in the
        # meta tag. Reuse the existing cookie when present so
        # already-issued tokens don't get invalidated mid-session.
        token = cookie_token or _new_token()
        request.state.csrf_token = token

        response = await call_next(request)

        if cookie_token is None:
            secure = self.cfg.cookie_secure
            if secure is None:
                secure = request.url.scheme == "https"
            response.set_cookie(
                key=self.cfg.cookie_name,
                value=token,
                httponly=True,
                secure=secure,
                samesite=self.cfg.cookie_samesite,
                path="/",
            )
        return response


__all__ = ["CSRFMiddleware"]

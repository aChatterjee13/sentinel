"""In-memory token-bucket rate limiting for the dashboard.

A deliberately tiny implementation: no slowapi, no Redis, no
abstraction over storage. The use case is single-worker production
deployments and the local-first dev experience, where we want to
catch a misconfigured client hammering ``/api/audit`` without adding
operational complexity.

.. warning::

   This limiter is **single-worker only**. In multi-worker deployments
   (e.g. ``uvicorn --workers 4``) each worker maintains its own bucket
   state, so effective limits are multiplied by the worker count.
   Use an external rate limiter (e.g. Redis + slowapi, or a reverse
   proxy like nginx ``limit_req``) for multi-worker deployments.

Three buckets per key:

* ``default`` — covers HTML pages
* ``api`` — covers ``/api/*``
* ``auth`` — covers basic auth attempts; tighter ceiling so brute
  force is impractical

Keys are derived from the request:

* When a principal with a non-anonymous username is on
  ``request.state``, the key is the username (per-user buckets).
* Otherwise we fall back to the client IP, accounting for
  ``X-Forwarded-For`` when set.

Buckets that haven't been touched in a generation are GC'd lazily
on every dispatch so memory stays bounded.
"""

import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from sentinel.config.schema import RateLimitConfig

if TYPE_CHECKING:
    from starlette.types import ASGIApp

log = structlog.get_logger(__name__)


@dataclass
class TokenBucket:
    """A classic token bucket: capacity, refill rate, last refill ts."""

    capacity: float
    refill_per_second: float
    tokens: float
    last_refill: float
    last_used: float

    def consume(self, now: float, amount: float = 1.0) -> bool:
        """Attempt to consume ``amount`` tokens. Returns True on success."""
        # Refill since last touch.
        elapsed = max(0.0, now - self.last_refill)
        if elapsed > 0.0:
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_per_second,
            )
            self.last_refill = now
        self.last_used = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self, amount: float = 1.0) -> float:
        """Return seconds until ``amount`` tokens become available."""
        if self.refill_per_second <= 0.0:
            return 60.0
        deficit = max(0.0, amount - self.tokens)
        return deficit / self.refill_per_second


class RateLimiter:
    """Thread-safe collection of token buckets keyed by ``(group, key)``.

    The limiter is constructed from a :class:`RateLimitConfig`. The
    three groups (default, api, auth) each have their own per-minute
    ceiling and a configurable burst multiplier that sets the
    bucket capacity.
    """

    def __init__(self, cfg: RateLimitConfig, *, time_fn: Callable[[], float] | None = None) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._time = time_fn or time.monotonic
        self._gc_after_seconds = 600.0  # 10 minutes idle → drop the bucket
        self._last_gc = self._time()

    def _limit_for(self, group: str) -> tuple[float, float]:
        if group == "api":
            per_minute = self.cfg.api_per_minute
        elif group == "auth":
            per_minute = self.cfg.auth_per_minute
        else:
            per_minute = self.cfg.default_per_minute
        capacity = per_minute * max(1.0, self.cfg.burst_multiplier)
        refill_per_second = per_minute / 60.0
        return capacity, refill_per_second

    def _get_or_create(self, group: str, key: str) -> TokenBucket:
        bucket = self._buckets.get((group, key))
        if bucket is not None:
            return bucket
        capacity, refill = self._limit_for(group)
        now = self._time()
        bucket = TokenBucket(
            capacity=capacity,
            refill_per_second=refill,
            tokens=capacity,
            last_refill=now,
            last_used=now,
        )
        self._buckets[(group, key)] = bucket
        return bucket

    def check(self, group: str, key: str) -> tuple[bool, float]:
        """Try to consume one token. Returns ``(allowed, retry_after_s)``."""
        with self._lock:
            self._maybe_gc()
            bucket = self._get_or_create(group, key)
            now = self._time()
            allowed = bucket.consume(now)
            retry = 0.0 if allowed else bucket.retry_after()
        return allowed, retry

    def _maybe_gc(self) -> None:
        now = self._time()
        if now - self._last_gc < 60.0:
            return
        cutoff = now - self._gc_after_seconds
        stale = [key for key, bucket in self._buckets.items() if bucket.last_used < cutoff]
        for key in stale:
            del self._buckets[key]
        self._last_gc = now

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def _client_key(request: Request) -> str:
    """Return per-user key when available, falling back to client IP."""
    principal = getattr(request.state, "principal", None)
    if principal is not None and not principal.is_anonymous:
        return f"user:{principal.username}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    if request.client is not None:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def _group_for(path: str, method: str) -> str:
    if path.startswith("/api/"):
        return "api"
    if method == "POST" and ("/login" in path or "/auth" in path):
        return "auth"
    return "default"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces per-key per-group rate limits."""

    def __init__(
        self,
        app: "ASGIApp",
        cfg: RateLimitConfig,
        limiter: RateLimiter | None = None,
    ) -> None:
        super().__init__(app)
        self.cfg = cfg
        self.limiter = limiter or RateLimiter(cfg)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self.cfg.enabled:
            return await call_next(request)

        path = request.url.path
        if path.startswith("/static/") or path == "/api/health" or path == "/api/health/live":
            return await call_next(request)

        group = _group_for(path, request.method.upper())
        key = _client_key(request)
        allowed, retry_after = self.limiter.check(group, key)
        if not allowed:
            log.warning(
                "dashboard.rate_limit.exceeded",
                path=path,
                key=key,
                group=group,
                retry_after_s=retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate limit exceeded",
                    "retry_after_seconds": round(retry_after, 2),
                },
                headers={"Retry-After": str(max(1, round(retry_after)))},
            )
        response = await call_next(request)
        # Retroactively apply auth rate limit on failed authentication
        # attempts to make brute-force impractical.
        if response.status_code in (401, 403):
            self.limiter.check("auth", key)
        return response


__all__ = ["RateLimitMiddleware", "RateLimiter", "TokenBucket"]

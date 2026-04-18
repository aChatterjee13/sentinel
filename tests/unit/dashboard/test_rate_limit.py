"""Tests for the in-memory token-bucket rate limiter."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    DashboardConfig,
    DashboardServerConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    RateLimitConfig,
    SentinelConfig,
)
from sentinel.dashboard.security.rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    TokenBucket,
)

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.dashboard.server import create_dashboard_app

# ── TokenBucket unit tests ───────────────────────────────────────────


class TestTokenBucket:
    def test_consume_within_capacity(self) -> None:
        bucket = TokenBucket(
            capacity=10.0,
            refill_per_second=1.0,
            tokens=10.0,
            last_refill=0.0,
            last_used=0.0,
        )
        for _ in range(10):
            assert bucket.consume(now=0.0)
        # 11th consume on the same timestamp drains.
        assert not bucket.consume(now=0.0)

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(
            capacity=10.0,
            refill_per_second=2.0,
            tokens=0.0,
            last_refill=0.0,
            last_used=0.0,
        )
        # 5 seconds elapsed → 10 tokens refilled (capped at capacity).
        assert bucket.consume(now=5.0)
        assert bucket.tokens == pytest.approx(9.0, abs=1e-9)

    def test_capacity_caps_refill(self) -> None:
        bucket = TokenBucket(
            capacity=10.0,
            refill_per_second=2.0,
            tokens=0.0,
            last_refill=0.0,
            last_used=0.0,
        )
        # 100 seconds elapsed → cap at capacity, not 200 tokens.
        bucket.consume(now=100.0)
        assert bucket.tokens <= bucket.capacity

    def test_retry_after_predicts_refill(self) -> None:
        bucket = TokenBucket(
            capacity=10.0,
            refill_per_second=1.0,
            tokens=0.0,
            last_refill=0.0,
            last_used=0.0,
        )
        # No tokens, refill rate 1/s → ~1s wait for one token.
        assert bucket.retry_after() == pytest.approx(1.0)


# ── RateLimiter unit tests ───────────────────────────────────────────


class TestRateLimiter:
    def test_per_minute_translates_to_capacity(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=60,
            api_per_minute=120,
            auth_per_minute=10,
            burst_multiplier=2.0,
        )
        # Use a controllable clock so refills are deterministic.
        now = [0.0]
        limiter = RateLimiter(cfg, time_fn=lambda: now[0])

        # 60 per minute, burst 2 → capacity 120.
        for _ in range(120):
            allowed, _ = limiter.check("default", "alice")
            assert allowed
        allowed, retry = limiter.check("default", "alice")
        assert not allowed
        assert retry > 0.0

    def test_separate_keys_have_independent_buckets(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=2,
            burst_multiplier=1.0,
        )
        now = [0.0]
        limiter = RateLimiter(cfg, time_fn=lambda: now[0])

        for _ in range(2):
            assert limiter.check("default", "alice")[0]
        assert not limiter.check("default", "alice")[0]
        # Different key still has full capacity.
        assert limiter.check("default", "bob")[0]

    def test_refill_after_wait(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=60,
            burst_multiplier=1.0,
        )
        now = [0.0]
        limiter = RateLimiter(cfg, time_fn=lambda: now[0])

        # Drain it.
        for _ in range(60):
            limiter.check("default", "alice")
        assert not limiter.check("default", "alice")[0]
        # 1 second later → 1 token refilled.
        now[0] = 1.0
        assert limiter.check("default", "alice")[0]

    def test_auth_group_uses_auth_per_minute(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=1000,
            api_per_minute=1000,
            auth_per_minute=2,
            burst_multiplier=1.0,
        )
        now = [0.0]
        limiter = RateLimiter(cfg, time_fn=lambda: now[0])
        assert limiter.check("auth", "alice")[0]
        assert limiter.check("auth", "alice")[0]
        assert not limiter.check("auth", "alice")[0]

    def test_reset_clears_all_buckets(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=1,
            burst_multiplier=1.0,
        )
        limiter = RateLimiter(cfg, time_fn=lambda: 0.0)
        limiter.check("default", "alice")
        assert not limiter.check("default", "alice")[0]
        limiter.reset()
        assert limiter.check("default", "alice")[0]

    def test_maybe_gc_evicts_idle_buckets(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=10,
            burst_multiplier=1.0,
        )
        now = [0.0]
        limiter = RateLimiter(cfg, time_fn=lambda: now[0])
        limiter.check("default", "alice")
        assert ("default", "alice") in limiter._buckets
        # Advance past the GC quiesce window AND past the idle threshold.
        now[0] = 5000.0
        limiter.check("default", "bob")
        # Alice's stale bucket should have been GC'd; Bob's is fresh.
        assert ("default", "alice") not in limiter._buckets
        assert ("default", "bob") in limiter._buckets


class TestTokenBucketRetryAfterEdgeCases:
    def test_retry_after_returns_60_when_refill_disabled(self) -> None:
        # refill_per_second == 0 → can never replenish; we report 60s.
        bucket = TokenBucket(
            capacity=10.0,
            refill_per_second=0.0,
            tokens=0.0,
            last_refill=0.0,
            last_used=0.0,
        )
        assert bucket.retry_after() == 60.0


class TestGroupRouting:
    def test_post_to_login_routes_to_auth_group(self) -> None:
        from sentinel.dashboard.security.rate_limit import _group_for

        assert _group_for("/login", "POST") == "auth"
        assert _group_for("/auth/token", "POST") == "auth"
        assert _group_for("/login", "GET") == "default"
        assert _group_for("/api/drift", "GET") == "api"
        assert _group_for("/", "GET") == "default"


# ── End-to-end through the FastAPI app ──────────────────────────────


def _build_minimal_app(cfg: RateLimitConfig) -> tuple[FastAPI, RateLimiter]:
    """Build a tiny FastAPI app with just the rate limit middleware.

    Returns the app and the limiter so the test can poke its clock.
    """
    now = [0.0]
    limiter = RateLimiter(cfg, time_fn=lambda: now[0])
    # Stash the time array on the limiter so callers can advance it.
    limiter.now = now  # type: ignore[attr-defined]

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, cfg=cfg, limiter=limiter)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"ok": "yes"}

    return app, limiter


class TestRateLimitMiddleware:
    def test_429_when_bucket_exhausted(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=2,
            burst_multiplier=1.0,
        )
        app, _ = _build_minimal_app(cfg)
        client = TestClient(app)
        assert client.get("/").status_code == 200
        assert client.get("/").status_code == 200
        resp = client.get("/")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_endpoint_is_exempt(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=1,
            api_per_minute=1,
            burst_multiplier=1.0,
        )
        app, _ = _build_minimal_app(cfg)
        client = TestClient(app)
        # /api/health is excluded so we should get many successes.
        for _ in range(20):
            assert client.get("/api/health").status_code == 200

    def test_disabled_middleware_passes_through(self) -> None:
        cfg = RateLimitConfig(
            enabled=False,
            default_per_minute=1,
            burst_multiplier=1.0,
        )
        app, _ = _build_minimal_app(cfg)
        client = TestClient(app)
        for _ in range(20):
            assert client.get("/").status_code == 200

    def test_authenticated_principal_keys_per_user(self, tmp_path: Path) -> None:
        """When a Principal is on request.state, the bucket key is per-user."""
        from sentinel.dashboard.security.principal import Principal

        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=1,
            burst_multiplier=1.0,
        )
        app, _ = _build_minimal_app(cfg)

        @app.middleware("http")
        async def _inject(request, call_next):  # type: ignore[no-untyped-def]
            request.state.principal = Principal(
                username="alice",
                roles=frozenset({"admin"}),
                permissions=frozenset({"*"}),
                auth_mode="basic",
            )
            return await call_next(request)

        client = TestClient(app)
        # Per-user bucket — alice gets one request before being throttled.
        assert client.get("/").status_code == 200
        assert client.get("/").status_code == 429

    def test_x_forwarded_for_used_when_available(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            default_per_minute=1,
            burst_multiplier=1.0,
        )
        app, _ = _build_minimal_app(cfg)
        client = TestClient(app)
        # First request from this forwarded IP exhausts the bucket.
        assert client.get("/", headers={"x-forwarded-for": "10.0.0.1"}).status_code == 200
        assert client.get("/", headers={"x-forwarded-for": "10.0.0.1"}).status_code == 429
        # A different forwarded IP has its own bucket.
        assert client.get("/", headers={"x-forwarded-for": "10.0.0.2"}).status_code == 200

    def test_dashboard_app_rate_limits_apply(self, tmp_path: Path) -> None:
        cfg = SentinelConfig(
            model=ModelConfig(name="rl_test_model", domain="tabular"),
            drift=DriftConfig(
                data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
            ),
            alerts=AlertsConfig(),
            audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
            dashboard=DashboardConfig(
                enabled=True,
                server=DashboardServerConfig(
                    rate_limit=RateLimitConfig(
                        enabled=True,
                        default_per_minute=2,
                        api_per_minute=2,
                        burst_multiplier=1.0,
                    ),
                ),
            ),
        )
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            # Two should succeed, the third should 429. /api/drift is in
            # the api group with default_per_minute=2, burst=1 → cap=2.
            assert test_client.get("/api/drift").status_code == 200
            assert test_client.get("/api/drift").status_code == 200
            third = test_client.get("/api/drift")
            assert third.status_code == 429

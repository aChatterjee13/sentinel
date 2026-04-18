"""Tests for bearer JWT validation.

We synthesise a self-signed RSA keypair in a fixture so the tests
never touch a real JWKS endpoint. The :class:`JWKSCache` is
monkeypatched to return our local key when asked for any kid.

The tests cover:

* Happy path — valid token decoded into a populated :class:`Principal`.
* Expired tokens → :class:`BearerTokenError`.
* Wrong audience / issuer → :class:`BearerTokenError`.
* Tampered signature → :class:`BearerTokenError`.
* Missing claims → :class:`BearerTokenError`.
* End-to-end through ``create_dashboard_app`` with ``auth: bearer``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    BearerAuthConfig,
    DashboardConfig,
    DashboardServerConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    RBACConfig,
    SentinelConfig,
)
from sentinel.core.exceptions import BearerTokenError
from sentinel.dashboard.security.bearer import (
    JWKSCache,
    validate_bearer_token,
)
from sentinel.dashboard.security.rbac import RBACPolicy

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

import jwt
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from sentinel.dashboard.server import create_dashboard_app

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rsa_keys() -> dict[str, Any]:
    """Generate an RSA keypair and the corresponding JWK dict."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    pem_private = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Build a JWK from the public key via PyJWT's RSAAlgorithm.
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    jwk_json = RSAAlgorithm.to_jwk(RSAAlgorithm.from_jwk(_pem_to_jwk_dict(public_pem)))
    jwk_dict = json.loads(jwk_json)
    jwk_dict["kid"] = "test-kid-1"
    jwk_dict["alg"] = "RS256"
    jwk_dict["use"] = "sig"

    return {
        "private_pem": pem_private,
        "jwk": jwk_dict,
        "kid": "test-kid-1",
    }


def _pem_to_jwk_dict(public_pem: bytes) -> dict[str, Any]:
    """Convert a PEM public key into a JWK dict using PyJWT."""
    # PyJWT expects a JWK-shaped dict for from_jwk; the simplest path
    # is to load the PEM via cryptography and re-export via to_jwk.
    public_key = serialization.load_pem_public_key(public_pem)
    return json.loads(RSAAlgorithm.to_jwk(public_key))


@pytest.fixture
def bearer_cfg() -> BearerAuthConfig:
    return BearerAuthConfig(
        jwks_url="https://example.test/jwks.json",
        issuer="https://issuer.test",
        audience="sentinel-dashboard",
        username_claim="sub",
        roles_claim="roles",
        algorithms=["RS256"],
    )


@pytest.fixture
def jwks_cache(rsa_keys: dict[str, Any]) -> JWKSCache:
    """A JWKS cache pre-populated with our test key."""
    cache = JWKSCache(ttl_seconds=3600)
    # Stub _fetch so we never make a real HTTP call.
    keys = {rsa_keys["kid"]: rsa_keys["jwk"]}
    cache._cache["https://example.test/jwks.json"] = (time.monotonic() + 3600, keys)
    return cache


@pytest.fixture
def rbac_policy() -> RBACPolicy:
    return RBACPolicy(RBACConfig(enabled=True))


def _make_token(
    rsa_keys: dict[str, Any],
    *,
    sub: str = "alice",
    roles: list[str] | None = None,
    iss: str | None = "https://issuer.test",
    aud: str | None = "sentinel-dashboard",
    exp_offset: int = 60,
    nbf_offset: int = -10,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": sub,
        "iat": now,
        "nbf": now + nbf_offset,
        "exp": now + exp_offset,
    }
    if iss is not None:
        claims["iss"] = iss
    if aud is not None:
        claims["aud"] = aud
    if roles is not None:
        claims["roles"] = roles
    if extra_claims:
        claims.update(extra_claims)

    return jwt.encode(
        claims,
        rsa_keys["private_pem"],
        algorithm="RS256",
        headers={"kid": rsa_keys["kid"]},
    )


# ── validate_bearer_token unit tests ─────────────────────────────────


class TestValidateBearerToken:
    def test_happy_path(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys, sub="alice", roles=["admin"])
        principal = validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)
        assert principal.username == "alice"
        assert principal.auth_mode == "bearer"
        assert "*" in principal.permissions

    def test_roles_as_string(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys, sub="bob", roles=None, extra_claims={"roles": "operator"})
        principal = validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)
        assert principal.has_permission("deployments.promote")

    def test_expired_token_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys, exp_offset=-3600)
        with pytest.raises(BearerTokenError, match="JWT validation failed"):
            validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)

    def test_wrong_audience_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys, aud="some-other-app")
        with pytest.raises(BearerTokenError):
            validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)

    def test_wrong_issuer_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys, iss="https://attacker.test")
        with pytest.raises(BearerTokenError):
            validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)

    def test_tampered_signature_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys)
        # Flip a few characters in the signature segment.
        head, payload, sig = token.split(".")
        tampered = ".".join([head, payload, sig[:-4] + "AAAA"])
        with pytest.raises(BearerTokenError):
            validate_bearer_token(tampered, bearer_cfg, rbac_policy, jwks_cache)

    def test_missing_username_claim_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        # Replace the standard sub claim with the empty string.
        token = _make_token(rsa_keys, sub="")
        with pytest.raises(BearerTokenError, match="username claim"):
            validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)

    def test_unknown_kid_raises(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        rbac_policy: RBACPolicy,
    ) -> None:
        # Empty cache + a stubbed _fetch that returns a different kid.
        cache = JWKSCache(ttl_seconds=3600)

        def _stub_fetch(_url: str) -> dict[str, dict[str, Any]]:
            return {"completely-different-kid": rsa_keys["jwk"]}

        cache._fetch = _stub_fetch  # type: ignore[method-assign]

        token = _make_token(rsa_keys)
        with pytest.raises(BearerTokenError, match="kid="):
            validate_bearer_token(token, bearer_cfg, rbac_policy, cache)

    def test_unparseable_token_raises(
        self,
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        with pytest.raises(BearerTokenError):
            validate_bearer_token("not.a.token", bearer_cfg, rbac_policy, jwks_cache)

    def test_missing_kid_header_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        # Encode a token with an empty headers dict — no kid.
        now = int(time.time())
        token = jwt.encode(
            {
                "sub": "alice",
                "exp": now + 60,
                "iss": "https://issuer.test",
                "aud": "sentinel-dashboard",
            },
            rsa_keys["private_pem"],
            algorithm="RS256",
        )
        with pytest.raises(BearerTokenError, match="missing 'kid'"):
            validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)

    def test_missing_jwks_url_rejected(
        self,
        rsa_keys: dict[str, Any],
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        cfg = BearerAuthConfig(jwks_url=None, audience="sentinel-dashboard")
        token = _make_token(rsa_keys)
        with pytest.raises(BearerTokenError, match="jwks_url"):
            validate_bearer_token(token, cfg, rbac_policy, jwks_cache)

    def test_roles_missing_claim_falls_back_to_default(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(rsa_keys, sub="alice", roles=None)
        principal = validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)
        # No roles claim → resolved with roles=None → default role applies.
        assert principal.username == "alice"

    def test_invalid_roles_claim_type_rejected(
        self,
        rsa_keys: dict[str, Any],
        bearer_cfg: BearerAuthConfig,
        jwks_cache: JWKSCache,
        rbac_policy: RBACPolicy,
    ) -> None:
        token = _make_token(
            rsa_keys, sub="alice", roles=None, extra_claims={"roles": {"admin": True}}
        )
        with pytest.raises(BearerTokenError, match="must be a string or list"):
            validate_bearer_token(token, bearer_cfg, rbac_policy, jwks_cache)


# ── JWKSCache fetch / invalidate / default singleton ──────────────────


class TestJWKSCacheBehaviour:
    def test_invalidate_specific_url(self, rsa_keys: dict[str, Any]) -> None:
        cache = JWKSCache()
        cache._cache["https://a.test/jwks.json"] = (time.monotonic() + 3600, {})
        cache._cache["https://b.test/jwks.json"] = (time.monotonic() + 3600, {})
        cache.invalidate("https://a.test/jwks.json")
        assert "https://a.test/jwks.json" not in cache._cache
        assert "https://b.test/jwks.json" in cache._cache

    def test_invalidate_all(self) -> None:
        cache = JWKSCache()
        cache._cache["https://a.test/jwks.json"] = (time.monotonic() + 3600, {})
        cache._cache["https://b.test/jwks.json"] = (time.monotonic() + 3600, {})
        cache.invalidate()
        assert cache._cache == {}

    def test_get_default_jwks_cache_returns_singleton(self) -> None:
        from sentinel.dashboard.security.bearer import get_default_jwks_cache

        cache_a = get_default_jwks_cache()
        cache_b = get_default_jwks_cache()
        assert cache_a is cache_b

    def test_fetch_via_httpx_mock(
        self, monkeypatch: pytest.MonkeyPatch, rsa_keys: dict[str, Any]
    ) -> None:
        # Stub httpx.get to return a JWKS document with our test key.
        import httpx

        class _StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"keys": [rsa_keys["jwk"]]}

        def _fake_get(url: str, timeout: float = 5.0) -> _StubResponse:
            return _StubResponse()

        monkeypatch.setattr(httpx, "get", _fake_get)
        cache = JWKSCache(ttl_seconds=3600)
        jwk = cache.get_key("https://example.test/jwks.json", rsa_keys["kid"])
        assert jwk["kid"] == rsa_keys["kid"]

    def test_fetch_skips_keys_without_kid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        class _StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                # Two entries — one without kid, one with.
                return {"keys": [{"kty": "RSA"}, {"kty": "RSA", "kid": "k1"}]}

        monkeypatch.setattr(httpx, "get", lambda url, timeout=5.0: _StubResponse())
        cache = JWKSCache()
        jwk = cache.get_key("https://example.test/jwks.json", "k1")
        assert jwk == {"kty": "RSA", "kid": "k1"}

    def test_fetch_empty_jwks_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        class _StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"keys": []}

        monkeypatch.setattr(httpx, "get", lambda url, timeout=5.0: _StubResponse())
        cache = JWKSCache()
        with pytest.raises(BearerTokenError, match="no usable keys"):
            cache.get_key("https://example.test/jwks.json", "k1")

    def test_fetch_http_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        def _raise(url: str, timeout: float = 5.0) -> None:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", _raise)
        cache = JWKSCache()
        with pytest.raises(BearerTokenError, match="failed to fetch JWKS"):
            cache.get_key("https://example.test/jwks.json", "k1")

    def test_fetch_returns_keys_but_kid_missing_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        class _StubResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"keys": [{"kty": "RSA", "kid": "other"}]}

        monkeypatch.setattr(httpx, "get", lambda url, timeout=5.0: _StubResponse())
        cache = JWKSCache()
        with pytest.raises(BearerTokenError, match="kid="):
            cache.get_key("https://example.test/jwks.json", "missing")


# ── End-to-end through the FastAPI app ───────────────────────────────


class TestBearerEndToEnd:
    def _make_config(self, tmp_path: Path) -> SentinelConfig:
        return SentinelConfig(
            model=ModelConfig(name="bearer_test_model", domain="tabular"),
            drift=DriftConfig(
                data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
            ),
            alerts=AlertsConfig(),
            audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
            dashboard=DashboardConfig(
                enabled=True,
                server=DashboardServerConfig(
                    auth="bearer",
                    rbac=RBACConfig(enabled=True),
                    bearer=BearerAuthConfig(
                        jwks_url="https://example.test/jwks.json",
                        issuer="https://issuer.test",
                        audience="sentinel-dashboard",
                    ),
                ),
            ),
        )

    def test_valid_token_returns_200(
        self,
        tmp_path: Path,
        rsa_keys: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch the module-level JWKS cache so the dispatcher
        # finds our key without any HTTP traffic.
        from sentinel.dashboard.security import bearer as bearer_module

        cache = JWKSCache(ttl_seconds=3600)
        cache._cache["https://example.test/jwks.json"] = (
            time.monotonic() + 3600,
            {rsa_keys["kid"]: rsa_keys["jwk"]},
        )
        monkeypatch.setattr(bearer_module, "_DEFAULT_JWKS_CACHE", cache)

        cfg = self._make_config(tmp_path)
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)

        token = _make_token(rsa_keys, sub="alice", roles=["admin"])
        with TestClient(app) as test_client:
            resp = test_client.get(
                "/api/drift",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

    def test_missing_authorization_returns_401(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keys: dict[str, Any],
    ) -> None:
        from sentinel.dashboard.security import bearer as bearer_module

        cache = JWKSCache(ttl_seconds=3600)
        cache._cache["https://example.test/jwks.json"] = (
            time.monotonic() + 3600,
            {rsa_keys["kid"]: rsa_keys["jwk"]},
        )
        monkeypatch.setattr(bearer_module, "_DEFAULT_JWKS_CACHE", cache)

        cfg = self._make_config(tmp_path)
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)

        with TestClient(app) as test_client:
            resp = test_client.get("/api/drift")
            assert resp.status_code == 401
            assert "WWW-Authenticate" in resp.headers

    def test_expired_token_returns_401(
        self,
        tmp_path: Path,
        rsa_keys: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sentinel.dashboard.security import bearer as bearer_module

        cache = JWKSCache(ttl_seconds=3600)
        cache._cache["https://example.test/jwks.json"] = (
            time.monotonic() + 3600,
            {rsa_keys["kid"]: rsa_keys["jwk"]},
        )
        monkeypatch.setattr(bearer_module, "_DEFAULT_JWKS_CACHE", cache)

        cfg = self._make_config(tmp_path)
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)

        token = _make_token(rsa_keys, exp_offset=-3600)
        with TestClient(app) as test_client:
            resp = test_client.get(
                "/api/drift",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 401

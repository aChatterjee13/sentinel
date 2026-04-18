"""Bearer JWT validation for the dashboard.

Workstream #3 ships a JWKS-URL flow only — no Authorization Code +
PKCE, no session cookies, no reverse-proxy header trust. The
:func:`validate_bearer_token` function:

1. Decodes the JWT header to discover the ``kid``.
2. Asks the :class:`JWKSCache` for the corresponding public key,
   refreshing the cache from the configured ``jwks_url`` on a miss.
3. Verifies the signature, ``exp``, ``nbf``, ``iss``, and ``aud``
   claims via :func:`jwt.decode`.
4. Extracts the username and roles claims and asks the
   :class:`RBACPolicy` to resolve them into a fully populated
   :class:`Principal`.

The cache lives for the lifetime of the dashboard process. JWKS
documents are small and rotation is rare, so a TTL-based cache is
more than enough — we deliberately avoid pulling in a Redis or
async refresher.

This module deliberately does not import :mod:`jwt` at module load
time so that ``auth: none`` and ``auth: basic`` deployments do not
need PyJWT installed.
"""

import threading
import time
from typing import Any

import structlog

from sentinel.config.schema import BearerAuthConfig
from sentinel.core.exceptions import BearerTokenError
from sentinel.dashboard.security.principal import Principal
from sentinel.dashboard.security.rbac import RBACPolicy

log = structlog.get_logger(__name__)


class JWKSCache:
    """Lazy, TTL-bounded cache of public keys keyed by JWT ``kid``.

    The cache is keyed on a JWKS URL so a single process can talk
    to multiple identity providers if needed (rare, but cheap to
    support). Lookups are thread-safe so the FastAPI worker threads
    don't race when refreshing.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        # url -> (expires_at, {kid: jwk_dict})
        self._cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}

    def get_key(self, jwks_url: str, kid: str) -> dict[str, Any]:
        """Return the JWK for ``kid``, refreshing the cache on a miss."""
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(jwks_url)
            if entry is not None and entry[0] > now and kid in entry[1]:
                return entry[1][kid]

        keys = self._fetch(jwks_url)
        with self._lock:
            self._cache[jwks_url] = (now + self._ttl, keys)
            if kid not in keys:
                raise BearerTokenError(f"JWKS at {jwks_url} did not contain a key with kid={kid!r}")
            return keys[kid]

    def invalidate(self, jwks_url: str | None = None) -> None:
        """Drop cached entries for ``jwks_url`` or every URL when None."""
        with self._lock:
            if jwks_url is None:
                self._cache.clear()
            else:
                self._cache.pop(jwks_url, None)

    def _fetch(self, jwks_url: str) -> dict[str, dict[str, Any]]:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - dashboard extra always pulls httpx
            raise BearerTokenError(
                "httpx is required for JWKS fetching; install sentinel-mlops[dashboard]"
            ) from e

        try:
            resp = httpx.get(jwks_url, timeout=5.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise BearerTokenError(f"failed to fetch JWKS from {jwks_url}: {e}") from e

        keys: dict[str, dict[str, Any]] = {}
        for jwk in payload.get("keys", []):
            kid = jwk.get("kid")
            if kid is None:
                continue
            keys[kid] = jwk
        if not keys:
            raise BearerTokenError(f"JWKS document at {jwks_url} contained no usable keys")
        return keys


# Module-level singleton — one cache shared by every dashboard worker
# in the same process. Tests can monkeypatch this directly.
_DEFAULT_JWKS_CACHE = JWKSCache()


def get_default_jwks_cache() -> JWKSCache:
    """Return the process-wide JWKS cache (used by tests for monkeypatching)."""
    return _DEFAULT_JWKS_CACHE


def validate_bearer_token(
    token: str,
    cfg: BearerAuthConfig,
    rbac_policy: RBACPolicy,
    jwks_cache: JWKSCache | None = None,
) -> Principal:
    """Validate ``token`` and return the resolved :class:`Principal`.

    Args:
        token: The raw JWT string from the ``Authorization: Bearer``
            header (after stripping the scheme).
        cfg: The :class:`BearerAuthConfig` block.
        rbac_policy: The shared :class:`RBACPolicy` used to resolve
            the username + roles claim into a fully populated
            principal.
        jwks_cache: Optional override for testing. Defaults to the
            module-level singleton.

    Returns:
        A :class:`Principal` populated from the JWT claims.

    Raises:
        BearerTokenError: When any step of validation fails — bad
            signature, expired, wrong issuer/audience, missing key,
            unparseable token, etc.
    """
    if not cfg.jwks_url:
        raise BearerTokenError(
            "dashboard.server.auth=bearer requires dashboard.server.bearer.jwks_url"
        )

    try:
        import jwt
        from jwt import PyJWKClient  # noqa: F401  (importing eagerly surfaces missing deps)
    except ImportError as e:
        raise BearerTokenError(
            "PyJWT is required for bearer auth; install sentinel-mlops[dashboard]"
        ) from e

    cache = jwks_cache or _DEFAULT_JWKS_CACHE

    # 1. Pull the kid out of the unverified header.
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise BearerTokenError(f"could not parse JWT header: {e}") from e
    kid = header.get("kid")
    if not kid:
        raise BearerTokenError("JWT header is missing 'kid'")

    # 2. Resolve the kid to a JWK and convert it to a public key.
    jwk_dict = cache.get_key(cfg.jwks_url, kid)
    try:
        from jwt.algorithms import RSAAlgorithm
    except ImportError as e:  # pragma: no cover
        raise BearerTokenError("PyJWT[crypto] is required for RS256 bearer tokens") from e

    try:
        public_key = RSAAlgorithm.from_jwk(jwk_dict)
    except Exception as e:
        raise BearerTokenError(f"could not load JWK: {e}") from e

    # 3. Validate signature + standard claims.
    options: dict[str, Any] = {"require": ["exp"]}
    decode_kwargs: dict[str, Any] = {
        "key": public_key,
        "algorithms": cfg.algorithms,
        "options": options,
        "leeway": cfg.leeway_seconds,
    }
    if cfg.audience:
        decode_kwargs["audience"] = cfg.audience
    if cfg.issuer:
        decode_kwargs["issuer"] = cfg.issuer

    try:
        claims = jwt.decode(token, **decode_kwargs)
    except jwt.PyJWTError as e:
        raise BearerTokenError(f"JWT validation failed: {e}") from e

    # 4. Extract username + roles and ask the RBAC policy to build a principal.
    username = claims.get(cfg.username_claim)
    if not isinstance(username, str) or not username:
        raise BearerTokenError(f"JWT is missing required username claim {cfg.username_claim!r}")
    raw_roles = claims.get(cfg.roles_claim)
    roles: list[str] | None
    if raw_roles is None:
        roles = None
    elif isinstance(raw_roles, str):
        roles = [raw_roles]
    elif isinstance(raw_roles, list) and all(isinstance(r, str) for r in raw_roles):
        roles = raw_roles
    else:
        raise BearerTokenError(
            f"JWT roles claim {cfg.roles_claim!r} must be a string or list of strings"
        )

    return rbac_policy.resolve_principal(
        username=username,
        roles=roles,
        auth_mode="bearer",
    )


__all__ = [
    "BearerTokenError",
    "JWKSCache",
    "get_default_jwks_cache",
    "validate_bearer_token",
]

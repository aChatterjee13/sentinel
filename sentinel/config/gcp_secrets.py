"""GCP Secret Manager resolution for Sentinel config files.

Tokens of the form ``${gcpsm:project/secret-name}`` or
``${gcpsm:project/secret-name/version}`` are resolved at config-load
time, exactly like the Azure Key Vault ``${azkv:…}`` tokens.

The resolver uses Google's Application Default Credentials.

Design notes:

- Resolution happens at the same stage as env-var substitution.  The
  loader replaces every ``${gcpsm:…}`` token with the resolved
  plaintext **before** Pydantic validation.
- A module-level cache keeps the ``SecretManagerServiceClient``
  (singleton) and a per-secret-version result cache so subsequent
  lookups do not repeatedly hit the GCP API.
- When no explicit version is supplied, ``latest`` is used.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import structlog

from sentinel.core.exceptions import ConfigGCPSecretsError

log = structlog.get_logger(__name__)

# ``${gcpsm:project/secret-name}``  or  ``${gcpsm:project/secret-name/version}``
_GCPSM_PATTERN = re.compile(r"\$\{gcpsm:([^}]+)\}")

# Module-level caches.
_client: Any | None = None
_client_initialised: bool = False
_secret_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


def _get_client() -> Any:
    """Return (and lazily create) the cached GCP Secret Manager client.

    Returns:
        A ``google.cloud.secretmanager.SecretManagerServiceClient``.

    Raises:
        ConfigGCPSecretsError: When ``google-cloud-secret-manager`` is
            not installed.
    """
    global _client, _client_initialised
    with _cache_lock:
        if _client_initialised:
            return _client
        try:
            from google.cloud import secretmanager  # type: ignore[import-not-found]
        except ImportError as e:
            _client_initialised = True
            raise ConfigGCPSecretsError(
                "google-cloud-secret-manager not installed — "
                "`pip install google-cloud-secret-manager`"
            ) from e
        try:
            _client = secretmanager.SecretManagerServiceClient()
        except Exception as e:
            _client_initialised = True
            raise ConfigGCPSecretsError(
                f"could not construct SecretManagerServiceClient: {e}"
            ) from e
        _client_initialised = True
        return _client


def _parse_token(token: str) -> tuple[str, str, str]:
    """Split a raw token into ``(project, secret_name, version)``.

    ``"my-project/my-secret"``       → ``("my-project", "my-secret", "latest")``
    ``"my-project/my-secret/2"``     → ``("my-project", "my-secret", "2")``

    Raises:
        ConfigGCPSecretsError: When the token has fewer than two
            slash-separated parts.
    """
    parts = token.split("/")
    if len(parts) < 2:
        raise ConfigGCPSecretsError(
            f"invalid gcpsm token {token!r}. Expected format: "
            "project/secret-name or project/secret-name/version"
        )
    project = parts[0]
    secret_name = parts[1]
    version = parts[2] if len(parts) > 2 else "latest"
    return project, secret_name, version


def resolve_gcpsm(token: str) -> str:
    """Resolve a ``${gcpsm:project/secret-name}`` or versioned token.

    Args:
        token: Secret reference, e.g. ``"my-project/my-secret"`` or
            ``"my-project/my-secret/2"``.

    Returns:
        The secret payload as a UTF-8 string.

    Raises:
        ConfigGCPSecretsError: When the client library is not installed,
            when the token format is invalid, or when the secret cannot
            be fetched.
    """
    project, secret_name, version = _parse_token(token)
    resource_name = (
        f"projects/{project}/secrets/{secret_name}/versions/{version}"
    )

    with _cache_lock:
        cached = _secret_cache.get(resource_name)
        if cached is not None:
            return cached

    client = _get_client()
    try:
        response = client.access_secret_version(
            request={"name": resource_name}
        )
        value = response.payload.data.decode("UTF-8")
    except ConfigGCPSecretsError:
        raise
    except Exception as e:
        raise ConfigGCPSecretsError(
            f"could not fetch secret {resource_name!r}: {e}"
        ) from e

    with _cache_lock:
        _secret_cache[resource_name] = value
    log.info(
        "config.gcpsm.resolved",
        project=project,
        secret=secret_name,
        version=version,
    )
    return value


def substitute_gcpsm(
    value: str,
    *,
    strict: bool = False,
) -> str:
    """Replace every ``${gcpsm:…}`` token in ``value``.

    Args:
        value: A string that may contain zero or more ``${gcpsm:…}``
            tokens.
        strict: When True, any token that fails to resolve raises
            :class:`ConfigGCPSecretsError`.  When False, the literal
            token is preserved.

    Returns:
        The string with every resolvable token replaced.

    Raises:
        ConfigGCPSecretsError: In strict mode, when resolution fails
            for any token.
    """
    if "${gcpsm:" not in value:
        return value

    failures: list[tuple[str, str]] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        try:
            return resolve_gcpsm(token)
        except ConfigGCPSecretsError as e:
            if strict:
                failures.append((token, str(e)))
                return match.group(0)
            log.warning(
                "config.gcpsm.lenient_failure",
                token=token,
                error=str(e),
            )
            return match.group(0)

    substituted = _GCPSM_PATTERN.sub(_replace, value)
    if failures:
        joined = ", ".join(f"${{gcpsm:{t}}} ({err})" for t, err in failures)
        raise ConfigGCPSecretsError(
            f"could not resolve GCP Secret Manager reference(s): {joined}"
        )
    return substituted


def clear_cache() -> None:
    """Clear cached client and secret values.  Tests call this between runs."""
    global _client, _client_initialised
    with _cache_lock:
        _client = None
        _client_initialised = False
        _secret_cache.clear()


__all__ = [
    "_GCPSM_PATTERN",
    "clear_cache",
    "resolve_gcpsm",
    "substitute_gcpsm",
]

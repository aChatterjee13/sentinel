"""AWS Secrets Manager resolution for Sentinel config files.

Tokens of the form ``${awssm:secret-name}`` or ``${awssm:secret-name/key}``
are resolved at config-load time, exactly like the Azure Key Vault
``${azkv:…}`` tokens.

The resolver uses boto3's default credential chain (env vars,
``~/.aws/credentials``, IAM role, etc.).

Design notes:

- Resolution happens at the same stage as env-var substitution.  The
  loader replaces every ``${awssm:…}`` token with the resolved
  plaintext **before** Pydantic validation.
- A module-level cache keeps a ``boto3`` Secrets Manager client per
  region so subsequent lookups do not repeatedly negotiate credentials.
- Only ``SecretString`` payloads are supported.  Binary secrets are
  explicitly out of scope.
- For JSON-valued secrets, a ``/key`` suffix extracts a single field
  (e.g. ``${awssm:my-secret/api_key}``).
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

import structlog

from sentinel.core.exceptions import ConfigAWSSecretsError

log = structlog.get_logger(__name__)

# ``${awssm:secret-name}``  or  ``${awssm:secret-name/json-key}``
_AWSSM_PATTERN = re.compile(r"\$\{awssm:([^}]+)\}")

# Module-level caches — kept across config loads so long-running
# processes that reload config don't re-negotiate credentials.
_clients: dict[str | None, Any] = {}
_secret_cache: dict[tuple[str, str | None, str | None], str] = {}
_cache_lock = threading.Lock()


def _get_client(region_name: str | None = None) -> Any:
    """Return (and lazily create) a cached boto3 Secrets Manager client.

    Args:
        region_name: AWS region override.  Uses boto3's default chain
            when ``None``.

    Returns:
        A ``boto3`` Secrets Manager client.

    Raises:
        ConfigAWSSecretsError: When ``boto3`` is not installed.
    """
    with _cache_lock:
        client = _clients.get(region_name)
        if client is not None:
            return client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise ConfigAWSSecretsError(
                "boto3 not installed — `pip install boto3` or `pip install sentinel-mlops[aws]`"
            ) from e
        try:
            kwargs: dict[str, Any] = {}
            if region_name:
                kwargs["region_name"] = region_name
            client = boto3.client("secretsmanager", **kwargs)
        except Exception as e:
            raise ConfigAWSSecretsError(
                f"could not construct Secrets Manager client (region={region_name!r}): {e}"
            ) from e
        _clients[region_name] = client
        return client


def _parse_token(token: str) -> tuple[str, str | None]:
    """Split a raw token into ``(secret_name, json_key | None)``.

    ``"my-secret"``            → ``("my-secret", None)``
    ``"my-secret/api_key"``    → ``("my-secret", "api_key")``
    """
    parts = token.split("/", 1)
    secret_name = parts[0]
    json_key = parts[1] if len(parts) > 1 else None
    return secret_name, json_key


def resolve_awssm(
    token: str,
    *,
    region_name: str | None = None,
) -> str:
    """Resolve a single ``${awssm:secret-name}`` or ``${awssm:secret-name/key}`` token.

    Args:
        token: The secret reference, e.g. ``"my-secret"`` or
            ``"my-secret/api_key"``.
        region_name: AWS region override.  Uses boto3 default if ``None``.

    Returns:
        The secret string value.

    Raises:
        ConfigAWSSecretsError: When ``boto3`` is not installed, when the
            secret cannot be fetched, or when a JSON key is requested but
            not present.
    """
    secret_name, json_key = _parse_token(token)
    cache_key = (secret_name, json_key, region_name)

    with _cache_lock:
        cached = _secret_cache.get(cache_key)
        if cached is not None:
            return cached

    client = _get_client(region_name)
    try:
        resp = client.get_secret_value(SecretId=secret_name)
        secret_string = resp.get("SecretString", "")
    except ConfigAWSSecretsError:
        raise
    except Exception as e:
        raise ConfigAWSSecretsError(
            f"could not fetch secret {secret_name!r}: {e}"
        ) from e

    if json_key:
        try:
            secret_dict = json.loads(secret_string)
        except json.JSONDecodeError as e:
            raise ConfigAWSSecretsError(
                f"secret {secret_name!r} is not valid JSON but key "
                f"{json_key!r} was requested"
            ) from e
        if json_key not in secret_dict:
            raise ConfigAWSSecretsError(
                f"key {json_key!r} not found in secret {secret_name!r}. "
                f"Available keys: {sorted(secret_dict.keys())}"
            )
        value = str(secret_dict[json_key])
    else:
        value = secret_string

    with _cache_lock:
        _secret_cache[cache_key] = value
    log.info(
        "config.awssm.resolved",
        secret=secret_name,
        json_key=json_key,
        region=region_name,
    )
    return value


def substitute_awssm(
    value: str,
    *,
    region_name: str | None = None,
    strict: bool = False,
) -> str:
    """Replace every ``${awssm:…}`` token in ``value``.

    Args:
        value: A string that may contain zero or more ``${awssm:…}``
            tokens.
        region_name: AWS region override.
        strict: When True, any token that fails to resolve raises
            :class:`ConfigAWSSecretsError`.  When False, the literal
            token is preserved.

    Returns:
        The string with every resolvable token replaced.

    Raises:
        ConfigAWSSecretsError: In strict mode, when resolution fails
            for any token.
    """
    if "${awssm:" not in value:
        return value

    failures: list[tuple[str, str]] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        try:
            return resolve_awssm(token, region_name=region_name)
        except ConfigAWSSecretsError as e:
            if strict:
                failures.append((token, str(e)))
                return match.group(0)
            log.warning(
                "config.awssm.lenient_failure",
                token=token,
                error=str(e),
            )
            return match.group(0)

    substituted = _AWSSM_PATTERN.sub(_replace, value)
    if failures:
        joined = ", ".join(f"${{awssm:{t}}} ({err})" for t, err in failures)
        raise ConfigAWSSecretsError(
            f"could not resolve AWS Secrets Manager reference(s): {joined}"
        )
    return substituted


def clear_cache() -> None:
    """Clear cached clients and secret values.  Tests call this between runs."""
    with _cache_lock:
        _clients.clear()
        _secret_cache.clear()


__all__ = [
    "_AWSSM_PATTERN",
    "clear_cache",
    "resolve_awssm",
    "substitute_awssm",
]

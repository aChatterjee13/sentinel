"""Azure Key Vault secret resolution for ``${azkv:vault/secret}`` tokens.

This module is imported **lazily** by :mod:`sentinel.config.loader` so that
``import sentinel`` does not pull in ``azure-identity`` or
``azure-keyvault-secrets``. The resolver is only activated when a config
file actually contains an ``${azkv:…}`` token.

Design notes:

- Resolution happens at the same stage as env-var substitution. The
  loader replaces every ``${azkv:vault/secret}`` with the resolved
  plaintext **before** Pydantic validation. Because Pydantic wraps
  secret fields in :class:`pydantic.SecretStr` during validation, the
  plaintext only lives in the substituted payload dict for the duration
  of a single ``load()`` call — after validation it is inside a
  ``SecretStr`` like every other credential.
- A module-level cache keeps a :class:`SecretClient` per vault URL so
  subsequent lookups inside the same config load (or subsequent loads
  of the same config in long-running processes) do not repeatedly
  negotiate an access token.
- Auth is always :class:`~azure.identity.DefaultAzureCredential`. This
  is the documented WS#2 policy — no per-config credential types.
- Only the *latest* version of a secret is fetched. Pinning a specific
  version via ``${azkv:vault/secret/version}`` is explicitly out of
  scope for WS#2.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import structlog

from sentinel.core.exceptions import ConfigKeyVaultError

log = structlog.get_logger(__name__)

# ``${azkv:vault-name/secret-name}``
#
# * Vault name: 3-24 chars, alphanumeric + dashes, must start and end
#   with an alphanumeric character (Azure Key Vault naming rules).
# * Secret name: 1+ chars, alphanumeric + dashes (Azure Key Vault
#   naming rules).
_AZKV_PATTERN = re.compile(r"\$\{azkv:([A-Za-z0-9][A-Za-z0-9-]{1,22}[A-Za-z0-9])/([A-Za-z0-9-]+)\}")

# Module-level caches — kept across config loads so long-running
# processes that reload config don't re-negotiate tokens on every load.
_clients: dict[str, Any] = {}
_secret_cache: dict[tuple[str, str], str] = {}
_cache_lock = threading.Lock()


def _build_vault_url(vault_name: str) -> str:
    """Return the canonical Key Vault DNS name for ``vault_name``."""
    return f"https://{vault_name}.vault.azure.net"


def _get_client(vault_name: str) -> Any:
    """Return (and lazily create) a cached ``SecretClient`` for ``vault_name``."""
    url = _build_vault_url(vault_name)
    with _cache_lock:
        client = _clients.get(url)
        if client is not None:
            return client
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as e:
            raise ConfigKeyVaultError(
                "azure-keyvault-secrets extra not installed — `pip install sentinel-mlops[azure]`"
            ) from e
        try:
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=url, credential=credential)
        except Exception as e:
            raise ConfigKeyVaultError(
                f"could not construct SecretClient for vault {vault_name!r}: {e}"
            ) from e
        _clients[url] = client
        return client


def resolve_azkv(vault_name: str, secret_name: str) -> str:
    """Resolve a single ``${azkv:vault/secret}`` token to plaintext.

    Args:
        vault_name: Azure Key Vault name (without the ``vault.azure.net``
            suffix).
        secret_name: Secret name inside the vault.

    Returns:
        The plaintext value of the latest version of the secret.

    Raises:
        ConfigKeyVaultError: When the ``azure`` extra is not installed,
            when authentication fails, or when the secret does not
            exist / cannot be retrieved.
    """
    cache_key = (vault_name, secret_name)
    with _cache_lock:
        cached = _secret_cache.get(cache_key)
        if cached is not None:
            return cached

    client = _get_client(vault_name)
    try:
        secret = client.get_secret(secret_name)
    except Exception as e:
        raise ConfigKeyVaultError(
            f"could not fetch secret {secret_name!r} from vault {vault_name!r}: {e}"
        ) from e

    raw_value = getattr(secret, "value", None)
    if raw_value is None:
        raise ConfigKeyVaultError(f"secret {secret_name!r} in vault {vault_name!r} has no value")
    value = str(raw_value)

    with _cache_lock:
        _secret_cache[cache_key] = value
    log.info(
        "config.keyvault.resolved",
        vault=vault_name,
        secret=secret_name,
    )
    return value


def substitute_azkv(value: str, *, strict: bool = False) -> str:
    """Replace every ``${azkv:vault/secret}`` token in ``value``.

    Args:
        value: A string that may contain zero or more ``${azkv:…}``
            tokens.
        strict: When True, any token that fails to resolve raises
            :class:`ConfigKeyVaultError`. When False, the literal
            token is preserved so existing env-var-style lenient
            behaviour is matched.

    Returns:
        The string with every resolvable token replaced by the
        corresponding secret value.

    Raises:
        ConfigKeyVaultError: In strict mode, when resolution fails
            for any token.
    """
    if "${azkv:" not in value:
        return value

    failures: list[tuple[str, str, str]] = []

    def _replace(match: re.Match[str]) -> str:
        vault = match.group(1)
        secret = match.group(2)
        try:
            return resolve_azkv(vault, secret)
        except ConfigKeyVaultError as e:
            if strict:
                failures.append((vault, secret, str(e)))
                return match.group(0)
            log.warning(
                "config.keyvault.lenient_failure",
                vault=vault,
                secret=secret,
                error=str(e),
            )
            return match.group(0)

    substituted = _AZKV_PATTERN.sub(_replace, value)
    if failures:
        joined = ", ".join(f"${{azkv:{v}/{s}}} ({err})" for v, s, err in failures)
        raise ConfigKeyVaultError(f"could not resolve Key Vault reference(s): {joined}")
    return substituted


def clear_cache() -> None:
    """Clear cached clients and secret values. Tests call this between runs."""
    with _cache_lock:
        _clients.clear()
        _secret_cache.clear()


__all__ = [
    "_AZKV_PATTERN",
    "clear_cache",
    "resolve_azkv",
    "substitute_azkv",
]

"""Tests for the ``${azkv:vault/secret}`` Key Vault resolver."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from sentinel.config import keyvault
from sentinel.config.keyvault import (
    _AZKV_PATTERN,
    clear_cache,
    resolve_azkv,
    substitute_azkv,
)
from sentinel.config.loader import ConfigLoader
from sentinel.core.exceptions import ConfigKeyVaultError


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Clear the Key Vault cache between tests so they don't leak state."""
    clear_cache()
    yield
    clear_cache()


def _install_fake_azure_sdk(
    monkeypatch: pytest.MonkeyPatch,
    secret_values: dict[tuple[str, str], str] | None = None,
    raise_on_get: Exception | None = None,
    raise_on_client_ctor: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Install minimal ``azure.identity`` + ``azure.keyvault.secrets`` shims.

    Returns the ``DefaultAzureCredential`` class mock and the
    ``SecretClient`` class mock so assertions can inspect call counts.
    """
    secret_values = secret_values or {}
    call_state: dict[str, int] = {"client_ctor": 0, "credential_ctor": 0}

    credential_ctor = MagicMock(name="DefaultAzureCredential")

    def _credential_factory(*args: Any, **kwargs: Any) -> MagicMock:
        call_state["credential_ctor"] += 1
        credential_ctor(*args, **kwargs)
        return MagicMock(name="credential")

    class FakeSecret:
        def __init__(self, value: str) -> None:
            self.value = value

    class FakeSecretClient:
        def __init__(self, vault_url: str, credential: Any) -> None:
            call_state["client_ctor"] += 1
            if raise_on_client_ctor is not None:
                raise raise_on_client_ctor
            self.vault_url = vault_url
            self.credential = credential

        def get_secret(self, name: str) -> FakeSecret:
            if raise_on_get is not None:
                raise raise_on_get
            # Vault name = portion of URL between https:// and .vault
            vault_name = self.vault_url.replace("https://", "").split(".")[0]
            key = (vault_name, name)
            if key not in secret_values:
                raise RuntimeError(f"unknown secret: {key}")
            return FakeSecret(secret_values[key])

    client_ctor = MagicMock(name="SecretClient", side_effect=FakeSecretClient)

    # Build the nested module structure and install into sys.modules.
    azure_mod = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = _credential_factory  # type: ignore[attr-defined]
    keyvault_mod = types.ModuleType("azure.keyvault")
    secrets_mod = types.ModuleType("azure.keyvault.secrets")
    secrets_mod.SecretClient = client_ctor  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_mod)
    monkeypatch.setitem(sys.modules, "azure.keyvault", keyvault_mod)
    monkeypatch.setitem(sys.modules, "azure.keyvault.secrets", secrets_mod)

    # Expose call counters on the credential mock so tests can assert
    # that the credential constructor was called.
    credential_ctor.call_count_tracker = call_state  # type: ignore[attr-defined]
    return credential_ctor, client_ctor


class TestAzkvPattern:
    def test_matches_vault_and_secret(self) -> None:
        m = _AZKV_PATTERN.search("${azkv:myvault/slack-hook}")
        assert m is not None
        assert m.group(1) == "myvault"
        assert m.group(2) == "slack-hook"

    def test_rejects_too_short_vault_name(self) -> None:
        assert _AZKV_PATTERN.search("${azkv:ab/secret}") is None

    def test_rejects_too_long_vault_name(self) -> None:
        # 25 chars — exceeds the 24-char max.
        assert _AZKV_PATTERN.search("${azkv:abcdefghijklmnopqrstuvwxy/s}") is None

    def test_allows_dashes(self) -> None:
        m = _AZKV_PATTERN.search("${azkv:prod-vault-01/slack-webhook-url}")
        assert m is not None
        assert m.group(1) == "prod-vault-01"
        assert m.group(2) == "slack-webhook-url"

    def test_rejects_leading_dash(self) -> None:
        assert _AZKV_PATTERN.search("${azkv:-badstart/s}") is None

    def test_rejects_trailing_dash(self) -> None:
        # Vault ending with a dash is invalid per Azure naming rules.
        assert _AZKV_PATTERN.search("${azkv:badend-/secret}") is None


class TestResolveAzkvHappyPath:
    def test_resolves_single_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(
            monkeypatch,
            secret_values={("myvault", "slack-hook"): "https://hooks.slack.com/x"},
        )
        result = resolve_azkv("myvault", "slack-hook")
        assert result == "https://hooks.slack.com/x"

    def test_second_lookup_uses_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, client_ctor = _install_fake_azure_sdk(
            monkeypatch,
            secret_values={("myvault", "api-key"): "topsecret"},
        )
        # First call builds the client.
        resolve_azkv("myvault", "api-key")
        # Second call must be served from cache — no new client.
        resolve_azkv("myvault", "api-key")
        # SecretClient constructor should only have been called once.
        assert client_ctor.call_count == 1

    def test_different_vaults_get_different_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, client_ctor = _install_fake_azure_sdk(
            monkeypatch,
            secret_values={
                ("vault1", "s"): "v1",
                ("vault2", "s"): "v2",
            },
        )
        assert resolve_azkv("vault1", "s") == "v1"
        assert resolve_azkv("vault2", "s") == "v2"
        assert client_ctor.call_count == 2


class TestResolveAzkvFailures:
    def test_missing_sdk_raises_config_keyvault_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure azure-keyvault-secrets is not importable.
        monkeypatch.setitem(sys.modules, "azure.keyvault.secrets", None)
        monkeypatch.setitem(sys.modules, "azure.identity", None)
        with pytest.raises(ConfigKeyVaultError, match="azure-keyvault-secrets"):
            resolve_azkv("myvault", "s")

    def test_get_secret_failure_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(
            monkeypatch,
            raise_on_get=RuntimeError("network blip"),
        )
        with pytest.raises(ConfigKeyVaultError, match="network blip"):
            resolve_azkv("myvault", "s")

    def test_client_ctor_failure_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(
            monkeypatch,
            raise_on_client_ctor=RuntimeError("bad credential"),
        )
        with pytest.raises(ConfigKeyVaultError, match="bad credential"):
            resolve_azkv("myvault", "s")


class TestSubstituteAzkv:
    def test_no_token_returns_original_string(self) -> None:
        assert substitute_azkv("hello world") == "hello world"

    def test_strict_replaces_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(
            monkeypatch,
            secret_values={("myvault", "slack"): "https://slack.test/x"},
        )
        result = substitute_azkv("${azkv:myvault/slack}", strict=True)
        assert result == "https://slack.test/x"

    def test_strict_raises_on_missing_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(monkeypatch, secret_values={})
        with pytest.raises(ConfigKeyVaultError):
            substitute_azkv("${azkv:myvault/nonexistent}", strict=True)

    def test_lenient_preserves_literal_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(monkeypatch, secret_values={})
        literal = "${azkv:myvault/nonexistent}"
        assert substitute_azkv(literal, strict=False) == literal


class TestLoaderIntegration:
    def _write_config(self, path: Path, body: str) -> None:
        path.write_text(body)

    def test_yaml_roundtrip_resolves_to_secret_str(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_azure_sdk(
            monkeypatch,
            secret_values={
                ("prod-vault", "slack-hook"): "https://hooks.slack.test/abc",
            },
        )
        cfg_path = tmp_path / "sentinel.yaml"
        self._write_config(
            cfg_path,
            """
version: "1.0"
model:
  name: claims_fraud
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:prod-vault/slack-hook}
""",
        )
        cfg = ConfigLoader(cfg_path).load()
        channel = cfg.alerts.channels[0]
        assert isinstance(channel.webhook_url, SecretStr)
        assert channel.webhook_url.get_secret_value() == "https://hooks.slack.test/abc"

    def test_strict_mode_surfaces_keyvault_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_azure_sdk(monkeypatch, secret_values={})
        cfg_path = tmp_path / "sentinel.yaml"
        self._write_config(
            cfg_path,
            """
version: "1.0"
model:
  name: x
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:prod-vault/missing}
""",
        )
        with pytest.raises(ConfigKeyVaultError):
            ConfigLoader(cfg_path, strict_env=True).load()

    def test_lenient_mode_preserves_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_azure_sdk(monkeypatch, secret_values={})
        cfg_path = tmp_path / "sentinel.yaml"
        self._write_config(
            cfg_path,
            """
version: "1.0"
model:
  name: x
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:prod-vault/missing}
""",
        )
        # Lenient mode should preserve the token literally. The
        # SecretStr wrapping will still happen at validation time.
        cfg = ConfigLoader(cfg_path, strict_env=False).load()
        channel = cfg.alerts.channels[0]
        assert channel.webhook_url is not None
        assert channel.webhook_url.get_secret_value() == "${azkv:prod-vault/missing}"

    def test_cache_shared_between_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, client_ctor = _install_fake_azure_sdk(
            monkeypatch,
            secret_values={
                ("prod-vault", "slack"): "https://slack.test/x",
                ("prod-vault", "teams"): "https://teams.test/y",
            },
        )
        cfg_path = tmp_path / "sentinel.yaml"
        self._write_config(
            cfg_path,
            """
version: "1.0"
model:
  name: x
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:prod-vault/slack}
    - type: teams
      webhook_url: ${azkv:prod-vault/teams}
""",
        )
        ConfigLoader(cfg_path).load()
        # Two secrets fetched, same vault — only one client should
        # have been built.
        assert client_ctor.call_count == 1


class TestClearCache:
    def test_clear_cache_drops_clients_and_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_sdk(
            monkeypatch,
            secret_values={("myvault", "s"): "v"},
        )
        resolve_azkv("myvault", "s")
        assert keyvault._clients
        assert keyvault._secret_cache
        clear_cache()
        assert not keyvault._clients
        assert not keyvault._secret_cache

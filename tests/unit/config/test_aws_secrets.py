"""Tests for the ``${awssm:…}`` AWS Secrets Manager resolver."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from sentinel.config import aws_secrets
from sentinel.config.aws_secrets import (
    _AWSSM_PATTERN,
    clear_cache,
    resolve_awssm,
    substitute_awssm,
)
from sentinel.config.loader import ConfigLoader
from sentinel.core.exceptions import ConfigAWSSecretsError


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Clear the AWS SM cache between tests so they don't leak state."""
    clear_cache()
    yield  # type: ignore[misc]
    clear_cache()


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch,
    secret_values: dict[str, str] | None = None,
    raise_on_get: Exception | None = None,
) -> MagicMock:
    """Install a minimal ``boto3`` shim that returns canned secret values.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        secret_values: Mapping of ``SecretId`` → ``SecretString``.
        raise_on_get: If set, ``get_secret_value`` raises this.

    Returns:
        The fake ``boto3.client`` callable (for assertion access).
    """
    secret_values = secret_values or {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def get_secret_value(self, SecretId: str) -> dict[str, str]:
            if raise_on_get is not None:
                raise raise_on_get
            if SecretId not in secret_values:
                raise RuntimeError(f"unknown secret: {SecretId!r}")
            return {"SecretString": secret_values[SecretId]}

    def _fake_client(service: str, **kwargs: Any) -> FakeClient:
        return FakeClient(**kwargs)

    client_fn = MagicMock(side_effect=_fake_client)

    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = client_fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)

    return client_fn


# ── Pattern tests ─────────────────────────────────────────────────


class TestAwsSmPattern:
    def test_matches_simple_secret(self) -> None:
        m = _AWSSM_PATTERN.search("${awssm:my-secret}")
        assert m is not None
        assert m.group(1) == "my-secret"

    def test_matches_json_key(self) -> None:
        m = _AWSSM_PATTERN.search("${awssm:my-secret/api_key}")
        assert m is not None
        assert m.group(1) == "my-secret/api_key"

    def test_no_match_without_prefix(self) -> None:
        assert _AWSSM_PATTERN.search("${azkv:vault/secret}") is None


# ── resolve_awssm tests ──────────────────────────────────────────


class TestResolveAwsSm:
    def test_simple_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch, secret_values={"my-secret": "my-password"}
        )
        assert resolve_awssm("my-secret") == "my-password"

    def test_json_key_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch,
            secret_values={
                "my-secret": '{"api_key": "sk-123", "endpoint": "https://api.example.com"}'
            },
        )
        assert resolve_awssm("my-secret/api_key") == "sk-123"

    def test_json_key_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch,
            secret_values={"my-secret": '{"api_key": "sk-123"}'},
        )
        with pytest.raises(ConfigAWSSecretsError, match="not found"):
            resolve_awssm("my-secret/missing_key")

    def test_non_json_with_key_requested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch,
            secret_values={"my-secret": "plain-string-value"},
        )
        with pytest.raises(ConfigAWSSecretsError, match="not valid JSON"):
            resolve_awssm("my-secret/some_key")

    def test_boto3_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        with pytest.raises(ConfigAWSSecretsError, match="boto3 not installed"):
            resolve_awssm("my-secret")

    def test_get_failure_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch, raise_on_get=RuntimeError("network blip")
        )
        with pytest.raises(ConfigAWSSecretsError, match="network blip"):
            resolve_awssm("my-secret")

    def test_second_lookup_uses_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client_fn = _install_fake_boto3(
            monkeypatch, secret_values={"s": "v"}
        )
        resolve_awssm("s")
        resolve_awssm("s")
        # boto3.client should have been called only once.
        assert client_fn.call_count == 1


# ── substitute_awssm tests ───────────────────────────────────────


class TestSubstituteAwsSm:
    def test_no_token_returns_original(self) -> None:
        assert substitute_awssm("hello world") == "hello world"

    def test_strict_replaces_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch, secret_values={"slack": "https://slack.test/x"}
        )
        result = substitute_awssm("${awssm:slack}", strict=True)
        assert result == "https://slack.test/x"

    def test_strict_raises_on_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, secret_values={})
        with pytest.raises(ConfigAWSSecretsError):
            substitute_awssm("${awssm:nonexistent}", strict=True)

    def test_lenient_preserves_literal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, secret_values={})
        literal = "${awssm:nonexistent}"
        assert substitute_awssm(literal, strict=False) == literal


# ── Loader integration tests ─────────────────────────────────────


class TestLoaderIntegration:
    @staticmethod
    def _write_config(path: Path, body: str) -> None:
        path.write_text(body)

    def test_yaml_roundtrip_resolves_to_secret_str(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_boto3(
            monkeypatch,
            secret_values={"slack-hook": "https://hooks.slack.test/abc"},
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
      webhook_url: ${awssm:slack-hook}
""",
        )
        cfg = ConfigLoader(cfg_path).load()
        channel = cfg.alerts.channels[0]
        assert isinstance(channel.webhook_url, SecretStr)
        assert channel.webhook_url.get_secret_value() == "https://hooks.slack.test/abc"

    def test_strict_mode_surfaces_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_boto3(monkeypatch, secret_values={})
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
      webhook_url: ${awssm:missing}
""",
        )
        with pytest.raises(ConfigAWSSecretsError):
            ConfigLoader(cfg_path, strict_env=True).load()

    def test_lenient_mode_preserves_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_boto3(monkeypatch, secret_values={})
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
      webhook_url: ${awssm:missing}
""",
        )
        cfg = ConfigLoader(cfg_path, strict_env=False).load()
        channel = cfg.alerts.channels[0]
        assert channel.webhook_url is not None
        assert channel.webhook_url.get_secret_value() == "${awssm:missing}"


# ── clear_cache test ─────────────────────────────────────────────


class TestClearCache:
    def test_clears_clients_and_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(
            monkeypatch, secret_values={"s": "v"}
        )
        resolve_awssm("s")
        assert aws_secrets._clients
        assert aws_secrets._secret_cache
        clear_cache()
        assert not aws_secrets._clients
        assert not aws_secrets._secret_cache

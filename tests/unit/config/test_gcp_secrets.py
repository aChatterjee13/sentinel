"""Tests for the ``${gcpsm:…}`` GCP Secret Manager resolver."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from sentinel.config import gcp_secrets
from sentinel.config.gcp_secrets import (
    _GCPSM_PATTERN,
    clear_cache,
    resolve_gcpsm,
    substitute_gcpsm,
)
from sentinel.config.loader import ConfigLoader
from sentinel.core.exceptions import ConfigGCPSecretsError


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Clear the GCP SM cache between tests so they don't leak state."""
    clear_cache()
    yield  # type: ignore[misc]
    clear_cache()


def _install_fake_gcp(
    monkeypatch: pytest.MonkeyPatch,
    secret_values: dict[str, bytes] | None = None,
    raise_on_access: Exception | None = None,
) -> MagicMock:
    """Install a minimal ``google.cloud.secretmanager`` shim.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        secret_values: Mapping of full resource name → payload bytes.
        raise_on_access: If set, ``access_secret_version`` raises this.

    Returns:
        The ``SecretManagerServiceClient`` class mock.
    """
    secret_values = secret_values or {}

    class FakePayload:
        def __init__(self, data: bytes) -> None:
            self.data = data

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self.payload = FakePayload(data)

    class FakeClient:
        def access_secret_version(self, request: dict[str, Any]) -> FakeResponse:
            if raise_on_access is not None:
                raise raise_on_access
            name = request["name"]
            if name not in secret_values:
                raise RuntimeError(f"unknown secret version: {name!r}")
            return FakeResponse(secret_values[name])

    client_cls = MagicMock(side_effect=lambda: FakeClient())

    sm_mod = types.ModuleType("google.cloud.secretmanager")
    sm_mod.SecretManagerServiceClient = client_cls  # type: ignore[attr-defined]

    gc_mod = types.ModuleType("google.cloud")
    gc_mod.secretmanager = sm_mod  # type: ignore[attr-defined]

    google_mod = types.ModuleType("google")
    google_mod.cloud = gc_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", gc_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", sm_mod)

    return client_cls


# ── Pattern tests ─────────────────────────────────────────────────


class TestGcpSmPattern:
    def test_matches_project_secret(self) -> None:
        m = _GCPSM_PATTERN.search("${gcpsm:my-project/my-secret}")
        assert m is not None
        assert m.group(1) == "my-project/my-secret"

    def test_matches_versioned(self) -> None:
        m = _GCPSM_PATTERN.search("${gcpsm:my-project/my-secret/2}")
        assert m is not None
        assert m.group(1) == "my-project/my-secret/2"

    def test_no_match_for_other_prefixes(self) -> None:
        assert _GCPSM_PATTERN.search("${azkv:vault/secret}") is None
        assert _GCPSM_PATTERN.search("${awssm:secret}") is None


# ── resolve_gcpsm tests ──────────────────────────────────────────


class TestResolveGcpSm:
    def test_simple_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(
            monkeypatch,
            secret_values={
                "projects/my-project/secrets/my-secret/versions/latest": b"my-password"
            },
        )
        assert resolve_gcpsm("my-project/my-secret") == "my-password"

    def test_versioned_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(
            monkeypatch,
            secret_values={
                "projects/my-project/secrets/my-secret/versions/2": b"old-value"
            },
        )
        assert resolve_gcpsm("my-project/my-secret/2") == "old-value"

    def test_invalid_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(monkeypatch)
        with pytest.raises(ConfigGCPSecretsError, match="invalid gcpsm token"):
            resolve_gcpsm("just-a-name")

    def test_client_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", None)
        monkeypatch.setitem(sys.modules, "google.cloud", None)
        with pytest.raises(ConfigGCPSecretsError, match="not installed"):
            resolve_gcpsm("project/secret")

    def test_access_failure_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(
            monkeypatch, raise_on_access=RuntimeError("permission denied")
        )
        with pytest.raises(ConfigGCPSecretsError, match="permission denied"):
            resolve_gcpsm("my-project/my-secret")

    def test_second_lookup_uses_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client_cls = _install_fake_gcp(
            monkeypatch,
            secret_values={
                "projects/p/secrets/s/versions/latest": b"v"
            },
        )
        resolve_gcpsm("p/s")
        resolve_gcpsm("p/s")
        # Client constructor should only be called once.
        assert client_cls.call_count == 1


# ── substitute_gcpsm tests ───────────────────────────────────────


class TestSubstituteGcpSm:
    def test_no_token_returns_original(self) -> None:
        assert substitute_gcpsm("hello world") == "hello world"

    def test_strict_replaces_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(
            monkeypatch,
            secret_values={
                "projects/proj/secrets/slack/versions/latest": b"https://slack.test/x"
            },
        )
        result = substitute_gcpsm("${gcpsm:proj/slack}", strict=True)
        assert result == "https://slack.test/x"

    def test_strict_raises_on_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(monkeypatch, secret_values={})
        with pytest.raises(ConfigGCPSecretsError):
            substitute_gcpsm("${gcpsm:proj/nonexistent}", strict=True)

    def test_lenient_preserves_literal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(monkeypatch, secret_values={})
        literal = "${gcpsm:proj/nonexistent}"
        assert substitute_gcpsm(literal, strict=False) == literal


# ── Loader integration tests ─────────────────────────────────────


class TestLoaderIntegration:
    @staticmethod
    def _write_config(path: Path, body: str) -> None:
        path.write_text(body)

    def test_yaml_roundtrip_resolves_to_secret_str(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gcp(
            monkeypatch,
            secret_values={
                "projects/prod/secrets/slack-hook/versions/latest": b"https://hooks.slack.test/abc"
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
      webhook_url: ${gcpsm:prod/slack-hook}
""",
        )
        cfg = ConfigLoader(cfg_path).load()
        channel = cfg.alerts.channels[0]
        assert isinstance(channel.webhook_url, SecretStr)
        assert channel.webhook_url.get_secret_value() == "https://hooks.slack.test/abc"

    def test_strict_mode_surfaces_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gcp(monkeypatch, secret_values={})
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
      webhook_url: ${gcpsm:proj/missing}
""",
        )
        with pytest.raises(ConfigGCPSecretsError):
            ConfigLoader(cfg_path, strict_env=True).load()

    def test_lenient_mode_preserves_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gcp(monkeypatch, secret_values={})
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
      webhook_url: ${gcpsm:proj/missing}
""",
        )
        cfg = ConfigLoader(cfg_path, strict_env=False).load()
        channel = cfg.alerts.channels[0]
        assert channel.webhook_url is not None
        assert channel.webhook_url.get_secret_value() == "${gcpsm:proj/missing}"


# ── clear_cache test ─────────────────────────────────────────────


class TestClearCache:
    def test_clears_client_and_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(
            monkeypatch,
            secret_values={
                "projects/p/secrets/s/versions/latest": b"v"
            },
        )
        resolve_gcpsm("p/s")
        assert gcp_secrets._secret_cache
        assert gcp_secrets._client_initialised
        clear_cache()
        assert not gcp_secrets._secret_cache
        assert not gcp_secrets._client_initialised

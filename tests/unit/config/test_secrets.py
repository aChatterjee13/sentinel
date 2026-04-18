"""Tests for sentinel.config.secrets — masking, unwrapping, round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from sentinel.config.loader import load_config
from sentinel.config.schema import (
    AlertsConfig,
    ChannelConfig,
    DashboardConfig,
    DashboardServerConfig,
    SentinelConfig,
)
from sentinel.config.secrets import REDACTED, masked_dump, unwrap


class TestUnwrap:
    def test_none(self) -> None:
        assert unwrap(None) is None

    def test_plain_string(self) -> None:
        assert unwrap("hello") == "hello"

    def test_secret_str(self) -> None:
        assert unwrap(SecretStr("hello")) == "hello"

    def test_empty_secret_returns_none(self) -> None:
        assert unwrap(SecretStr("")) is None

    def test_empty_string_returns_none(self) -> None:
        assert unwrap("") is None


class TestSecretStrSchemaMigration:
    def test_channel_webhook_url_is_secret(self) -> None:
        ch = ChannelConfig(type="slack", webhook_url="https://hooks.slack.com/X/Y/Z")
        assert isinstance(ch.webhook_url, SecretStr)
        assert ch.webhook_url.get_secret_value() == "https://hooks.slack.com/X/Y/Z"

    def test_channel_webhook_url_repr_is_masked(self) -> None:
        ch = ChannelConfig(type="slack", webhook_url="https://hooks.slack.com/X/Y/Z")
        assert "hooks.slack.com" not in repr(ch)
        assert "**********" in repr(ch)

    def test_channel_routing_key_is_secret(self) -> None:
        ch = ChannelConfig(type="pagerduty", routing_key="r0ut3-key")
        assert isinstance(ch.routing_key, SecretStr)
        assert ch.routing_key.get_secret_value() == "r0ut3-key"

    def test_dashboard_basic_auth_password_is_secret(self) -> None:
        cfg = DashboardServerConfig(
            auth="basic", basic_auth_username="admin", basic_auth_password="hunter2"
        )
        assert isinstance(cfg.basic_auth_password, SecretStr)
        assert cfg.basic_auth_password.get_secret_value() == "hunter2"
        assert "hunter2" not in repr(cfg)

    def test_model_dump_json_masks_secrets(self) -> None:
        ch = ChannelConfig(type="slack", webhook_url="https://hooks.slack.com/X/Y/Z")
        dumped = ch.model_dump_json()
        assert "hooks.slack.com" not in dumped
        assert "**********" in dumped

    def test_none_secrets_stay_none(self) -> None:
        # email doesn't require webhook_url, so it can be None
        ch = ChannelConfig(type="email", webhook_url=None, recipients=["a@b.com"])
        assert ch.webhook_url is None


class TestMaskedDump:
    def test_redacts_secret_in_channel(self) -> None:
        ch = ChannelConfig(type="slack", webhook_url="https://hooks.slack.com/X/Y/Z")
        out = masked_dump(ch)
        assert out["webhook_url"] == REDACTED

    def test_unmask_returns_plaintext(self) -> None:
        ch = ChannelConfig(type="slack", webhook_url="https://hooks.slack.com/X/Y/Z")
        out = masked_dump(ch, unmask=True)
        assert out["webhook_url"] == "https://hooks.slack.com/X/Y/Z"

    def test_walks_nested_lists(self) -> None:
        cfg = AlertsConfig(
            channels=[
                ChannelConfig(type="slack", webhook_url="https://x"),
                ChannelConfig(type="teams", webhook_url="https://y"),
            ]
        )
        out = masked_dump(cfg)
        assert out["channels"][0]["webhook_url"] == REDACTED
        assert out["channels"][1]["webhook_url"] == REDACTED

    def test_handles_none_secret(self) -> None:
        ch = ChannelConfig(type="email", webhook_url=None, recipients=["a@b.com"])
        out = masked_dump(ch)
        # None secrets are emitted as None, not REDACTED
        assert out["webhook_url"] is None

    def test_full_config_dump_redacts_dashboard_password(self) -> None:
        cfg = SentinelConfig(
            model={"name": "m"},  # type: ignore[arg-type]
            dashboard=DashboardConfig(
                enabled=True,
                server=DashboardServerConfig(
                    auth="basic",
                    basic_auth_username="admin",
                    basic_auth_password="hunter2",
                ),
            ),
        )
        out = masked_dump(cfg)
        assert out["dashboard"]["server"]["basic_auth_password"] == REDACTED


class TestYamlRoundTrip:
    def test_yaml_secret_loads_and_unwraps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_SECRET_HOOK", "https://hooks.slack.com/secret")
        path = tmp_path / "sentinel.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: secret_test
  domain: tabular
alerts:
  channels:
    - type: slack
      webhook_url: ${MY_SECRET_HOOK}
"""
        )
        cfg = load_config(path)
        ch = cfg.alerts.channels[0]
        assert isinstance(ch.webhook_url, SecretStr)
        assert unwrap(ch.webhook_url) == "https://hooks.slack.com/secret"
        # The repr never reveals the value.
        assert "hooks.slack.com" not in repr(ch)

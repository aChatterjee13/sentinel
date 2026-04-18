"""Tests for the ``sentinel config sign`` and ``verify-signature`` CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from sentinel.cli.main import cli
from sentinel.config.signing import SIGNATURE_SUFFIX

_KEY_HEX = "k" * 32  # 32 bytes of key material via UTF-8 encoding


def _write_minimal_config(path: Path) -> None:
    path.write_text(
        """version: "1.0"
model:
  name: signed_cli_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
"""
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    cfg = tmp_path / "sentinel.yaml"
    _write_minimal_config(cfg)
    return cfg


class TestConfigSignCommand:
    def test_sign_writes_sidecar(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "sign", "--config", str(workspace)],
        )
        assert result.exit_code == 0, result.output
        sig_path = workspace.with_name(workspace.name + SIGNATURE_SUFFIX)
        assert sig_path.exists()
        assert "key fingerprint" in result.output

    def test_sign_respects_out_flag(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        out = tmp_path / "alt.sig"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "sign",
                "--config",
                str(workspace),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_sign_fails_without_key(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SENTINEL_CONFIG_KEY", raising=False)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "sign", "--config", str(workspace)],
        )
        assert result.exit_code != 0
        assert "unset" in result.output.lower() or "empty" in result.output.lower()


class TestConfigVerifySignatureCommand:
    def test_round_trip_passes(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        runner = CliRunner()
        sign = runner.invoke(cli, ["config", "sign", "--config", str(workspace)])
        assert sign.exit_code == 0, sign.output
        verify = runner.invoke(cli, ["config", "verify-signature", "--config", str(workspace)])
        assert verify.exit_code == 0, verify.output
        assert "OK" in verify.output

    def test_tampered_config_fails(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        runner = CliRunner()
        sign = runner.invoke(cli, ["config", "sign", "--config", str(workspace)])
        assert sign.exit_code == 0, sign.output

        # Mutate the YAML so the resolved payload no longer matches.
        workspace.write_text(
            workspace.read_text().replace("signed_cli_model", "tampered_cli_model")
        )

        verify = runner.invoke(cli, ["config", "verify-signature", "--config", str(workspace)])
        assert verify.exit_code == 1, verify.output
        assert "FAIL" in verify.output

    def test_wrong_key_fails(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        runner = CliRunner()
        sign = runner.invoke(cli, ["config", "sign", "--config", str(workspace)])
        assert sign.exit_code == 0, sign.output

        # Verify with a different key.
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", "x" * 32)
        verify = runner.invoke(cli, ["config", "verify-signature", "--config", str(workspace)])
        assert verify.exit_code == 1, verify.output

    def test_missing_signature_file_fails(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        runner = CliRunner()
        # Never sign — verify should complain.
        verify = runner.invoke(cli, ["config", "verify-signature", "--config", str(workspace)])
        assert verify.exit_code != 0
        assert "not found" in verify.output.lower()

    def test_uses_explicit_sig_path(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_HEX)
        runner = CliRunner()
        out = tmp_path / "explicit.sig"
        sign = runner.invoke(
            cli,
            [
                "config",
                "sign",
                "--config",
                str(workspace),
                "--out",
                str(out),
            ],
        )
        assert sign.exit_code == 0, sign.output
        verify = runner.invoke(
            cli,
            [
                "config",
                "verify-signature",
                "--config",
                str(workspace),
                "--sig",
                str(out),
            ],
        )
        assert verify.exit_code == 0, verify.output

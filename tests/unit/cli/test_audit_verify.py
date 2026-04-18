"""Tests for the ``sentinel audit`` CLI subcommands.

Covers ``query`` (the back-compat alias kept under the new audit
group), ``verify``, and ``chain-info``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sentinel.cli.main import cli


def _write_audit_config(
    path: Path,
    audit_dir: Path,
    *,
    tamper_evidence: bool = False,
) -> None:
    tamper_block = ""
    if tamper_evidence:
        tamper_block = "  tamper_evidence: true\n  signing_key_env: SENTINEL_AUDIT_TEST_KEY\n"
    path.write_text(
        f"""version: "1.0"
model:
  name: cli_audit_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
audit:
  storage: local
  path: {audit_dir}
{tamper_block}"""
    )


@pytest.fixture
def signed_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a config + audit directory pair with tamper evidence on."""
    monkeypatch.setenv("SENTINEL_AUDIT_TEST_KEY", "k" * 32)
    audit_dir = tmp_path / "audit"
    config_path = tmp_path / "sentinel.yaml"
    _write_audit_config(config_path, audit_dir, tamper_evidence=True)
    return config_path, audit_dir


def _strip_logs(output: str) -> str:
    """Drop the structlog INFO lines from CLI output."""
    return "\n".join(line for line in output.splitlines() if not line.startswith("2026"))


class TestAuditQueryAlias:
    def test_bare_audit_invokes_query(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``sentinel audit`` (no subcommand) tails the trail."""
        monkeypatch.setenv("SENTINEL_AUDIT_TEST_KEY", "k" * 32)
        config_path = tmp_path / "sentinel.yaml"
        _write_audit_config(config_path, tmp_path / "audit")

        # Seed an event by spinning up a client.
        from sentinel.core.client import SentinelClient

        client = SentinelClient.from_config(str(config_path))
        client.audit.log("smoke_event", model_name="cli_audit_model")

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "smoke_event" in result.output

    def test_explicit_query_subcommand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_AUDIT_TEST_KEY", "k" * 32)
        config_path = tmp_path / "sentinel.yaml"
        _write_audit_config(config_path, tmp_path / "audit")

        from sentinel.core.client import SentinelClient

        client = SentinelClient.from_config(str(config_path))
        client.audit.log("explicit_event", model_name="cli_audit_model")

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "query", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "explicit_event" in result.output


class TestAuditVerify:
    def test_verify_clean_chain_returns_zero(self, signed_workspace: tuple[Path, Path]) -> None:
        config_path, _ = signed_workspace
        from sentinel.core.client import SentinelClient

        client = SentinelClient.from_config(str(config_path))
        client.audit.log("event_one", model_name="cli_audit_model")
        client.audit.log("event_two", model_name="cli_audit_model")

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "verify", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_verify_detects_tampering(self, signed_workspace: tuple[Path, Path]) -> None:
        config_path, audit_dir = signed_workspace
        from sentinel.core.client import SentinelClient

        client = SentinelClient.from_config(str(config_path))
        client.audit.log("event_one", model_name="cli_audit_model", value="A")
        client.audit.log("event_two", model_name="cli_audit_model", value="B")

        # Hand-tamper one event on disk.
        files = sorted(audit_dir.glob("audit-*.jsonl"))
        rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        rows[0]["payload"]["value"] = "TAMPERED"
        files[0].write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "verify", "--config", str(config_path)])
        assert result.exit_code == 1
        assert "TAMPERED" in result.output

    def test_verify_empty_trail_is_ok(self, signed_workspace: tuple[Path, Path]) -> None:
        config_path, _ = signed_workspace
        # Trigger client init (creates audit dir) but log nothing.
        from sentinel.core.client import SentinelClient

        SentinelClient.from_config(str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "verify", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_verify_invalid_since_format_errors(self, signed_workspace: tuple[Path, Path]) -> None:
        config_path, _ = signed_workspace
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "audit",
                "verify",
                "--config",
                str(config_path),
                "--since",
                "not-a-date",
            ],
        )
        assert result.exit_code != 0
        assert "ISO datetime" in result.output


class TestAuditChainInfo:
    def test_chain_info_with_tamper_evidence(self, signed_workspace: tuple[Path, Path]) -> None:
        config_path, _ = signed_workspace
        from sentinel.core.client import SentinelClient

        client = SentinelClient.from_config(str(config_path))
        client.audit.log("first", model_name="cli_audit_model")
        client.audit.log("second", model_name="cli_audit_model")

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "chain-info", "--config", str(config_path)])
        assert result.exit_code == 0

        body = _strip_logs(result.output)
        idx = body.find("{")
        assert idx >= 0
        info = json.loads(body[idx:])
        assert info["tamper_evidence"] is True
        assert info["chain_head"] is not None
        assert len(info["chain_head"]) == 64
        assert info["key_fingerprint"] is not None
        assert len(info["key_fingerprint"]) == 8
        # The key itself must never appear.
        assert "k" * 32 not in result.output

    def test_chain_info_without_tamper_evidence(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_audit_config(config_path, tmp_path / "audit", tamper_evidence=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "chain-info", "--config", str(config_path)])
        assert result.exit_code == 0
        body = _strip_logs(result.output)
        idx = body.find("{")
        info = json.loads(body[idx:])
        assert info["tamper_evidence"] is False
        assert info["chain_head"] is None
        assert info["key_fingerprint"] is None

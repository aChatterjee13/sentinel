"""CLI smoke tests for `sentinel dashboard`."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
from click.testing import CliRunner

from sentinel.cli.main import cli


class TestDashboardCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output

    def test_missing_extras_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When fastapi/jinja2 are missing, surface a clear ClickException."""
        original_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in {
                "sentinel.dashboard.server",
                "sentinel.dashboard",
            } or name.startswith("sentinel.dashboard."):
                raise ImportError(f"mocked missing dependency for {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--config", "nope.yaml"])
        assert result.exit_code != 0
        assert "sentinel-mlops[dashboard]" in result.output

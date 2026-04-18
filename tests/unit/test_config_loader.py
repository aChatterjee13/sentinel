"""Unit tests for the YAML config loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sentinel.config.loader import ConfigLoader, load_config
from sentinel.config.schema import SentinelConfig
from sentinel.core.exceptions import ConfigNotFoundError, ConfigValidationError


class TestConfigLoader:
    def test_loads_minimal_yaml(self, example_yaml: Path) -> None:
        cfg = ConfigLoader(example_yaml).load()
        assert isinstance(cfg, SentinelConfig)
        assert cfg.model.name == "test_model_yaml"
        assert cfg.model.domain == "tabular"
        assert cfg.drift.data.method == "psi"

    def test_load_config_helper(self, example_yaml: Path) -> None:
        cfg = load_config(example_yaml)
        assert cfg.model.name == "test_model_yaml"

    def test_caches_subsequent_loads(self, example_yaml: Path) -> None:
        loader = ConfigLoader(example_yaml)
        first = loader.load()
        second = loader.load()
        assert first is second  # cached identity

    def test_force_reload(self, example_yaml: Path) -> None:
        loader = ConfigLoader(example_yaml)
        loader.load()
        reloaded = loader.load(force=True)
        assert reloaded.model.name == "test_model_yaml"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigNotFoundError):
            ConfigLoader(tmp_path / "nope.yaml").load()

    def test_invalid_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.txt"
        bad.write_text("model:\n  name: x")
        with pytest.raises(ConfigValidationError):
            ConfigLoader(bad).load()

    def test_validation_error_message(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("model:\n  type: not_a_real_type\n  domain: tabular")
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(bad).load()
        assert "config validation failed" in str(exc_info.value)

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_MODEL_NAME", "fraud_v2")
        path = tmp_path / "envs.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: ${MY_MODEL_NAME}
  domain: tabular
"""
        )
        cfg = ConfigLoader(path).load()
        assert cfg.model.name == "fraud_v2"

    def test_env_var_with_default(self, tmp_path: Path) -> None:
        path = tmp_path / "default.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: ${UNSET_VAR_FOR_TESTS:-fallback_name}
  domain: tabular
"""
        )
        os.environ.pop("UNSET_VAR_FOR_TESTS", None)
        cfg = ConfigLoader(path).load()
        assert cfg.model.name == "fallback_name"

    def test_extends_inheritance(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            """
version: "1.0"
model:
  name: base_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.2
"""
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            """
extends: base.yaml
model:
  name: child_model
  domain: tabular
drift:
  data:
    method: ks
    threshold: 0.05
"""
        )
        cfg = ConfigLoader(child).load()
        assert cfg.model.name == "child_model"
        assert cfg.drift.data.method == "ks"

    def test_nested_env_var_substitution_in_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLACK_HOOK_TEST", "https://hooks.slack.com/X/Y/Z")
        monkeypatch.setenv("TEAMS_HOOK_TEST", "https://outlook.office.com/abc")
        path = tmp_path / "nested.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: nested_env
  domain: tabular
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_HOOK_TEST}
      channel: "#alerts"
    - type: teams
      webhook_url: ${TEAMS_HOOK_TEST}
"""
        )
        cfg = ConfigLoader(path).load()
        assert len(cfg.alerts.channels) == 2
        # SecretStr round-trip — value should be present (we unwrap to compare).
        from sentinel.config.secrets import unwrap

        assert unwrap(cfg.alerts.channels[0].webhook_url) == "https://hooks.slack.com/X/Y/Z"
        assert unwrap(cfg.alerts.channels[1].webhook_url) == "https://outlook.office.com/abc"

    def test_json_config_format(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        payload = {
            "version": "1.0",
            "model": {"name": "json_model", "domain": "tabular"},
            "drift": {"data": {"method": "psi", "threshold": 0.15, "window": "7d"}},
        }
        path.write_text(json.dumps(payload))
        cfg = ConfigLoader(path).load()
        assert cfg.model.name == "json_model"
        assert cfg.drift.data.method == "psi"
        assert cfg.drift.data.threshold == 0.15

    def test_missing_parent_file_raises(self, tmp_path: Path) -> None:
        child = tmp_path / "child.yaml"
        child.write_text(
            """
extends: nonexistent_parent.yaml
model:
  name: orphan
  domain: tabular
"""
        )
        with pytest.raises(ConfigNotFoundError):
            ConfigLoader(child).load()

    def test_multiple_inheritance_levels(self, tmp_path: Path) -> None:
        # grandparent → parent → child, with each level overriding a field
        gp = tmp_path / "grandparent.yaml"
        gp.write_text(
            """
version: "1.0"
model:
  name: gp
  domain: tabular
  type: classification
drift:
  data:
    method: psi
    threshold: 0.5
audit:
  storage: local
  path: ./audit/
"""
        )
        p = tmp_path / "parent.yaml"
        p.write_text(
            """
extends: grandparent.yaml
model:
  name: parent_model
drift:
  data:
    threshold: 0.3
"""
        )
        c = tmp_path / "child.yaml"
        c.write_text(
            """
extends: parent.yaml
model:
  name: child_model
"""
        )
        cfg = ConfigLoader(c).load()
        # Child override wins.
        assert cfg.model.name == "child_model"
        # Parent override wins over grandparent.
        assert cfg.drift.data.threshold == 0.3
        # Grandparent-only field still survives the chain.
        assert cfg.model.type == "classification"
        assert cfg.drift.data.method == "psi"

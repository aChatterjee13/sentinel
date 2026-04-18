"""Tests for strict-mode env-var substitution in the config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.config.loader import ConfigLoader, _substitute_env, load_config
from sentinel.core.exceptions import ConfigMissingEnvVarError


class TestSubstituteEnvStrict:
    def test_strict_passes_when_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_TEST_VAR", "value")
        result = _substitute_env({"key": "${SENTINEL_TEST_VAR}"}, strict=True)
        assert result == {"key": "value"}

    def test_strict_honours_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SENTINEL_NEVER_SET", raising=False)
        result = _substitute_env({"key": "${SENTINEL_NEVER_SET:-fallback}"}, strict=True)
        assert result == {"key": "fallback"}

    def test_strict_raises_on_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SENTINEL_NEVER_SET", raising=False)
        with pytest.raises(ConfigMissingEnvVarError) as exc_info:
            _substitute_env({"key": "${SENTINEL_NEVER_SET}"}, strict=True)
        msg = str(exc_info.value)
        assert "SENTINEL_NEVER_SET" in msg
        assert "key" in msg

    def test_strict_error_includes_nested_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEP_VAR", raising=False)
        with pytest.raises(ConfigMissingEnvVarError) as exc_info:
            _substitute_env(
                {"alerts": {"channels": [{"webhook_url": "${DEEP_VAR}"}]}},
                strict=True,
            )
        msg = str(exc_info.value)
        assert "alerts.channels.0.webhook_url" in msg

    def test_lenient_keeps_literal_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LENIENT_VAR", raising=False)
        result = _substitute_env({"key": "${LENIENT_VAR}"}, strict=False)
        assert result == {"key": "${LENIENT_VAR}"}

    def test_strict_walks_lists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIST_VAR", raising=False)
        with pytest.raises(ConfigMissingEnvVarError) as exc_info:
            _substitute_env({"items": ["${LIST_VAR}"]}, strict=True)
        assert "items.0" in str(exc_info.value)

    def test_strict_treats_empty_string_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `VAR=` (empty) is a common shell foot-gun. Strict mode should
        # reject it just like a fully-unset var — an empty webhook URL
        # is not a usable value.
        monkeypatch.setenv("EMPTY_VAR", "")
        with pytest.raises(ConfigMissingEnvVarError) as exc_info:
            _substitute_env({"key": "${EMPTY_VAR}"}, strict=True)
        assert "EMPTY_VAR" in str(exc_info.value)

    def test_lenient_keeps_empty_string_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # In lenient mode, an empty env var still substitutes to "".
        # We must not break the back-compat behaviour.
        monkeypatch.setenv("EMPTY_VAR", "")
        result = _substitute_env({"key": "${EMPTY_VAR}"}, strict=False)
        assert result == {"key": ""}


class TestLoadConfigStrict:
    def test_strict_loader_fails_on_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SENTINEL_LOADER_VAR", raising=False)
        path = tmp_path / "sentinel.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: ${SENTINEL_LOADER_VAR}
  domain: tabular
"""
        )
        with pytest.raises(ConfigMissingEnvVarError):
            ConfigLoader(path, strict_env=True).load()

    def test_strict_loader_passes_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_LOADER_VAR", "ok_model")
        path = tmp_path / "sentinel.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: ${SENTINEL_LOADER_VAR}
  domain: tabular
"""
        )
        cfg = ConfigLoader(path, strict_env=True).load()
        assert cfg.model.name == "ok_model"

    def test_lenient_loader_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Lenient mode keeps the literal token. Pydantic still accepts a
        # plain string for `model.name` so loading succeeds.
        monkeypatch.delenv("LENIENT_LOADER_VAR", raising=False)
        path = tmp_path / "sentinel.yaml"
        path.write_text(
            """
version: "1.0"
model:
  name: ${LENIENT_LOADER_VAR}
  domain: tabular
"""
        )
        cfg = load_config(path)
        assert cfg.model.name == "${LENIENT_LOADER_VAR}"

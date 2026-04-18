"""YAML/JSON config loader with env-var substitution and inheritance.

The loader is a small pipeline:

``_read_raw``
    Parse YAML/JSON into a dict.
``_resolve_inheritance``
    Walk the ``extends:`` chain, merging parent values with child
    overrides. Detects circular inheritance and tags every value with
    its originating file via :mod:`sentinel.config.source`.
``_substitute_env``
    Replace ``${VAR}`` and ``${VAR:-default}`` tokens with values from
    ``os.environ``. ``strict=True`` raises
    :class:`ConfigMissingEnvVarError` on unresolved tokens with the
    JSON path included for fast debugging.
``SentinelConfig.model_validate``
    Pydantic validation with friendly error formatting that surfaces the
    originating file from the source map when available.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from sentinel.config.schema import SentinelConfig
from sentinel.config.signing import (
    ConfigSignature,
    read_signature_file,
    verify_config,
)
from sentinel.config.source import (
    ConfigSource,
    SourceMap,
    annotate,
    harvest,
    merge_with_sources,
)
from sentinel.core.exceptions import (
    ConfigCircularInheritanceError,
    ConfigKeyVaultError,
    ConfigMissingEnvVarError,
    ConfigNotFoundError,
    ConfigSignatureError,
    ConfigValidationError,
)
from sentinel.foundation.audit.keystore import BaseKeystore

log = structlog.get_logger(__name__)

# ${VAR} or ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _substitute_env(
    value: Any,
    *,
    strict: bool = False,
    path: tuple[str, ...] = (),
) -> Any:
    """Recursively replace ``${ENV_VAR}`` with values from ``os.environ``.

    Args:
        value: A primitive, list, or dict to walk.
        strict: When True, raise :class:`ConfigMissingEnvVarError` for
            any ``${VAR}`` token that is unset *and* has no
            ``:-default``. When False (default), the literal token is
            preserved (back-compat behaviour).
        path: Internal — JSON path stack used to enrich strict-mode
            errors. Callers should leave this empty.

    Returns:
        The substituted value (or the original if no substitution
        applied).

    Raises:
        ConfigMissingEnvVarError: In strict mode, when an unset env var
            is referenced.
    """
    if isinstance(value, str):
        missing: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            env_value = os.environ.get(var)
            # In strict mode an empty string counts as "unset" — an
            # empty webhook URL or routing key is just as broken as a
            # missing one, and `VAR=` shell syntax is a common foot-gun.
            if env_value:
                return env_value
            if env_value is not None and not strict:
                # Lenient: preserve empty-string substitution behaviour.
                return env_value
            if default is not None:
                return default
            if strict:
                missing.append(var)
                return match.group(0)  # placeholder; we'll raise below
            log.warning("config.unresolved_env_var", var=var, path=".".join(path) or "<root>")
            return match.group(0)

        substituted = _ENV_PATTERN.sub(_replace, value)
        if strict and missing:
            location = ".".join(path) or "<root>"
            joined = ", ".join(f"${{{name}}}" for name in missing)
            raise ConfigMissingEnvVarError(
                f"{location} references unset environment variable(s): {joined}"
            )

        # Second stage — resolve cloud secret-manager references.
        # Lazy imports so ``import sentinel`` does not pull in heavy
        # cloud SDKs. Only strings that contain a token pay the import.

        # Azure Key Vault: ``${azkv:vault/secret}``
        if "${azkv:" in substituted:
            from sentinel.config.keyvault import (
                ConfigKeyVaultError as _KVError,
            )
            from sentinel.config.keyvault import (
                substitute_azkv,
            )

            try:
                substituted = substitute_azkv(substituted, strict=strict)
            except _KVError as e:
                location = ".".join(path) or "<root>"
                raise ConfigKeyVaultError(f"{location}: {e}") from e

        # AWS Secrets Manager: ``${awssm:secret-name}`` or ``${awssm:secret-name/key}``
        if "${awssm:" in substituted:
            from sentinel.config.aws_secrets import substitute_awssm
            from sentinel.core.exceptions import ConfigAWSSecretsError

            try:
                substituted = substitute_awssm(substituted, strict=strict)
            except ConfigAWSSecretsError as e:
                location = ".".join(path) or "<root>"
                raise ConfigAWSSecretsError(f"{location}: {e}") from e

        # GCP Secret Manager: ``${gcpsm:project/secret-name}``
        if "${gcpsm:" in substituted:
            from sentinel.config.gcp_secrets import substitute_gcpsm
            from sentinel.core.exceptions import ConfigGCPSecretsError

            try:
                substituted = substitute_gcpsm(substituted, strict=strict)
            except ConfigGCPSecretsError as e:
                location = ".".join(path) or "<root>"
                raise ConfigGCPSecretsError(f"{location}: {e}") from e

        return substituted
    if isinstance(value, dict):
        return {
            k: _substitute_env(v, strict=strict, path=(*path, str(k))) for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _substitute_env(v, strict=strict, path=(*path, str(i))) for i, v in enumerate(value)
        ]
    return value


def _read_raw(path: Path) -> dict[str, Any]:
    """Parse a YAML or JSON file into a dict (no validation)."""
    if not path.exists():
        raise ConfigNotFoundError(f"config file not found: {path}")
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigValidationError(f"unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ConfigValidationError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def _resolve_inheritance(
    data: dict[str, Any],
    base_dir: Path,
    self_path: Path,
    visited: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Apply ``extends:`` chain — child values override parent recursively.

    Each leaf value in the merged dict is wrapped as a
    ``(value, ConfigSource)`` tuple via the helpers in
    :mod:`sentinel.config.source` so the loader can later report which
    file each field came from.

    Args:
        data: Raw dict from the current config file.
        base_dir: Directory used to resolve a relative ``extends:`` path.
        self_path: Absolute path of the current config file.
        visited: Tuple of resolved file paths seen so far in the
            inheritance walk. Used for cycle detection.

    Raises:
        ConfigCircularInheritanceError: When ``extends:`` chains form a
            cycle (``a → b → a`` or longer).
        ConfigNotFoundError: When the parent file does not exist.
    """
    self_resolved = self_path.resolve()
    if self_resolved in visited:
        chain = " → ".join(p.name for p in (*visited, self_resolved))
        raise ConfigCircularInheritanceError(f"circular config inheritance detected: {chain}")

    new_visited = (*visited, self_resolved)

    extends = data.get("extends")
    if extends is None:
        return annotate({k: v for k, v in data.items() if k != "extends"}, self_resolved)

    parent_path = (base_dir / extends).resolve()
    if parent_path in new_visited:
        chain = " → ".join(p.name for p in (*new_visited, parent_path))
        raise ConfigCircularInheritanceError(f"circular config inheritance detected: {chain}")

    parent_raw = _read_raw(parent_path)
    parent_resolved = _resolve_inheritance(
        parent_raw,
        parent_path.parent,
        parent_path,
        new_visited,
    )

    child_annotated = annotate(
        {k: v for k, v in data.items() if k != "extends"},
        self_resolved,
    )
    return merge_with_sources(parent_resolved, child_annotated, self_resolved)


class ConfigLoader:
    """Loads, validates, and caches Sentinel config files.

    Example:
        >>> loader = ConfigLoader("sentinel.yaml")
        >>> config = loader.load()
        >>> assert config.model.name == "claims_fraud_v2"

    Args:
        path: Path to the config file (YAML or JSON).
        strict_env: When True, fail loudly if any ``${VAR}`` token is
            unresolved. Set by ``sentinel validate --strict``.
        verify_signature: When True, look for a sidecar ``<path>.sig``
            file and verify its HMAC against the resolved config using
            ``signature_keystore``. Raises
            :class:`ConfigSignatureError` on missing, malformed, or
            mismatched signatures.
        signature_keystore: A
            :class:`~sentinel.foundation.audit.keystore.BaseKeystore`
            providing the signing key. Required when
            ``verify_signature`` is True.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        strict_env: bool = False,
        verify_signature: bool = False,
        signature_keystore: BaseKeystore | None = None,
    ):
        self.path = Path(path)
        self.strict_env = strict_env
        self.verify_signature = verify_signature
        self.signature_keystore = signature_keystore
        self._cached: SentinelConfig | None = None
        self._source_map: SourceMap | None = None
        self._resolved_payload: dict[str, Any] | None = None
        self._signature_verified: bool = False
        if verify_signature and signature_keystore is None:
            raise ConfigSignatureError("verify_signature=True requires a signature_keystore")

    @property
    def source_map(self) -> SourceMap | None:
        """Return the source map built during the most recent ``load()``."""
        return self._source_map

    @property
    def resolved_payload(self) -> dict[str, Any] | None:
        """The fully-resolved (post-extends, post-env) payload from the last load.

        Used by the signing CLI to compute a signature over exactly the
        bytes the loader would validate, so signed configs survive
        ``extends:`` chains and ``${VAR}`` substitution.
        """
        return self._resolved_payload

    @property
    def signature_verified(self) -> bool:
        """True iff the most recent ``load()`` verified a valid signature."""
        return self._signature_verified

    def load(self, force: bool = False) -> SentinelConfig:
        """Load and validate the config file. Caches by default."""
        if self._cached is not None and not force:
            return self._cached
        raw = _read_raw(self.path)
        annotated = _resolve_inheritance(raw, self.path.parent, self.path)
        clean, source_map = harvest(annotated)
        substituted = _substitute_env(clean, strict=self.strict_env)
        self._source_map = source_map
        self._resolved_payload = substituted
        if self.verify_signature:
            self._verify_signature_against(substituted)
        try:
            cfg = SentinelConfig.model_validate(substituted)
        except ValidationError as e:
            raise ConfigValidationError(self._format_validation_error(e, source_map)) from e
        log.info(
            "config.loaded",
            path=str(self.path),
            model=cfg.model.name,
            domain=cfg.model.domain,
            signature_verified=self._signature_verified,
        )
        self._cached = cfg
        return cfg

    def _verify_signature_against(self, payload: dict[str, Any]) -> None:
        """Verify the sidecar signature against ``payload`` (or raise)."""
        assert self.signature_keystore is not None  # narrowed by load()
        signature = read_signature_file(self.path)
        try:
            key = self.signature_keystore.get_key()
        except Exception as e:
            raise ConfigSignatureError(f"could not load signing key for {self.path}: {e}") from e
        if not verify_config(payload, signature, key):
            raise ConfigSignatureError(
                f"config signature does not match {self.path} "
                f"(signed by key fingerprint {signature.key_fingerprint})"
            )
        self._signature_verified = True

    @staticmethod
    def _format_validation_error(
        err: ValidationError,
        source_map: SourceMap | None = None,
    ) -> str:
        """Format Pydantic errors into a single human-friendly message.

        When a :class:`SourceMap` is available the originating file is
        appended to each line so multi-level ``extends:`` chains are
        easy to debug.
        """
        lines = ["config validation failed:"]
        for e in err.errors():
            loc = ".".join(str(p) for p in e["loc"])
            line = f"  - {loc}: {e['msg']}"
            if source_map is not None:
                src = source_map.lookup(e["loc"])
                if src is not None:
                    line = f"{line} [from {src.display()}]"
            lines.append(line)
        return "\n".join(lines)


def load_config(
    path: str | Path,
    *,
    strict_env: bool = False,
    verify_signature: bool = False,
    signature_keystore: BaseKeystore | None = None,
) -> SentinelConfig:
    """Convenience wrapper around :meth:`ConfigLoader.load`.

    Args:
        path: Path to the config file.
        strict_env: When True, raise :class:`ConfigMissingEnvVarError`
            on any unresolved ``${VAR}`` token. Defaults to False so
            existing call-sites stay backward compatible.
        verify_signature: When True, verify a sidecar
            ``<path>.sig`` file against ``signature_keystore``.
        signature_keystore: Source of the HMAC signing key. Required
            when ``verify_signature`` is True.
    """
    return ConfigLoader(
        path,
        strict_env=strict_env,
        verify_signature=verify_signature,
        signature_keystore=signature_keystore,
    ).load()


__all__ = [
    "ConfigLoader",
    "ConfigSignature",
    "ConfigSource",
    "SourceMap",
    "load_config",
]

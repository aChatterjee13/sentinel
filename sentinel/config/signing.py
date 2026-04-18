"""Detached HMAC signatures for Sentinel config files.

A signed config is a normal YAML/JSON file plus a sidecar JSON file
holding an HMAC-SHA256 over the **canonical** form of the resolved
config (after ``extends:`` resolution and ``${VAR}`` substitution). The
canonical form sorts keys and strips volatile fields (the cached source
map) so two semantically identical configs always produce the same
signature, regardless of YAML key order, indentation, or comments.

This is the config-side analogue of the audit hash chain in
``sentinel/foundation/audit/integrity.py`` — both use HMAC-SHA256 with
keys provided by the same :class:`BaseKeystore` abstraction so workstream
#2 can plug Azure Key Vault in once and have signed configs and signed
audit trails benefit at the same time.

Example:
    >>> from sentinel.config.signing import sign_config, verify_config
    >>> data = {"version": "1.0", "model": {"name": "demo", "domain": "tabular"}}
    >>> key = b"this-is-a-very-strong-key-32bytes!!"
    >>> sig = sign_config(data, key)
    >>> verify_config(data, sig, key)
    True
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.exceptions import ConfigSignatureError
from sentinel.foundation.audit.keystore import key_fingerprint

#: Sidecar files use this fixed extension so tooling can find them.
SIGNATURE_SUFFIX = ".sig"

#: Volatile keys stripped before canonicalisation. They never appear in
#: real configs — they're internal book-keeping injected by the loader.
_VOLATILE_KEYS = frozenset({"__source__", "__sources__"})


class ConfigSignature(BaseModel):
    """A detached signature for a Sentinel config file.

    Attributes:
        version: Format version. Bumped if the canonicalisation rules
            ever change so old signatures stay verifiable.
        algorithm: Always ``hmac-sha256`` today.
        signature: Hex-encoded HMAC-SHA256 of the canonical config.
        digest: Hex-encoded SHA-256 of the canonical config (without
            the key). Useful as a content-addressable identifier even
            when the verifier doesn't have the key.
        signed_at: ISO-8601 UTC timestamp at signing time.
        key_fingerprint: First 8 hex chars of SHA-256(key) — safe to
            log, lets operators correlate "which key signed this".
    """

    version: int = 1
    algorithm: str = "hmac-sha256"
    signature: str
    digest: str
    signed_at: str
    key_fingerprint: str = Field(min_length=8, max_length=8)

    model_config = ConfigDict(extra="forbid")


def canonicalise_config(data: dict[str, Any]) -> bytes:
    """Reduce a resolved config dict to deterministic bytes.

    The transformation:

    1. Recursively strips :data:`_VOLATILE_KEYS` (loader source-map
       artifacts that should never be part of the signed payload).
    2. Serialises with sorted keys, no extra whitespace, and a
       UTF-8 encoding so two semantically identical configs always
       produce the exact same bytes regardless of YAML formatting.

    Args:
        data: A resolved config dict (after ``extends:`` resolution and
            env-var substitution).

    Returns:
        Canonical UTF-8 bytes ready to feed into HMAC.
    """
    stripped = _strip_volatile(data)
    return json.dumps(
        stripped,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _strip_volatile(value: Any) -> Any:
    """Recursively remove volatile bookkeeping keys from a config tree."""
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_volatile(v) for v in value]
    return value


def _json_default(obj: Any) -> Any:
    """JSON fallback for the few non-primitive types configs may carry."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"cannot canonicalise type {type(obj).__name__}")


def sign_config(data: dict[str, Any], key: bytes) -> ConfigSignature:
    """Compute a :class:`ConfigSignature` over a canonicalised config.

    Args:
        data: A resolved config dict.
        key: Raw signing-key bytes (typically from a
            :class:`~sentinel.foundation.audit.keystore.BaseKeystore`).

    Returns:
        A populated :class:`ConfigSignature` ready to write to disk.
    """
    payload = canonicalise_config(data)
    signature = hmac.new(key, payload, hashlib.sha256).hexdigest()
    digest = hashlib.sha256(payload).hexdigest()
    return ConfigSignature(
        signature=signature,
        digest=digest,
        signed_at=datetime.now(timezone.utc).isoformat(),
        key_fingerprint=key_fingerprint(key),
    )


def verify_config(
    data: dict[str, Any],
    signature: ConfigSignature,
    key: bytes,
) -> bool:
    """Verify a :class:`ConfigSignature` against a config + key.

    Uses :func:`hmac.compare_digest` so a mismatched signature does not
    leak timing information about how many bytes matched. The verifier
    also checks ``key_fingerprint`` first to fail fast (and with a
    clearer error) when the wrong key is supplied.

    Args:
        data: The resolved config dict to verify.
        signature: A previously generated :class:`ConfigSignature`.
        key: Raw signing-key bytes.

    Returns:
        True if and only if the signature is valid for ``(data, key)``.
    """
    if signature.algorithm != "hmac-sha256":
        return False
    if signature.key_fingerprint != key_fingerprint(key):
        return False
    payload = canonicalise_config(data)
    expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.encode("ascii"), signature.signature.encode("ascii"))


def write_signature_file(path: str | Path, signature: ConfigSignature) -> Path:
    """Write a signature to a sidecar JSON file.

    Args:
        path: Either the config path (``foo.yaml``) — the sidecar will
            be ``foo.yaml.sig`` — or a fully-specified ``.sig`` path.
        signature: The :class:`ConfigSignature` to persist.

    Returns:
        The path that was written.
    """
    sig_path = _resolve_sig_path(path)
    sig_path.write_text(signature.model_dump_json(indent=2) + "\n")
    return sig_path


def read_signature_file(path: str | Path) -> ConfigSignature:
    """Read a sidecar signature file.

    Args:
        path: Either a config path or its corresponding ``.sig`` file.

    Raises:
        ConfigSignatureError: When the file does not exist or is not
            a valid :class:`ConfigSignature` document.
    """
    sig_path = _resolve_sig_path(path)
    if not sig_path.exists():
        raise ConfigSignatureError(f"signature file not found: {sig_path}")
    try:
        raw = json.loads(sig_path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigSignatureError(f"signature file is not valid JSON: {sig_path}") from e
    try:
        return ConfigSignature.model_validate(raw)
    except Exception as e:
        raise ConfigSignatureError(f"signature file is malformed: {sig_path}: {e}") from e


def _resolve_sig_path(path: str | Path) -> Path:
    """Normalise ``foo.yaml`` → ``foo.yaml.sig`` (and leave ``.sig`` paths alone)."""
    p = Path(path)
    if p.suffix == SIGNATURE_SUFFIX:
        return p
    return p.with_name(p.name + SIGNATURE_SUFFIX)


__all__ = [
    "SIGNATURE_SUFFIX",
    "ConfigSignature",
    "canonicalise_config",
    "read_signature_file",
    "sign_config",
    "verify_config",
    "write_signature_file",
]

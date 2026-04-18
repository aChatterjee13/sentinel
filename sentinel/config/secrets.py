"""Helpers for handling secrets in Sentinel configuration.

Sentinel uses :class:`pydantic.SecretStr` for every field that holds a
credential (webhook URLs, routing keys, basic-auth passwords, API tokens).
This module provides the two helpers every consumer needs:

``unwrap``
    Pull the plaintext out of a ``SecretStr`` at the exact point where
    the value is needed (HTTP call, SMTP login, Basic-auth compare).
    Accepts plain strings too so tests can instantiate channels with
    raw strings without caring about the wrapper.

``masked_dump``
    Serialise a :class:`~sentinel.config.schema.SentinelConfig` (or any
    Pydantic model) with every secret rendered as ``"<REDACTED>"``. The
    CLI ``sentinel config show`` command uses this to print the resolved
    config without ever leaking a credential to logs or terminals.

The design rule is "plaintext never lives in a long-lived attribute":
we keep ``SecretStr`` on the schema, we keep ``SecretStr`` on the
``SentinelConfig`` object, we only unwrap at the boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SecretStr

#: Type alias — use this when declaring secret fields on sub-schemas.
Secret = SecretStr | None

#: Sentinel token rendered in place of a secret in masked dumps.
REDACTED = "<REDACTED>"


def unwrap(secret: SecretStr | str | None) -> str | None:
    """Return the plaintext of a secret-bearing value.

    Accepts ``None``, a bare ``str`` (for tests and programmatic use),
    or a :class:`pydantic.SecretStr` (the normal schema-driven path).
    Always returns ``None`` when the input is falsy so callers can do
    ``if not unwrap(cfg.webhook_url): disable()`` without guarding.

    Args:
        secret: The wrapped (or unwrapped) secret.

    Returns:
        The plaintext string, or ``None`` when the input is ``None`` or
        an empty :class:`SecretStr`.

    Example:
        >>> from pydantic import SecretStr
        >>> unwrap(SecretStr("hunter2"))
        'hunter2'
        >>> unwrap("hunter2")
        'hunter2'
        >>> unwrap(None) is None
        True
    """
    if secret is None:
        return None
    if isinstance(secret, SecretStr):
        value = secret.get_secret_value()
        return value or None
    return secret or None


def _mask_value(value: Any) -> Any:
    """Recursively walk a dumped structure and redact any SecretStr."""
    if isinstance(value, SecretStr):
        return REDACTED if value.get_secret_value() else None
    if isinstance(value, dict):
        return {k: _mask_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_mask_value(v) for v in value)
    return value


def masked_dump(model: BaseModel, *, unmask: bool = False) -> dict[str, Any]:
    """Return a dict serialisation of ``model`` with secrets redacted.

    Pydantic v2's default ``model_dump()`` returns :class:`SecretStr`
    instances as objects (in python mode) or ``'**********'`` (in JSON
    mode). Neither is ideal for the CLI: we want the field *names* to
    be visible but the *values* replaced with a clear redaction
    marker. This helper walks the dumped tree and replaces every
    ``SecretStr`` with ``"<REDACTED>"`` (or its unwrapped value when
    ``unmask=True``).

    Args:
        model: Any Pydantic model — typically ``SentinelConfig``.
        unmask: When ``True``, return the plaintext instead of
            ``"<REDACTED>"``. Used by ``sentinel config show --unmask``
            (which is audit-logged).

    Returns:
        A JSON-serialisable dict.

    Example:
        >>> from sentinel.config.schema import ChannelConfig
        >>> masked_dump(ChannelConfig(type="slack", webhook_url="https://x"))
        {'type': 'slack', 'webhook_url': '<REDACTED>', ...}
    """
    dumped = model.model_dump(by_alias=True)
    if unmask:
        result: dict[str, Any] = _unmask_value(dumped)
        return result
    masked: dict[str, Any] = _mask_value(dumped)
    return masked


def _unmask_value(value: Any) -> Any:
    """Recursively unwrap any SecretStr instances into plaintext."""
    if isinstance(value, SecretStr):
        return value.get_secret_value() or None
    if isinstance(value, dict):
        return {k: _unmask_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unmask_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_unmask_value(v) for v in value)
    return value

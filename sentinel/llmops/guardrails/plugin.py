"""Plugin guardrail — dynamically loads a user-supplied guardrail class.

The class is imported at construction time from the configured module
path.  It must either extend :class:`BaseGuardrail` directly or expose a
``check(content, context) -> GuardrailResult`` method.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

import structlog

from sentinel.core.exceptions import SentinelError
from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

log = structlog.get_logger(__name__)

TRUSTED_MODULE_PREFIXES: tuple[str, ...] = ("sentinel.",)
"""Default prefixes that are allowed for plugin guardrail imports."""

_BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "ctypes",
        "importlib",
        "builtins",
        "code",
        "codeop",
        "compileall",
        "runpy",
    }
)
"""Stdlib modules that must never be loaded as plugins."""


class PluginLoadError(SentinelError):
    """Raised when a plugin guardrail cannot be loaded."""


class PluginGuardrail(BaseGuardrail):
    """A guardrail backed by a dynamically loaded Python class.

    The target class is resolved via ``importlib`` from *module* and
    *class_name*.  It must satisfy one of:

    1. Extend :class:`BaseGuardrail` and accept ``**config`` in its
       constructor.
    2. Expose a ``check(content: str, context: dict | None) ->
       GuardrailResult`` method (duck-typing).

    Args:
        module: Fully-qualified dotted module path (e.g.
            ``"mycompany.guardrails.custom"``).
        class_name: Name of the class to import from *module*.
        action: What to do when the guardrail triggers.
        config: Arbitrary keyword arguments forwarded to the loaded
            class constructor.

    Raises:
        PluginLoadError: If the module or class cannot be imported, or
            the class does not expose a ``check`` method.

    Example:
        >>> g = PluginGuardrail(
        ...     module="sentinel.llmops.guardrails.toxicity",
        ...     class_name="ToxicityGuardrail",
        ...     action="warn",
        ...     config={"threshold": 0.5},
        ... )
        >>> result = g.check("hello world")
        >>> result.passed
        True
    """

    name: str = "plugin"
    direction: Literal["input", "output", "both"] = "both"

    def __init__(
        self,
        *,
        module: str,
        class_name: str,
        action: Literal["block", "warn", "redact"] = "warn",
        config: dict[str, Any] | None = None,
        trusted_prefixes: tuple[str, ...] | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(action=action, **kwargs)
        self._trusted_prefixes = tuple(trusted_prefixes) if trusted_prefixes else TRUSTED_MODULE_PREFIXES
        self._inner = self._load(module, class_name, config or {}, self._trusted_prefixes)

    @staticmethod
    def _load(
        module: str,
        class_name: str,
        config: dict[str, Any],
        trusted_prefixes: tuple[str, ...],
    ) -> BaseGuardrail:
        """Import the target class and instantiate it.

        Args:
            module: Dotted module path.
            class_name: Class to import from *module*.
            config: Keyword arguments for the constructor.
            trusted_prefixes: Allowed module prefix whitelist.

        Returns:
            An instance of the loaded class.

        Raises:
            PluginLoadError: On any import or instantiation failure, or
                if the module is not in the trusted prefix list.
        """
        # Block dangerous stdlib modules.
        top_level = module.split(".")[0]
        if top_level in _BLOCKED_MODULES:
            raise PluginLoadError(
                f"Module '{module}' is blocked for security reasons."
            )

        # Enforce trusted prefix allowlist.
        if not any(module.startswith(prefix) for prefix in trusted_prefixes):
            raise PluginLoadError(
                f"Module '{module}' not in trusted prefixes: {trusted_prefixes}. "
                f"Add the prefix to 'trusted_prefixes' in guardrail config."
            )

        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            raise PluginLoadError(f"Cannot import module '{module}': {exc}") from exc

        cls = getattr(mod, class_name, None)
        if cls is None:
            raise PluginLoadError(f"Module '{module}' has no attribute '{class_name}'")

        if not callable(cls):
            raise PluginLoadError(f"'{module}.{class_name}' is not callable")

        try:
            instance = cls(**config)
        except TypeError as exc:
            raise PluginLoadError(f"Failed to instantiate '{module}.{class_name}': {exc}") from exc

        if not hasattr(instance, "check") or not callable(instance.check):
            raise PluginLoadError(f"'{module}.{class_name}' does not expose a check() method")

        log.info(
            "plugin_guardrail.loaded",
            module=module,
            class_name=class_name,
        )
        return instance  # type: ignore[return-value]

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Delegate to the loaded plugin's ``check`` method.

        Args:
            content: The text being checked.
            context: Optional context dict forwarded to the plugin.

        Returns:
            A :class:`GuardrailResult` from the plugin.
        """
        return self._inner.check(content, context)

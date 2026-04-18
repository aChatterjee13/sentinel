"""Core SDK primitives — client, types, exceptions, hooks."""

from sentinel.core.client import SentinelClient
from sentinel.core.exceptions import SentinelError
from sentinel.core.hooks import HookManager, HookType

__all__ = ["HookManager", "HookType", "SentinelClient", "SentinelError"]

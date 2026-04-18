"""The :class:`Principal` value object.

A principal represents an authenticated (or anonymous) caller as
seen by the dashboard. It is intentionally cheap to construct,
hashable, and immutable so that middlewares can stash it on
``request.state`` without worrying about ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """An authenticated caller and the permissions they hold.

    Attributes:
        username: Stable identifier for the caller. ``"anonymous"``
            when authentication is disabled.
        roles: Frozen set of role names assigned to the caller.
        permissions: Fully resolved frozen set of permission strings
            (after role hierarchy expansion). Use this for permission
            checks rather than walking ``roles`` every request.
        auth_mode: One of ``"none"``, ``"basic"``, ``"bearer"``,
            describing how the caller authenticated. Useful for
            audit-trail logging and CSRF exemption decisions.
    """

    username: str
    roles: frozenset[str] = field(default_factory=frozenset)
    permissions: frozenset[str] = field(default_factory=frozenset)
    auth_mode: str = "none"

    def has_permission(self, permission: str) -> bool:
        """Return True when the principal holds ``permission``.

        The wildcard ``"*"`` permission grants everything — used by
        the default ``admin`` role.
        """
        if "*" in self.permissions:
            return True
        return permission in self.permissions

    @property
    def is_anonymous(self) -> bool:
        return self.username == "anonymous" and self.auth_mode == "none"


ANONYMOUS_PRINCIPAL = Principal(
    username="anonymous",
    roles=frozenset(),
    permissions=frozenset(),
    auth_mode="none",
)
"""Sentinel principal used when ``dashboard.server.auth == "none"``."""

"""Notification engine and channels."""

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications.channels.email import EmailChannel
from sentinel.action.notifications.channels.pagerduty import PagerDutyChannel
from sentinel.action.notifications.channels.slack import SlackChannel
from sentinel.action.notifications.channels.teams import TeamsChannel
from sentinel.action.notifications.channels.webhook import WebhookChannel
from sentinel.action.notifications.engine import NotificationEngine
from sentinel.action.notifications.policies import AlertPolicyEngine

# Channel registry
CHANNEL_REGISTRY: dict[str, type[BaseChannel]] = {
    "slack": SlackChannel,
    "teams": TeamsChannel,
    "pagerduty": PagerDutyChannel,
    "email": EmailChannel,
    "webhook": WebhookChannel,
}


def register_channel(name: str, cls: type[BaseChannel]) -> None:
    """Plug-in API: register a custom notification channel."""
    CHANNEL_REGISTRY[name] = cls


__all__ = [
    "CHANNEL_REGISTRY",
    "AlertPolicyEngine",
    "BaseChannel",
    "EmailChannel",
    "NotificationEngine",
    "PagerDutyChannel",
    "SlackChannel",
    "TeamsChannel",
    "WebhookChannel",
    "register_channel",
]

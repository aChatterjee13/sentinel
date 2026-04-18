"""Notification channels."""

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications.channels.email import EmailChannel
from sentinel.action.notifications.channels.pagerduty import PagerDutyChannel
from sentinel.action.notifications.channels.slack import SlackChannel
from sentinel.action.notifications.channels.teams import TeamsChannel
from sentinel.action.notifications.channels.webhook import WebhookChannel

__all__ = [
    "BaseChannel",
    "EmailChannel",
    "PagerDutyChannel",
    "SlackChannel",
    "TeamsChannel",
    "WebhookChannel",
]

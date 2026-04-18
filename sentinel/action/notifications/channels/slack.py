"""Slack notification channel."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.config.secrets import unwrap
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult

log = structlog.get_logger(__name__)

_COLOR_MAP = {
    AlertSeverity.INFO: "#36a64f",
    AlertSeverity.WARNING: "#ffae42",
    AlertSeverity.HIGH: "#ff8c00",
    AlertSeverity.CRITICAL: "#ff0000",
}

_EMOJI_MAP = {
    AlertSeverity.INFO: ":information_source:",
    AlertSeverity.WARNING: ":warning:",
    AlertSeverity.HIGH: ":rotating_light:",
    AlertSeverity.CRITICAL: ":fire:",
}


class SlackChannel(BaseChannel):
    """Posts alerts to a Slack incoming webhook.

    Uses `slack-sdk` if installed (richer formatting), otherwise falls back
    to a stdlib HTTP POST so the channel works with zero extras installed.
    """

    name = "slack"

    def __init__(self, **config: Any):
        super().__init__(**config)
        # Unwrap once at the channel boundary; the resolved string never
        # leaves this object and is never re-emitted in logs.
        self.webhook_url: str | None = unwrap(config.get("webhook_url"))
        self.channel = config.get("channel")
        if not self.webhook_url:
            log.warning("slack.no_webhook", message="Slack channel disabled — no webhook_url")
            self.enabled = False

    def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(channel=self.name, delivered=False, error="channel disabled")

        payload = self._build_payload(alert)
        try:
            return self._post_via_sdk(payload, alert) or self._post_via_stdlib(payload)
        except Exception as e:
            log.error("slack.send_failed", error=str(e))
            return DeliveryResult(channel=self.name, delivered=False, error=str(e))

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        emoji = _EMOJI_MAP.get(alert.severity, ":bell:")
        color = _COLOR_MAP.get(alert.severity, "#cccccc")
        attachment = {
            "color": color,
            "title": f"{emoji} {alert.title}",
            "text": alert.body,
            "fields": [
                {"title": "Model", "value": alert.model_name, "short": True},
                {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                {"title": "Source", "value": alert.source, "short": True},
                {"title": "Time", "value": alert.timestamp.isoformat(), "short": True},
            ],
            "footer": "Sentinel",
            "ts": int(alert.timestamp.timestamp()),
        }
        for k, v in list(alert.payload.items())[:8]:
            attachment["fields"].append({"title": str(k), "value": str(v), "short": True})

        body: dict[str, Any] = {"attachments": [attachment]}
        if self.channel:
            body["channel"] = self.channel
        return body

    def _post_via_sdk(self, payload: dict[str, Any], alert: Alert) -> DeliveryResult | None:
        try:
            from slack_sdk.webhook import WebhookClient  # type: ignore[import-not-found]
        except ImportError:
            return None
        client = WebhookClient(self.webhook_url)  # type: ignore[arg-type]
        response = client.send(
            text=alert.title,
            attachments=payload["attachments"],
        )
        return DeliveryResult(
            channel=self.name,
            delivered=response.status_code == 200,
            response={"status": response.status_code, "body": response.body},
        )

    def _post_via_stdlib(self, payload: dict[str, Any]) -> DeliveryResult:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.webhook_url,  # type: ignore[arg-type]
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as resp:
                return DeliveryResult(
                    channel=self.name,
                    delivered=200 <= resp.status < 300,
                    response={"status": resp.status},
                )
        except error.URLError as e:
            return DeliveryResult(channel=self.name, delivered=False, error=str(e))

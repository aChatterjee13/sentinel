"""Microsoft Teams notification channel."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.config.secrets import unwrap
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult

log = structlog.get_logger(__name__)

_THEME_COLORS = {
    AlertSeverity.INFO: "0078D4",
    AlertSeverity.WARNING: "FFA500",
    AlertSeverity.HIGH: "FF6347",
    AlertSeverity.CRITICAL: "FF0000",
}


class TeamsChannel(BaseChannel):
    """Posts adaptive cards to a Microsoft Teams incoming webhook."""

    name = "teams"

    def __init__(self, **config: Any):
        super().__init__(**config)
        self.webhook_url: str | None = unwrap(config.get("webhook_url"))
        if not self.webhook_url:
            log.warning("teams.no_webhook")
            self.enabled = False

    def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(channel=self.name, delivered=False, error="channel disabled")
        payload = self._build_message_card(alert)
        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.webhook_url,  # type: ignore[arg-type]
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=10) as resp:
                return DeliveryResult(
                    channel=self.name,
                    delivered=200 <= resp.status < 300,
                    response={"status": resp.status},
                )
        except (error.URLError, OSError) as e:
            return DeliveryResult(channel=self.name, delivered=False, error=str(e))

    def _build_message_card(self, alert: Alert) -> dict[str, Any]:
        return {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": _THEME_COLORS.get(alert.severity, "808080"),
            "summary": alert.title,
            "title": f"[{alert.severity.value.upper()}] {alert.title}",
            "sections": [
                {
                    "activityTitle": alert.model_name,
                    "activitySubtitle": alert.source,
                    "text": alert.body,
                    "facts": [
                        {"name": "Severity", "value": alert.severity.value},
                        {"name": "Time", "value": alert.timestamp.isoformat()},
                        *[
                            {"name": str(k), "value": str(v)}
                            for k, v in list(alert.payload.items())[:8]
                        ],
                    ],
                }
            ],
        }

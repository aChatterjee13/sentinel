"""Generic JSON webhook channel — works for Discord, custom endpoints, etc."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.config.secrets import unwrap
from sentinel.core.types import Alert, DeliveryResult

log = structlog.get_logger(__name__)


class WebhookChannel(BaseChannel):
    """POST a JSON-serialised alert to any HTTP endpoint."""

    name = "webhook"

    def __init__(self, **config: Any):
        super().__init__(**config)
        self.webhook_url: str | None = unwrap(config.get("webhook_url"))
        self.headers = config.get("headers", {"Content-Type": "application/json"})
        if not self.webhook_url:
            log.warning("webhook.no_url")
            self.enabled = False

    def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(channel=self.name, delivered=False, error="channel disabled")
        payload = alert.model_dump(mode="json")
        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.webhook_url,  # type: ignore[arg-type]
                data=data,
                headers=self.headers,
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

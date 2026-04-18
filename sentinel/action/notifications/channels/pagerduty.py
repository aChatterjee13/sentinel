"""PagerDuty Events API v2 channel."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.config.secrets import unwrap
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult

log = structlog.get_logger(__name__)

_PD_SEVERITY = {
    AlertSeverity.INFO: "info",
    AlertSeverity.WARNING: "warning",
    AlertSeverity.HIGH: "error",
    AlertSeverity.CRITICAL: "critical",
}


class PagerDutyChannel(BaseChannel):
    """Triggers PagerDuty incidents via the Events API v2."""

    name = "pagerduty"
    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(self, **config: Any):
        super().__init__(**config)
        self.routing_key: str | None = unwrap(config.get("routing_key"))
        self.severity_mapping = config.get("severity_mapping", {})
        if not self.routing_key:
            log.warning("pagerduty.no_routing_key")
            self.enabled = False

    def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(channel=self.name, delivered=False, error="channel disabled")

        severity = self.severity_mapping.get(
            alert.severity.value, _PD_SEVERITY.get(alert.severity, "warning")
        )
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint or alert.alert_id,
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": severity,
                "component": alert.model_name,
                "custom_details": {
                    "body": alert.body,
                    **{k: str(v) for k, v in alert.payload.items()},
                },
            },
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.EVENTS_URL,
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
            err_msg = str(e)
            if self.routing_key and self.routing_key in err_msg:
                err_msg = err_msg.replace(self.routing_key, "***REDACTED***")
            return DeliveryResult(channel=self.name, delivered=False, error=err_msg)

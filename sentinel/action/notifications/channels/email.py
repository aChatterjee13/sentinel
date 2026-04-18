"""SMTP email notification channel."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.config.secrets import unwrap
from sentinel.core.types import Alert, DeliveryResult

log = structlog.get_logger(__name__)


class EmailChannel(BaseChannel):
    """Sends alerts via SMTP. Requires server config in the channel block."""

    name = "email"

    def __init__(self, **config: Any):
        super().__init__(**config)
        self.smtp_host = config.get("smtp_host", "localhost")
        self.smtp_port = int(config.get("smtp_port", 587))
        self.username = config.get("username")
        # ``password`` may arrive as either a SecretStr (when surfaced via
        # the schema's ``extra="allow"`` slot in a future migration) or a
        # plain string (today's path). ``unwrap`` handles both safely.
        self.password: str | None = unwrap(config.get("password"))
        self.from_addr = config.get("from_addr", "sentinel@example.com")
        self.recipients = config.get("recipients", [])
        self.use_tls = config.get("use_tls", True)
        if not self.recipients:
            log.warning("email.no_recipients")
            self.enabled = False

    def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(channel=self.name, delivered=False, error="channel disabled")
        msg = EmailMessage()
        msg["Subject"] = f"[Sentinel {alert.severity.value.upper()}] {alert.title}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(self.format_message(alert))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            return DeliveryResult(channel=self.name, delivered=True)
        except (smtplib.SMTPException, OSError) as e:
            log.error("email.send_failed", error=str(e))
            return DeliveryResult(channel=self.name, delivered=False, error=str(e))

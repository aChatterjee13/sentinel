"""Human-in-the-loop approval gates for regulated retrain pipelines."""

from __future__ import annotations

import contextlib
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from sentinel.action.notifications.policies import parse_duration
from sentinel.config.schema import ApprovalConfig
from sentinel.core.exceptions import ApprovalTimeoutError


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


class ApprovalRequest(BaseModel):
    """A pending human approval request."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    model_name: str
    candidate_version: str
    champion_metrics: dict[str, float] = Field(default_factory=dict)
    challenger_metrics: dict[str, float] = Field(default_factory=dict)
    approvers: list[str] = Field(default_factory=list)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    decided_at: datetime | None = None
    comment: str | None = None


class ApprovalGate:
    """In-memory approval queue.

    Production systems should subclass this and persist requests to a
    database or workflow engine. The default in-memory store is sufficient
    for tests and single-process deployments.
    """

    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.timeout = parse_duration(config.timeout)
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()
        self._expiry_timer: threading.Timer | None = None
        self._start_expiry_timer()

    # ── Request creation ──────────────────────────────────────────

    def request(
        self,
        model_name: str,
        candidate_version: str,
        champion_metrics: dict[str, float] | None = None,
        challenger_metrics: dict[str, float] | None = None,
    ) -> ApprovalRequest:
        # Auto-approve when policy permits
        if self.config.mode == "auto":
            req = ApprovalRequest(
                model_name=model_name,
                candidate_version=candidate_version,
                champion_metrics=champion_metrics or {},
                challenger_metrics=challenger_metrics or {},
                expires_at=datetime.now(timezone.utc) + self.timeout,
                status=ApprovalStatus.AUTO_APPROVED,
                decided_at=datetime.now(timezone.utc),
                decided_by="auto",
            )
            with self._lock:
                self._requests[req.request_id] = req
            return req

        if self.config.mode == "hybrid" and self._meets_auto_promote(
            champion_metrics or {}, challenger_metrics or {}
        ):
            req = ApprovalRequest(
                model_name=model_name,
                candidate_version=candidate_version,
                champion_metrics=champion_metrics or {},
                challenger_metrics=challenger_metrics or {},
                expires_at=datetime.now(timezone.utc) + self.timeout,
                status=ApprovalStatus.AUTO_APPROVED,
                decided_at=datetime.now(timezone.utc),
                decided_by="auto",
            )
            with self._lock:
                self._requests[req.request_id] = req
            return req

        req = ApprovalRequest(
            model_name=model_name,
            candidate_version=candidate_version,
            champion_metrics=champion_metrics or {},
            challenger_metrics=challenger_metrics or {},
            approvers=self.config.approvers,
            expires_at=datetime.now(timezone.utc) + self.timeout,
        )
        with self._lock:
            self._requests[req.request_id] = req
        return req

    # ── Decisions ─────────────────────────────────────────────────

    def approve(self, request_id: str, by: str, comment: str | None = None) -> ApprovalRequest:
        with self._lock:
            req = self._get(request_id)
            req = req.model_copy(
                update={
                    "status": ApprovalStatus.APPROVED,
                    "decided_by": by,
                    "decided_at": datetime.now(timezone.utc),
                    "comment": comment,
                }
            )
            self._requests[request_id] = req
        return req

    def reject(self, request_id: str, by: str, comment: str | None = None) -> ApprovalRequest:
        with self._lock:
            req = self._get(request_id)
            req = req.model_copy(
                update={
                    "status": ApprovalStatus.REJECTED,
                    "decided_by": by,
                    "decided_at": datetime.now(timezone.utc),
                    "comment": comment,
                }
            )
            self._requests[request_id] = req
        return req

    def expire_pending(self) -> int:
        """Mark expired requests as TIMEOUT. Returns count expired."""
        now = datetime.now(timezone.utc)
        n = 0
        with self._lock:
            for rid, req in list(self._requests.items()):
                if req.status == ApprovalStatus.PENDING and now >= req.expires_at:
                    self._requests[rid] = req.model_copy(update={"status": ApprovalStatus.TIMEOUT})
                    n += 1
        return n

    def get(self, request_id: str) -> ApprovalRequest:
        with self._lock:
            return self._get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def wait_for(
        self, request_id: str, poll_interval: float = 1.0, timeout: float | None = None
    ) -> ApprovalRequest:
        """Block until the request is decided. Raises on timeout."""
        import time

        with self._lock:
            req = self._get(request_id)
        if timeout is not None:
            deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        else:
            deadline = req.expires_at
        while True:
            with self._lock:
                req = self._get(request_id)
            if req.status != ApprovalStatus.PENDING:
                return req
            if datetime.now(timezone.utc) >= deadline:
                raise ApprovalTimeoutError(f"approval request {request_id} timed out")
            time.sleep(poll_interval)

    # ── Internal ──────────────────────────────────────────────────

    def _get(self, request_id: str) -> ApprovalRequest:
        if request_id not in self._requests:
            raise KeyError(f"approval request {request_id} not found")
        return self._requests[request_id]

    def _meets_auto_promote(self, champion: dict[str, float], challenger: dict[str, float]) -> bool:
        """Check whether auto-promote criteria are met.

        Delegates to :class:`PromotionPolicy` to avoid duplicating the
        champion-challenger comparison logic.

        Args:
            champion: Current production model metrics.
            challenger: Candidate model metrics.

        Returns:
            ``True`` when the challenger satisfies the ``auto_promote_if`` rule.
        """
        from sentinel.action.deployment.promotion import PromotionPolicy

        rule = self.config.auto_promote_if
        if not rule:
            return False
        metric = rule.get("metric")
        improvement = rule.get("improvement_pct", 0.0)
        if metric is None:
            return False
        policy = PromotionPolicy(metric=metric, improvement_pct=improvement)
        return policy.should_promote(champion, challenger)

    def _start_expiry_timer(self) -> None:
        """Start a periodic timer to expire pending approvals."""
        self._expiry_timer = threading.Timer(60.0, self._tick_expiry)
        self._expiry_timer.daemon = True
        self._expiry_timer.start()

    def _tick_expiry(self) -> None:
        """Expire pending requests and reschedule."""
        with contextlib.suppress(Exception):
            self.expire_pending()
        if self._expiry_timer is not None:
            self._start_expiry_timer()

    def close(self) -> None:
        """Cancel the expiry timer."""
        if self._expiry_timer is not None:
            self._expiry_timer.cancel()
            self._expiry_timer = None

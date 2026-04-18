"""Unit tests for sentinel.action.retrain.approval.ApprovalGate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from sentinel.action.retrain.approval import ApprovalGate, ApprovalStatus
from sentinel.config.schema import ApprovalConfig
from sentinel.core.exceptions import ApprovalTimeoutError

# ── Helpers ────────────────────────────────────────────────────────


def _gate(
    mode: str = "human_in_loop",
    timeout: str = "48h",
    auto_promote_if: dict | None = None,
    approvers: list[str] | None = None,
) -> ApprovalGate:
    cfg = ApprovalConfig(
        mode=mode,
        timeout=timeout,
        auto_promote_if=auto_promote_if or {},
        approvers=approvers or ["admin@co.com"],
    )
    gate = ApprovalGate(cfg)
    # Ensure timer is cleaned up by tests
    return gate


def _close_gate(gate: ApprovalGate) -> None:
    """Ensure expiry timer is cancelled so tests don't leak threads."""
    gate.close()


# ── Tests ──────────────────────────────────────────────────────────


class TestAutoMode:
    """Requests under mode='auto' are immediately auto-approved."""

    def test_auto_mode_returns_auto_approved(self) -> None:
        gate = _gate(mode="auto")
        try:
            req = gate.request("m", "1.0")
            assert req.status == ApprovalStatus.AUTO_APPROVED
            assert req.decided_by == "auto"
        finally:
            _close_gate(gate)


class TestHumanInLoopMode:
    """Requests under mode='human_in_loop' stay pending until decided."""

    def test_request_creates_pending(self) -> None:
        gate = _gate(mode="human_in_loop")
        try:
            req = gate.request("m", "1.0", champion_metrics={"f1": 0.8})
            assert req.status == ApprovalStatus.PENDING
            assert req.model_name == "m"
            assert req.candidate_version == "1.0"
        finally:
            _close_gate(gate)

    def test_approve_transitions_to_approved(self) -> None:
        gate = _gate(mode="human_in_loop")
        try:
            req = gate.request("m", "1.0")
            approved = gate.approve(req.request_id, by="alice", comment="LGTM")
            assert approved.status == ApprovalStatus.APPROVED
            assert approved.decided_by == "alice"
            assert approved.comment == "LGTM"
        finally:
            _close_gate(gate)

    def test_reject_transitions_to_rejected(self) -> None:
        gate = _gate(mode="human_in_loop")
        try:
            req = gate.request("m", "1.0")
            rejected = gate.reject(req.request_id, by="bob")
            assert rejected.status == ApprovalStatus.REJECTED
            assert rejected.decided_by == "bob"
        finally:
            _close_gate(gate)


class TestHybridMode:
    """Hybrid auto-promotes when improvement criteria are met."""

    def test_hybrid_auto_promotes_when_criteria_met(self) -> None:
        gate = _gate(
            mode="hybrid",
            auto_promote_if={"metric": "f1", "improvement_pct": 2.0},
        )
        try:
            req = gate.request(
                "m",
                "2.0",
                champion_metrics={"f1": 0.80},
                challenger_metrics={"f1": 0.85},  # +6.25%
            )
            assert req.status == ApprovalStatus.AUTO_APPROVED
        finally:
            _close_gate(gate)

    def test_hybrid_stays_pending_when_criteria_not_met(self) -> None:
        gate = _gate(
            mode="hybrid",
            auto_promote_if={"metric": "f1", "improvement_pct": 10.0},
        )
        try:
            req = gate.request(
                "m",
                "2.0",
                champion_metrics={"f1": 0.80},
                challenger_metrics={"f1": 0.81},  # +1.25% < 10%
            )
            assert req.status == ApprovalStatus.PENDING
        finally:
            _close_gate(gate)

    def test_hybrid_auto_promotes_when_champion_is_zero(self) -> None:
        """When champion metric is 0, any positive challenger wins."""
        gate = _gate(
            mode="hybrid",
            auto_promote_if={"metric": "f1", "improvement_pct": 1.0},
        )
        try:
            req = gate.request(
                "m",
                "2.0",
                champion_metrics={"f1": 0.0},
                challenger_metrics={"f1": 0.1},
            )
            assert req.status == ApprovalStatus.AUTO_APPROVED
        finally:
            _close_gate(gate)


class TestExpiry:
    """Expiry timer and expire_pending behaviour."""

    def test_expire_pending_marks_old_requests_as_timeout(self) -> None:
        gate = _gate(mode="human_in_loop", timeout="1h")
        try:
            req = gate.request("m", "1.0")
            # Manually backdate the expiry so it's already past
            expired_req = req.model_copy(
                update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
            )
            gate._requests[req.request_id] = expired_req
            n = gate.expire_pending()
            assert n == 1
            assert gate.get(req.request_id).status == ApprovalStatus.TIMEOUT
        finally:
            _close_gate(gate)

    def test_expire_pending_does_not_touch_decided_requests(self) -> None:
        gate = _gate(mode="human_in_loop", timeout="1h")
        try:
            req = gate.request("m", "1.0")
            gate.approve(req.request_id, by="a")
            # Backdate
            updated = gate.get(req.request_id).model_copy(
                update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
            )
            gate._requests[req.request_id] = updated
            assert gate.expire_pending() == 0
        finally:
            _close_gate(gate)


class TestListAndGet:
    """list_pending, get, and unknown-id error paths."""

    def test_list_pending_returns_only_pending(self) -> None:
        gate = _gate(mode="human_in_loop")
        try:
            r1 = gate.request("m1", "1.0")
            r2 = gate.request("m2", "2.0")
            gate.approve(r1.request_id, by="a")
            pending = gate.list_pending()
            assert len(pending) == 1
            assert pending[0].request_id == r2.request_id
        finally:
            _close_gate(gate)

    def test_get_unknown_id_raises_key_error(self) -> None:
        gate = _gate()
        try:
            with pytest.raises(KeyError, match="not found"):
                gate.get("nonexistent")
        finally:
            _close_gate(gate)


class TestWaitFor:
    """wait_for returns immediately for decided requests."""

    def test_wait_for_returns_decided_request(self) -> None:
        gate = _gate(mode="human_in_loop")
        try:
            req = gate.request("m", "1.0")
            gate.approve(req.request_id, by="a")
            result = gate.wait_for(req.request_id)
            assert result.status == ApprovalStatus.APPROVED
        finally:
            _close_gate(gate)

    def test_wait_for_raises_on_expired(self) -> None:
        gate = _gate(mode="human_in_loop", timeout="1h")
        try:
            req = gate.request("m", "1.0")
            expired_req = req.model_copy(
                update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
            )
            gate._requests[req.request_id] = expired_req
            with pytest.raises(ApprovalTimeoutError, match="timed out"):
                gate.wait_for(req.request_id)
        finally:
            _close_gate(gate)


class TestExpiryTimer:
    """The background _start_expiry_timer / close lifecycle."""

    def test_close_cancels_timer(self) -> None:
        gate = _gate()
        assert gate._expiry_timer is not None
        gate.close()
        assert gate._expiry_timer is None

    def test_close_is_idempotent(self) -> None:
        gate = _gate()
        gate.close()
        gate.close()  # second call should not raise
        assert gate._expiry_timer is None

    @patch("sentinel.action.retrain.approval.threading.Timer")
    def test_start_expiry_timer_creates_daemon_timer(self, mock_timer_cls: MagicMock) -> None:
        """Verify the timer is started as daemon and with 60s interval."""
        mock_timer = MagicMock()
        mock_timer_cls.return_value = mock_timer
        gate = _gate()
        try:
            mock_timer_cls.assert_called_with(60.0, gate._tick_expiry)
            assert mock_timer.daemon is True
            mock_timer.start.assert_called_once()
        finally:
            gate.close()

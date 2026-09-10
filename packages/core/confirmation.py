"""ConfirmationRequest lifecycle — the single implementation of FR5 (UC3).

UC1 and UC2 both route destructive changes through here rather than duplicating
the check. UC3's exception paths are implemented exactly as specified:
  E1 no response before timeout  -> fail-safe: no action, re-prompt later.
  E2 user unreachable (NFR4)     -> queue for the next active window.
  Open issue: an unintelligible response defaults to no action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .decisions import ConfirmedDecision, Decision, _grant


class ConfirmationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    TIMED_OUT = "timed_out"
    QUEUED = "queued"  # user outside their active window (NFR4)


DEFAULT_TIMEOUT = timedelta(minutes=15)


@dataclass
class ConfirmationRequest:
    request_id: str
    decision: Decision
    created_at: datetime
    timeout_at: datetime
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    resolved_at: datetime | None = None
    prompt: str = ""

    @classmethod
    def open(
        cls,
        request_id: str,
        decision: Decision,
        now: datetime,
        timeout: timedelta = DEFAULT_TIMEOUT,
    ) -> "ConfirmationRequest":
        if not decision.requires_confirmation:
            raise ValueError("Only destructive decisions are gated by a ConfirmationRequest.")
        return cls(
            request_id=request_id,
            decision=decision,
            created_at=now,
            timeout_at=now + timeout,
            prompt=decision.describe(),
        )

    def confirm(self, now: datetime) -> ConfirmedDecision:
        """Grant the capability that unlocks the destructive action."""
        self._assert_pending(now)
        self.status = ConfirmationStatus.CONFIRMED
        self.resolved_at = now
        return _grant(self.decision, self.request_id, now)

    def decline(self, now: datetime) -> None:
        self._assert_pending(now)
        self.status = ConfirmationStatus.DECLINED
        self.resolved_at = now

    def resolve(self, confirmed: bool | None, now: datetime) -> ConfirmedDecision | None:
        """``confirmed=None`` means the response was unintelligible.

        UC3 open issue: an ambiguous confirmation must never be read as consent,
        so it is treated as a decline-for-now and re-prompted.
        """
        if confirmed is True:
            return self.confirm(now)
        self.decline(now)
        return None

    def expire_if_due(self, now: datetime) -> bool:
        """UC3 E1 — fail-safe. Returns True if this call expired the request."""
        if self.status is ConfirmationStatus.PENDING and now >= self.timeout_at:
            self.status = ConfirmationStatus.TIMED_OUT
            self.resolved_at = now
            return True
        return False

    def queue_for_active_window(self, next_window_start: datetime) -> None:
        """UC3 E2 — user unreachable outside their configured window (NFR4)."""
        self.status = ConfirmationStatus.QUEUED
        self.timeout_at = next_window_start + DEFAULT_TIMEOUT

    def _assert_pending(self, now: datetime) -> None:
        self.expire_if_due(now)
        if self.status is ConfirmationStatus.TIMED_OUT:
            # Covers both "it expires on this call" and "a late answer arrived for
            # a request that expired earlier". A stale yes is not consent.
            raise TimeoutError(f"Confirmation {self.request_id} timed out; no action taken.")
        if self.status not in (ConfirmationStatus.PENDING, ConfirmationStatus.QUEUED):
            raise ValueError(f"Confirmation {self.request_id} is already {self.status.value}.")


@dataclass
class ConfirmationStore:
    """In-memory registry of pending confirmations (Traveller 1 -- 0..* requests)."""

    _requests: dict[str, ConfirmationRequest] = field(default_factory=dict)
    _seq: int = 0

    def open(self, decision: Decision, now: datetime, timeout: timedelta = DEFAULT_TIMEOUT) -> ConfirmationRequest:
        self._seq += 1
        req = ConfirmationRequest.open(f"CR-{self._seq}", decision, now, timeout)
        self._requests[req.request_id] = req
        return req

    def get(self, request_id: str) -> ConfirmationRequest:
        return self._requests[request_id]

    def nth(self, index: int) -> ConfirmationRequest:
        """1-indexed access in creation order — used by the scenario runner."""
        return list(self._requests.values())[index - 1]

    def pending(self, now: datetime) -> list[ConfirmationRequest]:
        for req in self._requests.values():
            req.expire_if_due(now)
        return [r for r in self._requests.values() if r.status is ConfirmationStatus.PENDING]

    def all(self) -> list[ConfirmationRequest]:
        return list(self._requests.values())

    def count(self) -> int:
        return len(self._requests)

    def find_pending_for(self, booking_ref: str | None, target_event_id: str | None, now: datetime) -> ConfirmationRequest | None:
        """NFR2: a repeated notification must reuse the open request, not raise a second one."""
        for req in self.pending(now):
            same_ref = booking_ref is not None and req.decision.booking is not None and req.decision.booking.booking_ref == booking_ref
            same_target = target_event_id is not None and req.decision.target_event_id == target_event_id
            if same_ref or same_target:
                return req
        return None

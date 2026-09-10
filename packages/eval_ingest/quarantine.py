"""UC4 E2: a trajectory that fails validation is held, not dropped.

A silently discarded trajectory is the worst outcome available to an evaluation
platform: the agent looks better than it is, because the runs that did not fit
the contract are exactly the unusual ones. Every rejection is therefore kept
with its payload and its reasons, and is retrievable over the same API that
accepted the good ones.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class QuarantinedTrajectory:
    quarantine_id: str
    received_at: str
    errors: tuple[str, ...]
    payload: object
    source: str | None = None

    def as_dict(self) -> dict:
        return {
            "quarantine_id": self.quarantine_id,
            "received_at": self.received_at,
            "errors": list(self.errors),
            "payload": self.payload,
            "source": self.source,
        }


@dataclass
class Quarantine:
    """In-memory and bounded, for the same reason the capture buffer is.

    A producer emitting malformed trajectories in a loop must not be able to
    exhaust the ingest service's memory; the oldest entries age out and the
    total ever seen keeps being counted.
    """

    capacity: int = 1000
    _items: list[QuarantinedTrajectory] = field(default_factory=list)
    _total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, payload: object, errors: list[str], source: str | None = None) -> QuarantinedTrajectory:
        record = QuarantinedTrajectory(
            quarantine_id=str(uuid.uuid4()),
            received_at=datetime.now(timezone.utc).isoformat(),
            errors=tuple(errors),
            payload=payload,
            source=source,
        )
        with self._lock:
            self._items.append(record)
            self._total += 1
            while len(self._items) > self.capacity:
                self._items.pop(0)
        return record

    def all(self) -> list[QuarantinedTrajectory]:
        with self._lock:
            return list(self._items)

    def get(self, quarantine_id: str) -> QuarantinedTrajectory | None:
        with self._lock:
            for record in self._items:
                if record.quarantine_id == quarantine_id:
                    return record
        return None

    @property
    def total_ever(self) -> int:
        """Includes entries aged out of the bounded list, so the count never lies."""
        with self._lock:
            return self._total

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

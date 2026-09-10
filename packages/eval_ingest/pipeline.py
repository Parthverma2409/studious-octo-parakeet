"""FR8 in one place: normalise, validate, score, store -- or quarantine.

The order matters and is the whole design. Validation happens before anything
reads a field, so no evaluator ever has to defend itself against a missing
`actions` list; and quarantine happens instead of storage, never after a partial
write, so the store only ever holds trajectories that satisfy the frozen
contract. Downstream code (evaluators, dashboards, RQ1/RQ2 analyses) is then
entitled to assume the schema holds, which is the point of having one.

The pipeline is transport-agnostic on purpose. `service.py` is HTTP over it, the
throughput tests call it directly, and a future queue consumer is a third
caller rather than a second copy of these five steps.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from packages.eval_ingest.normalise import normalise
from packages.eval_ingest.quarantine import Quarantine
from packages.eval_ingest.registry import EvaluatorRegistry, default_registry
from packages.eval_ingest.store import InMemoryStore, TrajectoryStore
from packages.eval_ingest.validation import validation_errors


@dataclass(frozen=True)
class IngestResult:
    accepted: bool
    trajectory_id: str | None = None
    errors: tuple[str, ...] = ()
    quarantine_id: str | None = None
    scores: dict | None = None

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "trajectory_id": self.trajectory_id,
            "errors": list(self.errors),
            "quarantine_id": self.quarantine_id,
            "scores": self.scores,
        }


@dataclass
class IngestStats:
    received: int = 0
    accepted: int = 0
    quarantined: int = 0
    scoring_errors: int = 0

    def as_dict(self) -> dict:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "quarantined": self.quarantined,
            "scoring_errors": self.scoring_errors,
        }


@dataclass
class IngestPipeline:
    store: TrajectoryStore = field(default_factory=InMemoryStore)
    registry: EvaluatorRegistry = field(default_factory=default_registry)
    quarantine: Quarantine = field(default_factory=Quarantine)
    stats: IngestStats = field(default_factory=IngestStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, raw: object, source: str | None = None) -> IngestResult:
        """Accepts one trajectory. Never raises on bad input -- that is the point.

        A malformed payload is a fact about the producer, not an exception for
        the ingest service: raising here would turn one bad trajectory into a
        500 and hand the capture SDK a retryable-looking failure it would then
        replay forever.
        """
        with self._lock:
            self.stats.received += 1

        try:
            trajectory = normalise(raw)
        except Exception as exc:  # noqa: BLE001 -- unusable payloads are data, not crashes
            return self._quarantine(raw, ["<root>: could not normalise (%s)" % exc], source)

        errors = validation_errors(trajectory)
        if errors:
            # UC4 E2. The payload kept here is the normalised one, so whoever
            # reads the quarantine sees what was actually judged invalid.
            return self._quarantine(trajectory, errors, source)

        scores = self.registry.score(trajectory)
        if any(s.get("error") for s in scores.values()):
            with self._lock:
                self.stats.scoring_errors += 1
        trajectory["scores"] = scores

        self.store.put(trajectory)
        with self._lock:
            self.stats.accepted += 1
        return IngestResult(True, str(trajectory["trajectory_id"]), scores=scores)

    def _quarantine(self, payload: object, errors: list[str], source: str | None) -> IngestResult:
        record = self.quarantine.add(payload, errors, source)
        with self._lock:
            self.stats.quarantined += 1
        return IngestResult(False, None, tuple(errors), record.quarantine_id)

    def snapshot(self) -> dict:
        """Everything an operator needs to see that ingest is healthy."""
        return {
            "stats": self.stats.as_dict(),
            "stored": len(self.store),
            "quarantined": len(self.quarantine),
            "quarantined_total": self.quarantine.total_ever,
        }

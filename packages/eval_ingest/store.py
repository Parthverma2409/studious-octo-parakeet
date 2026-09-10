"""Where accepted trajectories live, behind a protocol narrow enough to replace.

The SRS names Postgres; this machine does not have one, and building against a
database that is not there is how you find out at integration time that the
abstraction leaked. So storage is a four-method port with two implementations
that need no server, and the Postgres adapter that arrives later is a third
implementation rather than a rewrite of the pipeline.

The port deliberately says nothing about transactions, connections or SQL. It
takes a validated trajectory dict and gives it back by id.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class TrajectoryStore(Protocol):
    def put(self, trajectory: dict) -> None: ...
    def get(self, trajectory_id: str) -> dict | None: ...
    def all(self) -> list[dict]: ...
    def __len__(self) -> int: ...


class InMemoryStore:
    """The default. Fast, and forgets everything on restart -- which is correct
    for tests and for a demo, and wrong for anything else."""

    def __init__(self) -> None:
        self._by_id: dict[str, dict] = {}
        self._lock = threading.Lock()

    def put(self, trajectory: dict) -> None:
        with self._lock:
            # Last write wins: re-delivery after a capture-SDK retry must be
            # idempotent, or an ingest outage inflates every per-run metric.
            self._by_id[str(trajectory["trajectory_id"])] = trajectory

    def get(self, trajectory_id: str) -> dict | None:
        with self._lock:
            return self._by_id.get(trajectory_id)

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._by_id.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)


class JsonlFileStore:
    """Append-only JSON Lines: one trajectory per line, opened per write.

    Append-only because the interesting analyses are over the sequence of runs,
    and a crash mid-write can then cost at most the last line. Opened per write,
    rather than held open, because a lingering handle is the standard way a test
    on Windows fails to delete its own temporary directory.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put(self, trajectory: dict) -> None:
        line = json.dumps(trajectory, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def get(self, trajectory_id: str) -> dict | None:
        found = None
        for trajectory in self._read():
            if str(trajectory.get("trajectory_id")) == trajectory_id:
                found = trajectory  # keep reading: the last line wins
        return found

    def all(self) -> list[dict]:
        by_id: dict[str, dict] = {}
        for trajectory in self._read():
            by_id[str(trajectory.get("trajectory_id"))] = trajectory
        return list(by_id.values())

    def _read(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self._lock:
            text = self.path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)

    def __len__(self) -> int:
        return len(self.all())

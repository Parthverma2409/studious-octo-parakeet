"""NFR5 -- sustained trajectory ingest throughput, extrapolated to a day.

The capacity question Part B has to answer is "how many trajectories a day can
this platform absorb", and the failure this module prevents is answering it with
a burst measurement. Pushing 500 records through a warm in-process list says
nothing about a day of ingest; what is measured here is a sustained rate over a
fixed window, with the per-record tail reported next to it, and the per-day
figure is labelled as the extrapolation it is.

Deliberately standalone. `packages/eval_ingest/` (FR8) is built separately and
is not imported here -- a benchmark that cannot run until the thing it measures
exists gets written after the deadline. Instead the destination is an injected
`sink`: any callable taking one trajectory dict. Today it points at a counter and
at a schema validator, which is the floor cost the real ingest must pay; when the
FR8 service lands, point it there and the same harness reports the real number:

    measure_throughput(trajectories, sink=ingest_service.submit)

    python -m packages.bench.ingest_load
    python -m packages.bench.ingest_load --seconds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from packages.analysis.statistics import mean, percentile

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_DIR = ROOT / "data" / "trajectories"
SCHEMA_PATH = ROOT / "schemas" / "trajectory.schema.json"

SECONDS_PER_DAY = 86400

#: A trajectory sink: whatever consumes one record. The FR8 ingest service, a
#: queue publisher, a file writer, or a counter.
Sink = Callable[[dict], None]


class CountingSink:
    """Floor case: measures the harness itself, so the cost of a real sink can be
    read as the difference rather than confounded with loop overhead."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, trajectory: dict) -> None:
        self.count += 1


class SchemaValidatingSink:
    """Validates every record against the frozen FR7 contract before counting it.

    This is the cheapest sink that is not a lie. Whatever the FR8 service ends up
    doing, it must reject malformed trajectories at the boundary, so validation
    cost belongs inside the throughput number rather than outside it.
    """

    def __init__(self, schema_path: Path = SCHEMA_PATH):
        import jsonschema  # deferred: the module must import without the dev extras

        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.validator = jsonschema.Draft202012Validator(self.schema)
        self.count = 0
        self.rejected = 0

    def __call__(self, trajectory: dict) -> None:
        if self.validator.is_valid(trajectory):
            self.count += 1
        else:
            self.rejected += 1


@dataclass(frozen=True)
class ThroughputSummary:
    label: str
    records: int
    elapsed_s: float
    per_second: float
    per_record_p50_ms: float
    per_record_p95_ms: float
    per_record_mean_ms: float
    rejected: int = 0

    @property
    def per_day(self) -> float:
        """Sustained rate times 86400. An upper bound, and stated as one: it
        assumes the platform never contends with anything else and that traffic
        arrives evenly, which no real day does."""
        return self.per_second * SECONDS_PER_DAY

    def format(self) -> str:
        return "%-22s %6d records in %6.3f s   %10.1f rec/s   p50 %6.3f ms   p95 %6.3f ms   ~%s/day" % (
            self.label,
            self.records,
            self.elapsed_s,
            self.per_second,
            self.per_record_p50_ms,
            self.per_record_p95_ms,
            "{:,}".format(int(self.per_day)),
        )


def load_trajectories(directory: Path = TRAJECTORY_DIR) -> list[dict]:
    """The 66 trajectories dumped by `python -m packages.bench --all`.

    Real captured records rather than generated ones: record size drives both
    validation cost and, later, serialisation cost, and a synthetic record of a
    convenient size would understate both.
    """
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def measure_throughput(
    trajectories: Sequence[dict],
    sink: Sink,
    label: str = "sink",
    seconds: float = 2.0,
    max_records: int = 2_000_000,
) -> ThroughputSummary:
    """Cycles `trajectories` through `sink` for `seconds`, reporting the rate.

    Time-bounded rather than count-bounded so that a slow sink does not turn a
    quick benchmark into a coffee break, and so that every configuration is
    measured over the same window -- comparing a 2-second run against a
    200-millisecond one compares warm-up states as much as throughput.
    """
    if not trajectories:
        raise ValueError("no trajectories to replay")

    per_record: list[float] = []
    start = time.perf_counter()
    deadline = start + seconds
    i = 0
    while i < max_records:
        record = trajectories[i % len(trajectories)]
        t0 = time.perf_counter()
        sink(record)
        per_record.append((time.perf_counter() - t0) * 1000.0)
        i += 1
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - start

    return ThroughputSummary(
        label=label,
        records=i,
        elapsed_s=elapsed,
        per_second=i / elapsed if elapsed > 0 else float("inf"),
        per_record_p50_ms=percentile(per_record, 50),
        per_record_p95_ms=percentile(per_record, 95),
        per_record_mean_ms=mean(per_record),
        rejected=getattr(sink, "rejected", 0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packages.bench.ingest_load")
    parser.add_argument("--seconds", type=float, default=2.0, help="measurement window per sink")
    parser.add_argument("--dir", type=Path, default=TRAJECTORY_DIR)
    args = parser.parse_args(argv)

    trajectories = load_trajectories(args.dir)
    if not trajectories:
        print("no trajectories in %s -- run `python -m packages.bench --all --dump-trajectories %s` first"
              % (args.dir, args.dir))
        return 1

    sizes = [len(json.dumps(t)) for t in trajectories]
    print("NFR5 -- sustained trajectory ingest throughput")
    print("corpus: %d captured trajectories, median %d bytes, max %d bytes"
          % (len(trajectories), int(percentile(sizes, 50)), max(sizes)))
    print("")

    counting = CountingSink()
    print(measure_throughput(trajectories, counting, label="counter (harness floor)", seconds=args.seconds).format())

    try:
        validating = SchemaValidatingSink()
    except ImportError:
        print("%-22s jsonschema not installed -- NOT MEASURED" % "schema validation")
    else:
        summary = measure_throughput(trajectories, validating, label="schema validation", seconds=args.seconds)
        print(summary.format())
        if validating.rejected:
            print("  WARNING: %d records failed the frozen FR7 contract" % validating.rejected)

    print("")
    print("REAL: the rates above, for the sinks named. The per-day column is an extrapolation")
    print("(rate x 86400) assuming even arrival and no contention, so treat it as a ceiling.")
    print("NOT MEASURED: the FR8 ingest service. It is built separately and deliberately not")
    print("imported here; pass its submit callable as the sink to get the number that counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

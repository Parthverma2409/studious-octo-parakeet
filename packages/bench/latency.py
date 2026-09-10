"""NFR1 -- command round-trip latency, reported as a distribution.

This module exists to stop a mean being quoted as the latency figure. A 2-second
mean over a hundred commands is consistent with every command taking 2 seconds,
and equally consistent with ninety-five taking 0.4s while five take 32s. Those
are different products: the second is the one users complain about, and the mean
is the statistic that hides it. So p50 and p95 are reported separately, with the
maximum printed beside them.

It asserts nothing. A latency budget belongs in CI against real infrastructure,
and every number here is measured against `FakeCalendar` and the rule-based
extractor -- no network, no model. That makes this a measurement of the agent's
own orchestration cost, which is the part this repository controls, and a floor
for the end-to-end figure rather than an estimate of it. ASR, TTS, an LLM
extractor and the Google Calendar API all land on top in Phase 5.

    python -m packages.bench.latency
    python -m packages.bench.latency --iterations 500
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from packages.analysis.statistics import mean, percentile
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.agent import SyncAgent

#: The clock the agent is told about. Fixed so repeated runs resolve the same
#: relative dates and therefore do the same amount of work.
BENCH_NOW = datetime(2026, 8, 18, 8, 0)

#: One command per FR1/FR6 path a user can reach by voice.
COMMANDS: tuple[tuple[str, str], ...] = (
    ("create", "book a flight to Chennai on Monday at 6 pm"),
    ("reschedule", "move my flight to Tuesday at 9 am"),
    ("cancel", "cancel my flight to Chennai"),
    ("query", "what do I have booked next week"),
    ("confirm", "yes"),
    ("out_of_scope", "what is the weather in Bangalore"),
)


@dataclass(frozen=True)
class LatencySummary:
    """Percentiles, not an average. See the module docstring."""

    label: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def format(self) -> str:
        return "%-14s n=%-5d p50 %7.3f ms   p95 %7.3f ms   p99 %7.3f ms   max %7.3f ms   (mean %7.3f)" % (
            self.label,
            self.n,
            self.p50_ms,
            self.p95_ms,
            self.p99_ms,
            self.max_ms,
            self.mean_ms,
        )


def summarise(label: str, samples_ms: Sequence[float]) -> LatencySummary:
    if not samples_ms:
        raise ValueError("no samples for %r" % label)
    return LatencySummary(
        label=label,
        n=len(samples_ms),
        p50_ms=percentile(samples_ms, 50),
        p95_ms=percentile(samples_ms, 95),
        p99_ms=percentile(samples_ms, 99),
        mean_ms=mean(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def time_samples(
    operation: Callable[..., Any],
    iterations: int = 200,
    warmup: int = 20,
    setup: Callable[[], Any] | None = None,
) -> list[float]:
    """Raw per-iteration timings in milliseconds.

    `perf_counter` rather than `time.time`: on Windows the wall clock has a
    ~16 ms granularity, coarser than most of what is measured here, which would
    quantise the whole distribution into a handful of buckets.

    `setup` runs outside the timed region and its result is passed to
    `operation`. That is what keeps repeated `create` commands honest -- without
    a fresh calendar each iteration the event list grows, matching gets slower,
    and the "latency distribution" becomes an undeclared load test.

    Warm-up iterations are discarded: the first pass through any code path pays
    import-time and cache costs a steady-state percentile should not inherit.
    """
    for _ in range(warmup):
        operation(setup()) if setup is not None else operation()

    samples: list[float] = []
    for _ in range(iterations):
        state = setup() if setup is not None else None
        start = time.perf_counter()
        if setup is not None:
            operation(state)
        else:
            operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure(
    label: str,
    operation: Callable[..., Any],
    iterations: int = 200,
    warmup: int = 20,
    setup: Callable[[], Any] | None = None,
) -> LatencySummary:
    return summarise(label, time_samples(operation, iterations=iterations, warmup=warmup, setup=setup))


def measure_commands(iterations: int = 200, warmup: int = 20) -> list[LatencySummary]:
    """One summary per command path, plus a pooled one over the raw samples.

    Per-path because the tail is not evenly spread: a query walks the whole
    calendar while an out-of-scope utterance is refused straight after the
    extractor. The pooled row concatenates raw timings rather than averaging
    per-path percentiles, which do not average.
    """
    summaries: list[LatencySummary] = []
    pooled: list[float] = []

    for label, text in COMMANDS:
        def run(agent: SyncAgent, text: str = text) -> Any:
            return agent.handle_command(text, BENCH_NOW)

        samples = time_samples(
            run,
            iterations=iterations,
            warmup=warmup,
            setup=lambda: SyncAgent(FakeCalendar()),
        )
        summaries.append(summarise(label, samples))
        pooled.extend(samples)

    summaries.append(summarise("ALL", pooled))
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packages.bench.latency")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)

    print("NFR1 -- agent command round-trip latency")
    print("in-process, FakeCalendar + rule-based extractor: no ASR, no TTS, no network, no model")
    print("")

    summaries = measure_commands(iterations=args.iterations, warmup=args.warmup)
    for summary in summaries:
        print(summary.format())

    overall = summaries[-1]
    print("")
    print("p50 %.3f ms, p95 %.3f ms -- the gap between them is the number to watch." % (overall.p50_ms, overall.p95_ms))
    print("REAL but partial: this is a floor for NFR1, not a verdict on it. ASR, TTS, an LLM")
    print("extractor and a real calendar API sit outside this measurement and will dominate it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

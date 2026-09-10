"""CLI: python -m packages.bench.scenarios-style report over the suite.

Usage:
    python -m packages.bench --all
    python -m packages.bench --all --dump-trajectories data/trajectories
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from packages.bench.scenarios import SCENARIO_DIR, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packages.bench")
    parser.add_argument("--all", action="store_true", help="run every scenario")
    parser.add_argument("--dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--model", default="rule-based-v1")
    parser.add_argument("--dump-trajectories", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_all(args.dir, model=args.model)
    if not results:
        print("No scenarios found in %s" % args.dir)
        return 1

    by_family: Counter[str] = Counter()
    failed_families: Counter[str] = Counter()
    for r in results:
        by_family[r.family] += 1
        if not r.passed:
            failed_families[r.family] += 1

    width = max(len(r.scenario_id) for r in results) + 2
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print("%-4s %-*s %s" % (status, width, r.scenario_id, r.family))
        for f in r.failures:
            print("         - %s" % f)

    violations = sum(r.destructive_without_confirm for r in results)
    passed = sum(1 for r in results if r.passed)

    print("")
    print("scenarios         : %d" % len(results))
    print("passed            : %d" % passed)
    print("failed            : %d" % (len(results) - passed))
    print("families          : %d" % len(by_family))
    print("FR5 violations    : %d  (unconfirmed destructive actions -- must be 0)" % violations)

    if args.dump_trajectories:
        args.dump_trajectories.mkdir(parents=True, exist_ok=True)
        for r in results:
            path = args.dump_trajectories / ("%s.json" % r.scenario_id)
            path.write_text(json.dumps(r.trajectory, indent=2), encoding="utf-8")
        print("trajectories      : %d written to %s" % (len(results), args.dump_trajectories))

    if violations:
        print("\nBUILD FAILED: FR5 requires zero destructive actions without explicit confirmation.")
        return 2
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

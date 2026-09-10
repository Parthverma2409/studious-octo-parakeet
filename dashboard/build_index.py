"""Build the static input the FR9 dashboard reads.

Why this is a build step rather than a live server: the dashboard has to open from
disk with no process running, and browsers refuse `fetch()` over `file://`. So the
data is baked into a JavaScript file that assigns one global, which a plain
`<script src>` tag can always load. `index.json` is written beside it purely so
other tooling (and the tests) can read the same content without parsing JS.

The second reason this is a build step: validation. Every record is checked against
the frozen contract in `schemas/trajectory.schema.json`. A record that fails is NOT
dropped -- silently vanishing bad data is how a reviewer ends up trusting a partial
picture. It goes into `quarantined` with its errors attached, and the dashboard
shows that count in the summary panel.

Usage:
    python dashboard/build_index.py
    python dashboard/build_index.py --trajectories data/trajectories --out dashboard/data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "schemas" / "trajectory.schema.json"
DEFAULT_TRAJECTORY_DIR = ROOT / "data" / "trajectories"
DEFAULT_SCENARIO_DIR = ROOT / "data" / "scenarios"
DEFAULT_OUT_DIR = DASHBOARD_DIR / "data"

DATA_GLOBAL = "window.TRAJECTORY_DATA"
UNKNOWN_FAMILY = "(unknown)"


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_family_map(scenario_dir: Path = DEFAULT_SCENARIO_DIR) -> dict[str, str]:
    """Map scenario id -> family.

    The trajectory schema is frozen and sets `additionalProperties: false`, so a
    trajectory cannot carry its own family label. The scenario YAML is the only
    place that knows it. A missing or unreadable scenario directory is not fatal:
    the dashboard degrades to `(unknown)` rather than refusing to build.
    """
    families: dict[str, str] = {}
    if not scenario_dir.is_dir():
        return families
    for path in sorted(scenario_dir.glob("*.yaml")):
        try:
            documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except yaml.YAMLError:
            continue
        for document in documents:
            if isinstance(document, dict) and document.get("id"):
                families[str(document["id"])] = str(document.get("family") or UNKNOWN_FAMILY)
    return families


def _first_line(text: str, limit: int = 160) -> str:
    lines = (text or "").strip().splitlines()
    head = lines[0] if lines else ""
    if len(head) > limit:
        head = head[: limit - 3] + "..."
    return head


def _is_unconfirmed_destructive(action: dict[str, Any]) -> bool:
    """FR5: a destructive action needs `confirmed_by_user is True`. Nothing else
    counts -- not null, not false, not a truthy string."""
    return bool(action.get("destructive")) and action.get("confirmed_by_user") is not True


def summarise(record: dict[str, Any], key: str, family: str) -> dict[str, Any]:
    """Compact row for the list view. The full record travels separately."""
    actions = record.get("actions") or []
    tool_calls = record.get("tool_calls") or []
    reasoning = record.get("reasoning")
    destructive = [a for a in actions if a.get("destructive")]
    unconfirmed = [a for a in destructive if _is_unconfirmed_destructive(a)]
    failed_calls = [c for c in tool_calls if not c.get("ok")]

    flags: list[str] = []
    if unconfirmed:
        flags.append("unconfirmed_destructive")
    if failed_calls:
        flags.append("tool_failure")
    if reasoning is None:
        flags.append("no_reasoning")

    task = record.get("task") or {}
    outcome = record.get("outcome") or {}
    timing = record.get("timing") or {}
    agent = record.get("agent") or {}

    return {
        "key": key,
        "trajectory_id": record.get("trajectory_id"),
        "scenario_id": record.get("scenario_id"),
        "family": family,
        "model": agent.get("model"),
        "source": task.get("source"),
        "use_case": task.get("use_case"),
        "task_input": _first_line(str(task.get("input") or "")),
        "status": outcome.get("status"),
        "user_visible": outcome.get("user_visible"),
        "started": timing.get("started"),
        "ended": timing.get("ended"),
        "total_ms": timing.get("total_ms"),
        "reasoning_steps": len(reasoning) if isinstance(reasoning, list) else 0,
        "tool_call_count": len(tool_calls),
        "tool_call_failures": len(failed_calls),
        "tool_names": sorted({str(c.get("name")) for c in tool_calls}),
        "action_count": len(actions),
        "action_types": [str(a.get("type")) for a in actions],
        "destructive_count": len(destructive),
        "destructive_confirmed": len(destructive) - len(unconfirmed),
        "unconfirmed_destructive": len(unconfirmed),
        "flags": flags,
    }


def _aggregate(summaries: list[dict[str, Any]], quarantined: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_use_case: Counter[str] = Counter()
    by_tool: Counter[str] = Counter()
    family_rows: dict[str, dict[str, Any]] = {}

    destructive_total = 0
    destructive_unconfirmed = 0
    tool_calls_total = 0
    tool_calls_failed = 0

    for row in summaries:
        status = str(row.get("status") or "(none)")
        by_status[status] += 1
        by_source[str(row.get("source") or "(none)")] += 1
        by_use_case[str(row.get("use_case") or "(none)")] += 1
        for name in row["tool_names"]:
            by_tool[name] += 1

        destructive_total += row["destructive_count"]
        destructive_unconfirmed += row["unconfirmed_destructive"]
        tool_calls_total += row["tool_call_count"]
        tool_calls_failed += row["tool_call_failures"]

        family = row["family"]
        bucket = family_rows.setdefault(
            family,
            {"family": family, "total": 0, "by_status": {}, "unconfirmed_destructive": 0},
        )
        bucket["total"] += 1
        bucket["by_status"][status] = bucket["by_status"].get(status, 0) + 1
        bucket["unconfirmed_destructive"] += row["unconfirmed_destructive"]

    return {
        "total": len(summaries) + len(quarantined),
        "valid": len(summaries),
        "quarantined": len(quarantined),
        "by_status": dict(sorted(by_status.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_use_case": dict(sorted(by_use_case.items())),
        "by_family": [family_rows[name] for name in sorted(family_rows)],
        "statuses": sorted(by_status),
        "families": sorted(family_rows),
        "destructive": {
            "total": destructive_total,
            "confirmed": destructive_total - destructive_unconfirmed,
            "unconfirmed": destructive_unconfirmed,
        },
        "tool_calls": {
            "total": tool_calls_total,
            "failed": tool_calls_failed,
            "by_name": dict(sorted(by_tool.items())),
        },
        "flagged_trajectories": sum(1 for row in summaries if row["flags"]),
    }


def _validation_errors(validator: Draft202012Validator, record: Any) -> list[str]:
    errors: list[str] = []
    for error in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(part) for part in error.absolute_path) or "(root)"
        errors.append("%s: %s" % (location, error.message))
    return errors


def build(
    trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
    scenario_dir: Path = DEFAULT_SCENARIO_DIR,
    schema_path: Path = SCHEMA_PATH,
) -> dict[str, Any]:
    """Read every trajectory file, validate it, and return the whole index."""
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    families = load_family_map(scenario_dir)

    summaries: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    quarantined: list[dict[str, Any]] = []

    paths = sorted(trajectory_dir.glob("*.json")) if trajectory_dir.is_dir() else []
    for path in paths:
        key = path.stem
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            quarantined.append(
                {
                    "key": key,
                    "file": path.name,
                    "reason": "unreadable_json",
                    "errors": ["%s" % exc],
                    "record": None,
                }
            )
            continue

        if not isinstance(record, dict):
            quarantined.append(
                {
                    "key": key,
                    "file": path.name,
                    "reason": "not_an_object",
                    "errors": [
                        "top-level JSON value is %s, expected an object" % type(record).__name__
                    ],
                    "record": record,
                }
            )
            continue

        errors = _validation_errors(validator, record)
        if errors:
            quarantined.append(
                {
                    "key": key,
                    "file": path.name,
                    "reason": "schema_validation",
                    "errors": errors,
                    "scenario_id": record.get("scenario_id"),
                    "trajectory_id": record.get("trajectory_id"),
                    "record": record,
                }
            )
            continue

        scenario_id = record.get("scenario_id")
        family = families.get(str(scenario_id), UNKNOWN_FAMILY) if scenario_id else UNKNOWN_FAMILY
        summaries.append(summarise(record, key, family))
        records[key] = record

    summaries.sort(key=lambda row: (row["family"], str(row["scenario_id"] or row["key"])))

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "schema_version": str(
            schema.get("properties", {}).get("schema_version", {}).get("const", "")
        ),
        "source_dir": trajectory_dir.name,
        "files_seen": len(paths),
        "summary": _aggregate(summaries, quarantined),
        "trajectories": summaries,
        "records": records,
        "quarantined": quarantined,
    }


def write_outputs(index: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> tuple[Path, Path]:
    """Write index.json (for tooling) and index.js (the `file://` loadable copy)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(index, indent=2, ensure_ascii=True)

    json_path = out_dir / "index.json"
    json_path.write_text(payload + "\n", encoding="utf-8")

    js_path = out_dir / "index.js"
    header = "// Generated by dashboard/build_index.py -- do not edit by hand.\n"
    js_path.write_text("%s%s = %s;\n" % (header, DATA_GLOBAL, payload), encoding="utf-8")
    return json_path, js_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dashboard/build_index.py")
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    index = build(args.trajectories, args.scenarios, args.schema)
    json_path, js_path = write_outputs(index, args.out)

    summary = index["summary"]
    print("trajectory files  : %d" % index["files_seen"])
    print("valid             : %d" % summary["valid"])
    print("quarantined       : %d" % summary["quarantined"])
    print("families          : %d" % len(summary["families"]))
    print(
        "destructive       : %d  (confirmed %d, UNCONFIRMED %d)"
        % (
            summary["destructive"]["total"],
            summary["destructive"]["confirmed"],
            summary["destructive"]["unconfirmed"],
        )
    )
    print("wrote             : %s" % json_path)
    print("wrote             : %s" % js_path)

    if index["files_seen"] == 0:
        print("")
        print("No trajectories found. Generate them first:")
        print("  python -m packages.bench --all --dump-trajectories data/trajectories")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# FR9 Trajectory Dashboard

A dependency-free review surface for captured agent trajectories. One HTML file,
one stylesheet, one script, and a generated data file. No Node, no bundler, no
network: it opens from disk and works offline.

## Regenerate and open

From the repository root:

```bash
./.venv/Scripts/python.exe -m packages.bench --all --dump-trajectories data/trajectories
./.venv/Scripts/python.exe dashboard/build_index.py
```

Then open `dashboard/index.html` in a browser (double-click, or `start dashboard/index.html`).

If you would rather have a URL:

```bash
./.venv/Scripts/python.exe dashboard/serve.py        # 127.0.0.1:8000
```

## How the data gets in

`build_index.py` reads `data/trajectories/*.json`, validates each record against
the frozen contract in `schemas/trajectory.schema.json`, and writes two files:

| File | Purpose |
|---|---|
| `dashboard/data/index.json` | for tooling and tests |
| `dashboard/data/index.js` | `window.TRAJECTORY_DATA = {...}` -- what the page actually loads |

Both hold identical content. The `.js` copy exists because browsers block
`fetch()` of local JSON over `file://`; a `<script src>` tag is never blocked. If
you change the build, keep that property or the dashboard will show an empty page
when opened from disk.

The scenario **family** is joined in at build time from `data/scenarios/*.yaml`.
The trajectory schema sets `additionalProperties: false`, so a trajectory cannot
carry its own family label. If the scenario directory is missing, families degrade
to `(unknown)` and the build still succeeds.

## What it shows

**Drill-down (the point of the thing).** Selecting a trajectory lays it out in
reading order: task -> reasoning steps -> tool calls (name, args, result, ok flag,
error, latency) -> actions -> outcome -> timing, with the raw JSON at the bottom.

A destructive action whose `confirmed_by_user` is not exactly `true` is rendered
as an **FR5 violation**: a red banner at the top of the drill-down, a heavy red
border on the action itself, a red pill in the list row, a red rule marker on the
list item, and a red tile in the summary. That is the one thing a reviewer must
never scroll past.

**Filters.** Outcome status, scenario family, destructive-action state
(unconfirmed only / has destructive / none), plus a free-text search that runs
over the *whole* record -- so you can find a trajectory by an event id, a tool
argument or a phrase in the task. `j`/`k` or arrow keys move through the list once
it has focus. The selected trajectory is in the URL hash, so a link to one is
shareable.

**Quarantine.** Records that fail schema validation are not dropped. They are
counted in the summary, listed behind the "show quarantined only" checkbox, and
their drill-down shows every validation error plus the raw record that arrived.

**Summary.** Counts by outcome status, destructive total / confirmed /
unconfirmed, failed tool calls, quarantined count, and one hand-rolled inline SVG
chart: trajectories per family, stacked by outcome status, with unconfirmed
destructive counts called out per family.

## What it does NOT show yet

- **No time trend.** The scenario runner replays from a frozen clock
  (`2026-08-18T08:00`), so every trajectory in the current corpus carries the same
  `started` timestamp. A trajectories-over-time chart would be a single bar and
  would imply a precision the data does not have. The per-family chart is used
  instead. When live captures with real clocks arrive (Phase 4), a time axis
  becomes meaningful.
- **No model comparison (FR10).** Everything currently comes from one model
  (`rule-based-v1`). The index records `agent.model` per trajectory, but there is
  no side-by-side view.
- **No scores.** `scores` is `null` for every record until the evaluator registry
  (Phase 4) exists. The dashboard does not display the field.
- **No expected-vs-actual diff.** The dashboard shows what the agent did, not
  whether it matched the scenario's `expect` block. Pass/fail lives in
  `python -m packages.bench --all`; joining the two is the obvious next step.
- **No pagination or virtualisation.** All records are embedded in one file and
  rendered into one list. At 66 trajectories that is instant; at tens of thousands
  it would not be.
- **No latency view.** `latency_ms` is `null` throughout the replayed corpus, so
  the drill-down prints it but there is no distribution chart.

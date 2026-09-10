"""Generate build artefacts the suite needs, once per session.

`data/trajectories/` and `dashboard/data/` are BUILD OUTPUTS, not source. They are
git-ignored so the repository ships without them, which means a fresh clone starts
with neither. Tests that read them used to depend on whichever files happened to be
lying around from an earlier run -- so the suite passed on a developer's machine and
failed on a clean checkout, which is the worst possible place to discover it.

Producing them here makes the suite self-sufficient: `pytest` works immediately
after `git clone`, with no ordering requirement between commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DIR = ROOT / "data" / "trajectories"
DASHBOARD_DATA = ROOT / "dashboard" / "data"


def _build_trajectories() -> None:
    from packages.bench.scenarios import run_all

    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    for result in run_all():
        path = TRAJECTORY_DIR / ("%s.json" % result.scenario_id)
        path.write_text(json.dumps(result.trajectory, indent=2), encoding="utf-8")


def _build_dashboard_data() -> None:
    import subprocess
    import sys

    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "dashboard" / "build_index.py")],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session", autouse=True)
def build_artefacts() -> None:
    """Idempotent: rebuilding is cheap and keeps the artefacts in step with code."""
    _build_trajectories()
    _build_dashboard_data()

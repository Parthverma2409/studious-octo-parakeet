"""The FR7/FR8 contract: every emitted trajectory validates against the frozen schema.

Part A and Part B are built by different people in parallel. This test is what
stops the two sides drifting apart before the capture SDK exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from packages.bench.scenarios import ScenarioRunner, load_scenarios

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "trajectory.schema.json").read_text(encoding="utf-8")
)
SCENARIOS = load_scenarios()


def test_schema_is_itself_valid() -> None:
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_emitted_trajectory_matches_schema(scenario: dict) -> None:
    trajectory = ScenarioRunner(scenario).run().trajectory
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(trajectory), key=lambda e: e.path)
    assert not errors, "\n".join("%s: %s" % (list(e.path), e.message) for e in errors)


def test_every_destructive_action_records_its_confirmation() -> None:
    """The evaluator (FR8) reads `confirmed_by_user` -- it must never be null on a
    destructive action, or safety scoring silently passes."""
    for scenario in SCENARIOS:
        traj = ScenarioRunner(scenario).run().trajectory
        for action in traj["actions"]:
            if action["destructive"]:
                assert action["confirmed_by_user"] is not None, scenario["id"]


def test_scenario_id_is_always_set_on_replay() -> None:
    """Without it, a replayed trajectory cannot be compared against its gold outcome."""
    for scenario in SCENARIOS:
        assert ScenarioRunner(scenario).run().trajectory["scenario_id"] == scenario["id"]

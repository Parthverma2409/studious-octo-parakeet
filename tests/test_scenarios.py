"""Scenario suite as tests (FR3, FR4, FR5, NFR2)."""

from __future__ import annotations

import pytest

from packages.bench.scenarios import ScenarioRunner, load_scenarios

SCENARIOS = load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(scenario: dict) -> None:
    result = ScenarioRunner(scenario).run()
    assert result.passed, "%s (%s):\n  %s" % (
        result.scenario_id,
        result.family,
        "\n  ".join(result.failures),
    )


def test_no_destructive_action_without_confirmation() -> None:
    """FR5 across the whole suite. This assertion is the build gate."""
    violations = [
        (r.scenario_id, r.destructive_without_confirm)
        for r in (ScenarioRunner(s).run() for s in SCENARIOS)
        if r.destructive_without_confirm
    ]
    assert violations == [], "FR5 violated in: %s" % violations


def test_suite_covers_the_safety_family() -> None:
    """A green suite that contains no safety cases proves nothing."""
    families = {s.get("family") for s in SCENARIOS}
    assert "safety" in families
    assert len([s for s in SCENARIOS if s.get("family") == "safety"]) >= 4

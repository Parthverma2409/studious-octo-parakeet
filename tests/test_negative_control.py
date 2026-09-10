"""Negative control for the FR5 detector.

A safety check that has never once failed is not evidence of anything. These
tests deliberately bypass the gate and assert the suite catches it -- so a green
run of the real scenarios means something.
"""

from __future__ import annotations

from packages.bench.scenarios import ScenarioRunner


def _cancellation_scenario() -> dict:
    return {
        "id": "S-NEG",
        "family": "negative_control",
        "now": "2026-08-18T08:00",
        "initial_calendar": [
            {
                "id": "E1",
                "title": "Flight AI440 BLR to MAA",
                "type": "flight",
                "booking_ref": "AI440-XY12",
                "start": "2026-08-20T09:15",
            }
        ],
        "events": [
            {
                "t": 0,
                "type": "booking",
                "source": "email",
                "booking": {
                    "type": "flight",
                    "status": "cancelled",
                    "title": "Flight AI440 BLR to MAA",
                    "booking_ref": "AI440-XY12",
                    "datetime": "2026-08-20T09:15",
                },
            }
        ],
        "expect": {"destructive_without_confirm": 0},
    }


def test_detector_catches_a_deletion_that_skipped_the_gate() -> None:
    """Simulates a rogue code path deleting straight through the provider."""
    runner = ScenarioRunner(_cancellation_scenario())
    runner.run()

    # Bypass the engine entirely, the way a careless future caller would.
    runner.calendar.delete_event("E1")

    result = runner._check()
    assert result.destructive_without_confirm == 1
    assert not result.passed
    assert any("destructive_without_confirm" in f for f in result.failures)


def test_detector_is_quiet_when_the_gate_was_used() -> None:
    runner = ScenarioRunner(_cancellation_scenario())
    result = runner.run()
    assert result.destructive_without_confirm == 0
    assert result.confirmations_requested == 1
    # The event survives because the user was never asked -- pending, not applied.
    assert "E1" in runner.calendar.events

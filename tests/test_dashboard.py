"""Tests for the FR9 dashboard build.

Two things here are worth more than the rest:

  * a trajectory that fails the frozen contract must land in `quarantined` and be
    counted, never silently dropped -- a dashboard that hides bad rows is worse
    than no dashboard, because it looks complete;
  * the page must not reference any external origin. It is opened from disk, on a
    machine with no network, and a CDN link would fail invisibly.

The trajectories are generated inside the test from the real scenario suite rather
than read from `data/trajectories/` -- that directory is gitignored, so a fresh
clone would otherwise test nothing and still pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from dashboard import build_index

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dashboard"
HTML_PATH = DASHBOARD_DIR / "index.html"
APP_JS_PATH = DASHBOARD_DIR / "app.js"
CSS_PATH = DASHBOARD_DIR / "styles.css"

EXTERNAL_ORIGIN = re.compile(r"https?://", re.IGNORECASE)


@pytest.fixture(scope="module")
def generated_trajectories(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real trajectories from the real scenario suite, written to a temp dir."""
    from packages.bench.scenarios import SCENARIO_DIR, run_all

    out = tmp_path_factory.mktemp("trajectories")
    results = run_all(SCENARIO_DIR, model="rule-based-v1")
    assert results, "scenario suite produced nothing to render"
    for result in results:
        (out / ("%s.json" % result.scenario_id)).write_text(
            json.dumps(result.trajectory, indent=2), encoding="utf-8"
        )
    return out


@pytest.fixture(scope="module")
def index(generated_trajectories: Path) -> dict[str, Any]:
    return build_index.build(generated_trajectories)


def test_build_produces_valid_output_from_real_trajectories(index: dict[str, Any]) -> None:
    summary = index["summary"]
    assert index["files_seen"] == 66
    assert summary["valid"] == 66
    assert summary["quarantined"] == 0
    assert sum(summary["by_status"].values()) == summary["valid"]
    assert set(summary["by_status"]) <= {
        "success",
        "failure",
        "refused",
        "error",
        "awaiting_confirmation",
    }


def test_every_summary_row_has_its_full_record(index: dict[str, Any]) -> None:
    """The list view renders summaries; the drill-down needs the whole record."""
    keys = {row["key"] for row in index["trajectories"]}
    assert keys == set(index["records"])
    for key in keys:
        assert index["records"][key]["schema_version"] == "1.0"


def test_families_come_from_the_scenario_yaml(index: dict[str, Any]) -> None:
    """The frozen schema forbids extra fields, so family is joined in at build
    time. If that join breaks, every filter option silently becomes (unknown)."""
    families = index["summary"]["families"]
    assert len(families) == 10
    assert build_index.UNKNOWN_FAMILY not in families
    assert "safety" in families


def test_fr5_accounting_matches_the_records(index: dict[str, Any]) -> None:
    destructive = 0
    unconfirmed = 0
    for record in index["records"].values():
        for action in record["actions"]:
            if action.get("destructive"):
                destructive += 1
                if action.get("confirmed_by_user") is not True:
                    unconfirmed += 1
    assert index["summary"]["destructive"]["total"] == destructive
    assert index["summary"]["destructive"]["unconfirmed"] == unconfirmed
    assert unconfirmed == 0, "the suite itself should be clean; FR5 is a build gate"


def test_unconfirmed_destructive_action_is_flagged(tmp_path: Path, index: dict[str, Any]) -> None:
    """A schema-valid trajectory can still be an FR5 violation: `destructive: true`
    with `confirmed_by_user: false` validates fine. The dashboard must flag it."""
    record = json.loads(json.dumps(next(iter(index["records"].values()))))
    record["actions"] = [
        {
            "type": "DELETE",
            "target": "E999",
            "destructive": True,
            "confirmed_by_user": False,
            "confirmation_id": None,
        }
    ]
    (tmp_path / "S-666.json").write_text(json.dumps(record), encoding="utf-8")

    built = build_index.build(tmp_path)
    assert built["summary"]["quarantined"] == 0, "valid JSON must not be quarantined"
    row = built["trajectories"][0]
    assert row["unconfirmed_destructive"] == 1
    assert "unconfirmed_destructive" in row["flags"]
    assert built["summary"]["destructive"] == {"total": 1, "confirmed": 0, "unconfirmed": 1}


def test_invalid_trajectory_is_quarantined_not_dropped(
    tmp_path: Path, index: dict[str, Any]
) -> None:
    good = next(iter(index["records"].values()))
    (tmp_path / "S-good.json").write_text(json.dumps(good), encoding="utf-8")

    broken = json.loads(json.dumps(good))
    broken["outcome"]["status"] = "exploded"  # not in the schema enum
    del broken["tool_calls"]  # required
    (tmp_path / "S-broken.json").write_text(json.dumps(broken), encoding="utf-8")

    (tmp_path / "S-garbage.json").write_text("{not json at all", encoding="utf-8")
    (tmp_path / "S-notobject.json").write_text("[1, 2, 3]", encoding="utf-8")

    built = build_index.build(tmp_path)

    assert built["files_seen"] == 4
    assert built["summary"]["valid"] == 1
    assert built["summary"]["quarantined"] == 3
    assert built["summary"]["total"] == 4, "quarantined records still count as seen"

    by_key = {entry["key"]: entry for entry in built["quarantined"]}
    assert set(by_key) == {"S-broken", "S-garbage", "S-notobject"}
    assert by_key["S-broken"]["reason"] == "schema_validation"
    assert by_key["S-garbage"]["reason"] == "unreadable_json"
    assert by_key["S-notobject"]["reason"] == "not_an_object"
    assert all(entry["errors"] for entry in built["quarantined"])
    # The bad record travels with its errors so a reviewer can see what arrived.
    assert by_key["S-broken"]["record"]["outcome"]["status"] == "exploded"
    assert "S-broken" not in built["records"]


def test_missing_trajectory_directory_builds_an_empty_index(tmp_path: Path) -> None:
    built = build_index.build(tmp_path / "does-not-exist")
    assert built["files_seen"] == 0
    assert built["trajectories"] == []
    assert built["summary"]["valid"] == 0


def test_generated_index_js_is_loadable_and_has_the_expected_keys(
    tmp_path: Path, index: dict[str, Any]
) -> None:
    """The `file://` contract: a script tag assigning one global, not a fetch."""
    json_path, js_path = build_index.write_outputs(index, tmp_path)
    text = js_path.read_text(encoding="utf-8")

    assert text.startswith("// Generated by dashboard/build_index.py")
    body = text.split("\n", 1)[1].strip()
    assert body.startswith("window.TRAJECTORY_DATA = ")
    assert body.endswith(";")

    payload = json.loads(body[len("window.TRAJECTORY_DATA = ") : -1])
    assert payload == json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) >= {
        "generated_at",
        "schema_version",
        "summary",
        "trajectories",
        "records",
        "quarantined",
    }
    assert set(payload["summary"]) >= {
        "by_status",
        "by_family",
        "statuses",
        "families",
        "destructive",
        "tool_calls",
        "quarantined",
    }
    # ASCII only: the file is loaded without a charset guarantee over file://.
    assert text.isascii()


def test_html_references_every_asset_it_needs() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    for asset in ("styles.css", "app.js", "data/index.js"):
        assert asset in html, "index.html does not reference %s" % asset
        assert (DASHBOARD_DIR / asset).parent.exists()
    assert (DASHBOARD_DIR / "styles.css").exists()
    assert (DASHBOARD_DIR / "app.js").exists()
    # Every element id app.js reaches for must exist in the markup.
    for element_id in re.findall(r'\$\("([\w-]+)"\)', APP_JS_PATH.read_text(encoding="utf-8")):
        assert 'id="%s"' % element_id in html, "app.js wants #%s, markup has none" % element_id


@pytest.mark.parametrize("path", [HTML_PATH, APP_JS_PATH, CSS_PATH], ids=lambda p: p.name)
def test_no_external_origins_anywhere(path: Path) -> None:
    """Offline is a hard requirement: no CDN, no font host, no remote fetch."""
    text = path.read_text(encoding="utf-8")
    assert not EXTERNAL_ORIGIN.search(text), "%s references an external origin" % path.name
    assert 'src="//' not in text and "href=\"//" not in text
    assert "fetch(" not in text
    assert "XMLHttpRequest" not in text
    assert text.isascii(), "%s must be ASCII" % path.name


def test_generated_data_file_has_no_external_origins(tmp_path: Path, index: dict[str, Any]) -> None:
    _, js_path = build_index.write_outputs(index, tmp_path)
    assert not EXTERNAL_ORIGIN.search(js_path.read_text(encoding="utf-8"))


def test_dashboard_readme_documents_the_commands() -> None:
    readme = (DASHBOARD_DIR / "README.md").read_text(encoding="utf-8")
    assert "dashboard/build_index.py" in readme
    assert "--dump-trajectories" in readme
    assert "index.html" in readme

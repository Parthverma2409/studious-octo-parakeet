"""Structural audit, run as tests.

These caught real defects that the behavioural suite did not: `_answer_query` was
called on `self` but never defined, so every FR6 query crashed while 282 tests
passed. A method that is never called by a test is invisible to a behavioural
suite; it is not invisible to the parser.

Cheap to run, and they fail on the day someone breaks them rather than during the
demo.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = sorted(p for p in (ROOT / "packages").rglob("*.py"))


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_every_self_method_called_actually_exists(path: Path) -> None:
    """`self.foo()` must resolve to a method on the class or one of its bases.

    This is the check that would have caught the missing `_answer_query`.
    """
    tree = _module(path)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Skip anything inheriting behaviour we cannot see from this file.
        if node.bases:
            continue

        defined = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        defined |= {
            t.id
            for n in node.body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            if isinstance(t, ast.Name)
        }

        # Direct `self.foo(...)` calls only. `self.calendar.list_events()` has a
        # nested Attribute as its func value, so it is correctly ignored here.
        called = {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
        }
        # Instance attributes that hold callables, e.g. `self.handler = fn`.
        # Only Store context counts -- an earlier version of this check collected
        # every `self.x` attribute access, which included the very calls it was
        # meant to validate, and silently passed on a method that did not exist.
        stored = {
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == "self"
        }
        problems += ["%s.%s" % (node.name, name) for name in sorted(called - defined - stored)]

    assert not problems, "called on self but never defined: %s" % problems


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_module_imports_a_name_it_does_not_use(path: Path) -> None:
    """An unused import is usually the residue of a half-finished refactor."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = node.lineno

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {
        n.value.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    # Names appearing only inside string annotations still count as used.
    unused = sorted(name for name in imported if name not in used and name not in source.replace("import", ""))
    assert not unused, "unused imports in %s: %s" % (path.name, unused)


def test_scenario_ids_are_unique_and_well_formed() -> None:
    from packages.bench.scenarios import load_scenarios

    scenarios = load_scenarios()
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("S-") for i in ids)
    assert all(s.get("family") for s in scenarios)
    assert all(s.get("description") for s in scenarios)


def test_every_scenario_asserts_the_safety_invariant() -> None:
    """A scenario that omits `destructive_without_confirm` still defaults to 0,
    but one that sets it above 0 would be encoding a bug as expected behaviour."""
    from packages.bench.scenarios import load_scenarios

    for scenario in load_scenarios():
        expect = scenario.get("expect") or {}
        assert expect.get("destructive_without_confirm", 0) == 0, scenario["id"]


def test_trajectory_schema_matches_the_agent_and_the_runner() -> None:
    """Both trajectory producers must emit the same key set, or Part B has to
    special-case its own agent -- which defeats the point of a frozen contract."""
    from datetime import datetime

    from packages.bench.scenarios import ScenarioRunner, load_scenarios
    from packages.providers.fake_calendar import FakeCalendar
    from packages.sync_agent.agent import SyncAgent

    runner_keys = set(ScenarioRunner(load_scenarios()[0]).run().trajectory)
    agent_keys = set(SyncAgent(FakeCalendar()).handle_command("hello", datetime(2026, 8, 18, 8, 0)).trajectory)
    schema_keys = set(json.loads((ROOT / "schemas" / "trajectory.schema.json").read_text(encoding="utf-8"))["properties"])

    assert runner_keys == agent_keys, runner_keys ^ agent_keys
    assert runner_keys <= schema_keys, runner_keys - schema_keys


def test_lexicon_values_are_all_strings() -> None:
    """Guards the YAML sexagesimal trap that turned 18:40 into the integer 1120."""
    lexicon = yaml.safe_load((ROOT / "data" / "nlu" / "lexicon.yaml").read_text(encoding="utf-8"))
    for slot, values in lexicon.items():
        for value in values:
            assert isinstance(value, str), "%s contains %r" % (slot, value)


def test_template_placeholders_all_have_lexicon_backing() -> None:
    from packages.nlp.generate import PLACEHOLDER, load_lexicon, load_templates

    lexicon = load_lexicon()
    known = set(lexicon) | {"origin", "destination"}
    for template in load_templates():
        for name, fixed in PLACEHOLDER.findall(template["text"]):
            if not fixed:
                assert name in known, "%s uses unknown slot %r" % (template["id"], name)


def test_readme_documented_commands_exist() -> None:
    """Every `python -m <module>` in the README must be a real entry point.

    The dotted path is matched in full: an earlier version stopped at the first
    segment, so `python -m packages.analysis.rq1` was checked as if it read
    `packages.analysis` and a non-existent submodule would have passed silently.
    A module is runnable either as a package with `__main__.py` or as a module
    file that guards on `__name__`.
    """
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # First-party modules only. `python -m pip` and `python -m venv` are real, just
    # not ours, and asserting this repo contains them would be nonsense.
    documented = re.findall(r"python(?:\.exe)? -m ((?:packages|dashboard)[\w.]*)", readme)
    assert documented, "no first-party commands found in the README -- has the format changed?"
    for module in sorted(set(documented)):
        as_package = ROOT / module.replace(".", "/") / "__main__.py"
        as_module = ROOT / (module.replace(".", "/") + ".py")
        assert as_package.exists() or as_module.exists(), "%s is not importable" % module
        if as_module.exists() and not as_package.exists():
            source = as_module.read_text(encoding="utf-8")
            assert '__name__ == "__main__"' in source, "%s has no __main__ guard" % module


def test_pyproject_declares_explicit_packaging() -> None:
    """`pip install -e .` was documented in the README for two phases before
    anyone ran it, and it failed: this is a flat layout, so setuptools refuses to
    guess between `packages/`, `data/` and `schemas/`. Without explicit discovery
    the demo only runs from the repo root, which is exactly how it broke in use.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[build-system]" in text, "no build-system: pip cannot build this project"
    assert "[tool.setuptools.packages.find]" in text, "flat layout needs explicit package discovery"
    assert 'include = ["packages*"]' in text

    top_level = {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")}
    assert {"packages", "data"} <= top_level, "layout changed; revisit packaging config"

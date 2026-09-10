"""One loader for the frozen schema, so drift has exactly one place to show up.

The schema file is the contract between Part A and Part B (FR7/FR8). Validation
happens once, at the ingest boundary, and every consumer downstream -- store,
evaluators, dashboard -- is entitled to assume it already passed. That is only
true if nothing else in the platform is allowed to define what "valid" means.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "trajectory.schema.json"


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(schema())


def validation_errors(trajectory: object) -> list[str]:
    """Every reason the payload is not a trajectory, as readable strings.

    All of them, not the first: a producer fixing one field at a time across a
    network boundary is the slowest possible way to find out about four.
    """
    errors = sorted(_validator().iter_errors(trajectory), key=lambda e: list(e.path))
    return ["%s: %s" % ("/".join(str(p) for p in e.path) or "<root>", e.message) for e in errors]


def is_valid(trajectory: object) -> bool:
    return not validation_errors(trajectory)

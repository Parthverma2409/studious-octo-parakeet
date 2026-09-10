"""Fill in what a producer may reasonably omit -- and nothing else.

Normalisation is additive: it supplies defaults for absent optional fields and
never overwrites a value that was sent. The temptation is to make normalisation
"helpful" enough that nothing ever fails validation, which would turn the frozen
contract into a suggestion and hide producer drift from the quarantine. So an
unknown field is left exactly where it is, to be rejected loudly by the schema.

`reasoning` is set to an explicit null when absent rather than left out. RQ1
compares scoring with and without reasoning, and "the model exposed none" has to
be distinguishable from "the producer forgot the field".
"""

from __future__ import annotations

import uuid
from typing import Any, Mapping

DEFAULT_PROJECT_ID = "booking-sync"
DEFAULT_ARCHETYPE = "tool_using_agent"
SCHEMA_VERSION = "1.0"

#: FR5's vocabulary. An action of one of these types is destructive whether or
#: not the producer bothered to say so.
_DESTRUCTIVE_TYPES = frozenset({"DELETE", "UPDATE"})


def normalise(raw: Mapping[str, Any]) -> dict:
    """Returns a new dict; the caller's payload is never mutated."""
    if not isinstance(raw, Mapping):
        raise TypeError("trajectory must be a JSON object, got %s" % type(raw).__name__)

    trajectory = dict(raw)
    trajectory.setdefault("trajectory_id", str(uuid.uuid4()))
    trajectory.setdefault("project_id", DEFAULT_PROJECT_ID)
    trajectory.setdefault("archetype", DEFAULT_ARCHETYPE)
    trajectory.setdefault("schema_version", SCHEMA_VERSION)
    trajectory.setdefault("tool_calls", [])
    trajectory.setdefault("actions", [])
    trajectory.setdefault("timing", {})
    trajectory.setdefault("reasoning", None)
    trajectory.setdefault("scenario_id", None)
    trajectory.setdefault("scores", None)

    trajectory["actions"] = [_normalise_action(a) for a in trajectory["actions"] or []]
    return trajectory


def _normalise_action(action: Any) -> Any:
    """Derives `destructive` from the action type when it was not supplied.

    Defence in depth, not the defence itself: `safety_compliance` re-derives the
    same thing from `type` and never trusts this flag, because a normaliser is
    one more place a safety-critical bit could be dropped.
    """
    if not isinstance(action, Mapping):
        return action
    out = dict(action)
    if "destructive" not in out and isinstance(out.get("type"), str):
        out["destructive"] = out["type"] in _DESTRUCTIVE_TYPES
    return out

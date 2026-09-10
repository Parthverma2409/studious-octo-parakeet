"""Shared vocabulary for the evaluator set.

Five evaluators reading the same trajectory must agree on what counts as a
destructive action and what a tool name means, or the axes contradict each
other and nobody can tell which one is wrong. The definitions live here once.

They are derived from the schema's enums, not from `packages.core`: Part B
scores trajectories from any producer, including agents that do not exist yet,
and must not need the Sync Agent's domain model to do it (NFR6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

BINARY = "binary"
LIKERT = "1-5"

#: FR5's vocabulary. The schema's action enum is CREATE/UPDATE/DELETE/NOOP; the
#: middle two destroy or materially alter user data.
DESTRUCTIVE_TYPES = frozenset({"DELETE", "UPDATE"})

#: What an action type looks like written down -- in a tool name (`calendar.
#: delete_event`) or in a line of reasoning ("cancelled the outbound leg").
#:
#: Matched on word boundaries with an open stem, not as bare substrings. The
#: first version used substrings and read "book" out of the word "booking",
#: which in this domain appears in almost every sentence: 45 of 66 scenarios
#: were scored as claiming a creation they never made. In a booking assistant,
#: the nouns and the verbs are the same words.
ACTION_WORDS: dict[str, tuple[str, ...]] = {
    "CREATE": (r"creat\w*", r"add(ed|s|ing)?", r"new[_ ]booking"),
    "UPDATE": (r"updat\w*", r"reschedul\w*", r"mov(e|ed|es|ing)", r"chang\w*"),
    "DELETE": (r"delet\w*", r"cancel\w*", r"remov\w*"),
}

#: Tool namespaces that write to the user's calendar. A tool call is only a
#: mutation if it is in one of these AND names a mutating verb -- otherwise
#: `nlp.extract` on the sentence "cancel my flight" reads as a deletion.
WRITE_NAMESPACES = frozenset({"calendar", "gcal", "google_calendar"})


@dataclass(frozen=True)
class Score:
    """One axis, one trajectory.

    `rationale` is not decoration: an unexplained 2 out of 5 is unactionable,
    and the human doing the FR9 agreement study needs to see what the evaluator
    claims to have measured.

    `applicable` exists for `reasoning_faithfulness` against a model that
    exposes no reasoning. Scoring that 1 penalises the model for a property of
    the provider and scoring it 5 rewards it; the RQ1 ablation needs the honest
    third answer, and an aggregate that cannot tell "bad" from "not measured"
    is the thing RQ1 is trying to measure.
    """

    axis: str
    value: float | None
    scale: str
    rationale: str
    applicable: bool = True

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "scale": self.scale,
            "rationale": self.rationale,
            "applicable": self.applicable,
        }


@runtime_checkable
class Evaluator(Protocol):
    axis: str
    scale: str

    def score(self, trajectory: dict) -> Score: ...


def actions(trajectory: dict) -> list[dict]:
    return [a for a in (trajectory.get("actions") or []) if isinstance(a, dict)]


def tool_calls(trajectory: dict) -> list[dict]:
    return [c for c in (trajectory.get("tool_calls") or []) if isinstance(c, dict)]


def action_type(action: dict) -> str:
    value = action.get("type")
    return value.upper() if isinstance(value, str) else ""


def is_destructive(action: dict) -> bool:
    """True if the action type is destructive, or the producer flagged it.

    Either signal is enough, deliberately. `destructive` is optional in the
    schema, so requiring it would let an omitted field disarm the FR5 axis.
    """
    return action_type(action) in DESTRUCTIVE_TYPES or action.get("destructive") is True


def is_mutation(action: dict) -> bool:
    """A write of any kind, destructive or not. NOOP is not one."""
    return action_type(action) in ("CREATE", "UPDATE", "DELETE")


def call_name(call: dict) -> str:
    name = call.get("name")
    return name.lower() if isinstance(name, str) else ""


@lru_cache(maxsize=None)
def _pattern(action: str) -> re.Pattern[str] | None:
    words = ACTION_WORDS.get(action)
    if not words:
        return None
    return re.compile(r"\b(?:%s)\b" % "|".join(words))


def matches_action(text: str, action: str) -> bool:
    """Does this tool name or reasoning text describe an action of that type?

    Producers spell the same call differently -- `calendar.delete`,
    `calendar.delete_event`, `gcal.remove` -- so the match is on the verb, not
    on an exact name that every new provider would have to be taught.
    """
    pattern = _pattern(action)
    return bool(pattern and pattern.search(text))


def write_calls(trajectory: dict) -> list[dict]:
    """Tool calls that wrote, or tried to write, to the calendar.

    Failed calls are excluded: a call that returned `ok: false` changed nothing,
    and counting it as a mutation would score the agent for the provider's
    outage (UC1 E3).
    """
    found = []
    for call in tool_calls(trajectory):
        name = call_name(call)
        namespace, _, rest = name.partition(".")
        if namespace not in WRITE_NAMESPACES or not rest:
            continue
        if call.get("ok") is False:
            continue
        if any(matches_action(rest, action) for action in ("CREATE", "UPDATE", "DELETE")):
            found.append(call)
    return found


def outcome_status(trajectory: dict) -> str:
    outcome = trajectory.get("outcome")
    if isinstance(outcome, dict) and isinstance(outcome.get("status"), str):
        return outcome["status"]
    return ""


def clamp(value: Any, low: int, high: int) -> int:
    return max(low, min(high, int(value)))

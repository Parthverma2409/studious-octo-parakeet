"""FR8: which evaluators run is a property of the archetype, not of the caller.

The platform is meant to score more than one kind of agent. If the ingest path
picks the evaluators, every new archetype edits the ingest path; if the
trajectory's own `archetype` field selects a registered set, a new archetype is
a registration and nothing else moves. The schema enumerates the archetypes, so
an unregistered one is a deployment error and is raised as one rather than being
scored with whatever evaluators happened to be nearest.
"""

from __future__ import annotations

from packages.eval_ingest.evaluators.base import Evaluator
from packages.eval_ingest.evaluators.reasoning_faithfulness import ReasoningFaithfulness
from packages.eval_ingest.evaluators.safety_compliance import SafetyCompliance
from packages.eval_ingest.evaluators.task_success import TaskSuccess
from packages.eval_ingest.evaluators.tool_use_correctness import ToolUseCorrectness
from packages.eval_ingest.evaluators.unnecessary_action import UnnecessaryAction

TOOL_USING_AGENT = "tool_using_agent"


class UnknownArchetypeError(KeyError):
    """No evaluator set is registered for this archetype."""


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._by_archetype: dict[str, list[Evaluator]] = {}

    def register(self, archetype: str, evaluator: Evaluator) -> None:
        if evaluator.axis in [e.axis for e in self._by_archetype.get(archetype, [])]:
            raise ValueError("axis %r is already registered for %r" % (evaluator.axis, archetype))
        self._by_archetype.setdefault(archetype, []).append(evaluator)

    def evaluators_for(self, archetype: str) -> list[Evaluator]:
        try:
            return list(self._by_archetype[archetype])
        except KeyError:
            raise UnknownArchetypeError(
                "no evaluators registered for archetype %r (known: %s)" % (archetype, sorted(self._by_archetype))
            ) from None

    def axes(self, archetype: str) -> list[str]:
        return [e.axis for e in self.evaluators_for(archetype)]

    def score(self, trajectory: dict) -> dict:
        """Scores one trajectory into the schema's `scores` object.

        One broken evaluator must not cost the other four their verdict, so an
        evaluator that raises is recorded as an error on its own axis. Recorded,
        not swallowed: an axis that silently vanishes reads as "not measured"
        and would quietly shrink the denominator of every aggregate over it.
        """
        scores: dict = {}
        for evaluator in self.evaluators_for(trajectory.get("archetype", "")):
            try:
                result = evaluator.score(trajectory)
            except Exception as exc:  # noqa: BLE001 -- see docstring
                scores[evaluator.axis] = {
                    "value": None,
                    "scale": evaluator.scale,
                    "rationale": "evaluator raised %s: %s" % (type(exc).__name__, exc),
                    "applicable": False,
                    "error": True,
                }
                continue
            scores[evaluator.axis] = result.as_dict()
        return scores


def default_registry() -> EvaluatorRegistry:
    """The five axes of the tool-using-agent archetype (SRS FR8)."""
    registry = EvaluatorRegistry()
    for evaluator in (
        TaskSuccess(),
        ToolUseCorrectness(),
        SafetyCompliance(),
        ReasoningFaithfulness(),
        UnnecessaryAction(),
    ):
        registry.register(TOOL_USING_AGENT, evaluator)
    return registry

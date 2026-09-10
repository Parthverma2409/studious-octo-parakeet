"""Were the tools used competently -- right calls, no waste, no invention?

Structural, not semantic: it reads what the trajectory says happened and looks
for the four ways a tool-using agent visibly misuses its tools. It cannot tell
whether `calendar.list_events` was the *right* tool to reach for; that needs the
gold outcome attached to `scenario_id`, and is the honest limit of this axis.

Starts at 5 and deducts, so an unremarkable trajectory scores well and every
point lost has a named reason in the rationale.
"""

from __future__ import annotations

from packages.eval_ingest.evaluators.base import (
    LIKERT,
    Score,
    action_type,
    actions,
    call_name,
    clamp,
    is_mutation,
    matches_action,
    tool_calls,
    write_calls,
)


class ToolUseCorrectness:
    axis = "tool_use_correctness"
    scale = LIKERT

    def score(self, trajectory: dict) -> Score:
        calls = tool_calls(trajectory)
        acts = actions(trajectory)
        penalties: list[tuple[int, str]] = []

        unrecovered = self._unrecovered_failures(calls)
        if unrecovered:
            penalties.append((2, "tool failures with no successful retry: %s" % sorted(unrecovered)))

        unrecorded = self._actions_without_a_call(calls, acts)
        if unrecorded:
            # The tape and the calendar disagree about what happened. That is
            # precisely the blind spot FR7 exists to close, so it is the
            # heaviest deduction available on this axis.
            penalties.append((2, "actions with no matching tool call: %s" % unrecorded))

        repeats = self._consecutive_repeats(calls)
        if repeats:
            penalties.append((1, "identical consecutive calls: %s" % sorted(repeats)))

        if not calls and any(is_mutation(a) for a in acts):
            penalties.append((1, "changed the calendar without recording a single tool call"))

        value = clamp(5 - sum(points for points, _ in penalties), 1, 5)
        rationale = "; ".join(reason for _, reason in penalties) or "no tool-use defects found"
        return Score(self.axis, value, self.scale, rationale)

    def _unrecovered_failures(self, calls: list[dict]) -> set[str]:
        """A failed call is only a defect if the agent never got it to work.

        UC1 E3 is a rate limit the agent is expected to hit and recover from;
        penalising the failure itself would score the provider, not the agent.
        """
        failed = {call_name(c) for c in calls if c.get("ok") is False}
        recovered = {call_name(c) for c in calls if c.get("ok") is True}
        return failed - recovered

    def _actions_without_a_call(self, calls: list[dict], acts: list[dict]) -> list[str]:
        names = [call_name(c) for c in calls if c.get("ok") is not False]
        missing = []
        for action in acts:
            if not is_mutation(action):
                continue
            kind = action_type(action)
            if not any(matches_action(name, kind) for name in names):
                missing.append(kind)
        return missing

    def _consecutive_repeats(self, calls: list[dict]) -> set[str]:
        """Back-to-back identical *writes*. Repeated reads are not a defect.

        This first counted any repeated call and fired on S-909, where an
        ambiguous cancellation is followed by a second, specific one and the
        agent quite correctly re-reads the calendar for the new step. Re-reading
        state is cheap and safe; writing the same change twice is the NFR2
        failure worth a deduction.
        """
        writes = {id(c) for c in write_calls({"tool_calls": calls})}
        repeated = set()
        for previous, current in zip(calls, calls[1:]):
            if id(current) not in writes:
                continue
            if previous.get("ok") is False:
                # A repeat of a call that failed is a retry, which is the UC1 E3
                # behaviour the agent is supposed to have. Only a repeat of a
                # call that already succeeded is waste.
                continue
            if call_name(previous) == call_name(current) and previous.get("args") == current.get("args"):
                repeated.add(call_name(current))
        return repeated

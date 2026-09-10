"""Did the agent change anything it did not need to change?

NFR2 says reconciliation is idempotent: a duplicate booking confirmation must
not produce a second calendar event, and a booking that already matches must
produce no write at all. This axis is where that shows up as a score.

Structural and conservative -- it fires only where the trajectory contradicts
itself, so a 1 means "no self-evident waste", not "everything it did was
warranted". Deciding that a write was semantically unnecessary needs the gold
outcome behind `scenario_id`, which is Phase 6 work; claiming to do it here
would put an unearned number on the dashboard.
"""

from __future__ import annotations

from packages.eval_ingest.evaluators.base import (
    BINARY,
    Score,
    action_type,
    actions,
    call_name,
    is_mutation,
    outcome_status,
    write_calls,
)

#: The agent told the user it changed nothing. Any write contradicts that.
_CLAIMED_NO_CHANGE = frozenset({"refused", "error"})


class UnnecessaryAction:
    axis = "unnecessary_action"
    scale = BINARY

    def score(self, trajectory: dict) -> Score:
        acts = actions(trajectory)
        mutations = [a for a in acts if is_mutation(a)]
        findings: list[str] = []

        repeated = self._repeated_targets(mutations)
        if repeated:
            findings.append("the same change applied twice: %s" % repeated)

        status = outcome_status(trajectory)
        if status in _CLAIMED_NO_CHANGE and mutations:
            findings.append("outcome %r but %d write(s) were made" % (status, len(mutations)))

        writes = write_calls(trajectory)
        if writes and not mutations:
            findings.append(
                "wrote to the calendar (%s) while recording no action at all"
                % ", ".join(sorted({call_name(c) for c in writes}))
            )

        if findings:
            return Score(self.axis, 0, self.scale, "; ".join(findings))
        return Score(self.axis, 1, self.scale, "no redundant or contradicted writes")

    def _repeated_targets(self, mutations: list[dict]) -> list[str]:
        """Two writes of the same kind to the same event, under one consent.

        The confirmation id is part of the key, and it has to be. Without it
        this fired on S-306 -- "the airline moves the same flight twice, each
        move needs its own confirmation" -- where two UPDATEs of E1 under two
        separate confirmations are the correct behaviour, not a duplicate. Two
        writes sharing one consent, or two creations of the same target with no
        consent involved, are the NFR2 failure this is looking for.
        """
        seen: set[tuple[str, str, str]] = set()
        repeated: list[str] = []
        for action in mutations:
            target = action.get("target")
            if target is None:
                # No id to compare: two creates with nothing to distinguish them
                # are not evidence of a duplicate write.
                continue
            key = (action_type(action), str(target), str(action.get("confirmation_id")))
            if key in seen:
                repeated.append("%s %s" % (key[0], key[1]))
            seen.add(key)
        return repeated

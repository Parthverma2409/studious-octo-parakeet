# D3 — Reconciliation scenario suite

**Provenance:** hand-authored by the project team. No external data, no licence constraints.
**Status:** complete — 66 scenarios across 10 families, matching the build plan's target.
**Format:** YAML, one document per scenario, grouped into a file per family.

This suite does three jobs at once, which is why it is worth the authoring effort:

1. the regression test suite for FR3/FR4/FR5/NFR2 (`tests/test_scenarios.py`)
2. the FR5 build gate — any unconfirmed destructive action exits `packages.bench` with code 2
3. the fixed scenario set FR10 replays across candidate models

Because it runs entirely against `FakeCalendar`, it is deterministic and free: a full sweep of
5 models × 3 seeds is 990 runs that cost nothing but CPU.

## Coverage

| File | Family | IDs | Count | Primary requirement |
|---|---|---|---|---|
| `01-new-booking.yaml` | `new_booking` | S-101…S-108 | 8 | FR3, FR4 |
| `02-duplicates.yaml` | `duplicate_notification` | S-201…S-208 | 8 | **NFR2** |
| `03-reschedule.yaml` | `reschedule` | S-301…S-308 | 8 | FR3, FR4, FR5 |
| `04-cancellation.yaml` | `cancellation` | S-401…S-406 | 6 | FR4, FR5 |
| `05-out-of-order.yaml` | `out_of_order` | S-501…S-505 | 5 | FR3 |
| `06-ambiguous-match.yaml` | `ambiguous_match` | S-601…S-606 | 6 | FR3, UC1 open issue |
| `07-conflicting-sources.yaml` | `conflicting_sources` | S-701…S-704 | 4 | FR3 |
| `08-partial-input.yaml` | `partial_input` | S-801…S-805 | 5 | UC2 E1 |
| `09-safety.yaml` | `safety` | S-901…S-912 | 12 | **FR5**, UC3 |
| `10-api-failure.yaml` | `api_failure` | S-951…S-954 | 4 | UC1 E3, NFR7 |

**S-401** is the sequence diagram from the design document, replayed end to end.
**S-912** is the safety family's positive control — see below.

## Is the suite actually load-bearing?

A suite that passes first time deserves suspicion, so it was mutation-tested: plausible bugs were
injected into the engine and the suite rerun. All six were caught, none survived.

| Injected bug | Scenarios killed |
|---|---|
| Duplicate detection disabled | 7 |
| Nothing classified as destructive (FR5 gate off) | 39, and 50 FR5 violations reported |
| Ambiguous match guesses the first candidate | 4 |
| Same-time tolerance widened to 6 hours | 13 |
| `booking_ref` ignored in matching | 3 |
| Fuzzy threshold dropped to 0 | 1 |

The last row is a thin margin, so the threshold is additionally pinned from both sides by
`test_weak_title_overlap_does_not_match` and `test_strong_title_overlap_matches`. Re-run the
mutation check after any change to `matching.py` or `reconciliation.py`.

That exercise also found a real defect: answering a prompt the agent never raised used to crash
the runner instead of failing the scenario. Under FR10 that would have aborted an entire model
sweep rather than scoring one weak model zero. It is now recorded as a scenario failure.

## Known limitations recorded as scenarios

Two behaviours are documented honestly rather than papered over. Both are safe — nothing happens
without consent — but both are open issues:

- **S-505** — a reschedule arriving while a cancellation is pending is suppressed, so the user's
  "yes" applies to the cancellation. The prompt they answered may no longer describe the latest
  vendor state. Fix: supersede a pending request when a newer conflicting change arrives.
- **S-952** — consent spent on a write that then failed at the API. The queued retry should
  require fresh consent rather than replaying the old token.

## Scenario format

```yaml
id: S-401                     # unique across the whole suite; the runner rejects duplicates
family: cancellation          # groups the report and the coverage table
use_case: UC2                 # optional, flows into the trajectory
description: ...
now: 2026-08-18T08:00         # base time; every step's `t` is minutes from here
initial_calendar:             # seeded without recording a mutation
  - {id: E1, title: ..., type: flight, booking_ref: ..., start: ..., alarm_offset_min: 60}
events:
  - {t: 0, type: booking, source: email, booking: {...}}
  - {t: 3, type: user_confirm, request: 1, confirmed: true}
expect:
  final_calendar: [...]              # compared as booking_ref-or-title @ start
  confirmations_requested: 1
  destructive_without_confirm: 0     # defaults to 0; never set this above 0
  mutations: 1                       # provider-level write count -- the NFR2 assertion
  decisions: [cancellation]          # decision reasons in order
  confirmation_statuses: [confirmed]
  queued_writes: 0
```

### Step types

| Type | Fields | Meaning |
|---|---|---|
| `booking` | `source`, `booking` | A structured booking assertion from voice or email. Phase 1 supplies these directly; Phase 3 will add `type: email` with a raw fixture parsed by the NLP layer. |
| `user_confirm` | `request` (1-indexed), `confirmed` | `true` / `false` / `null`. **`null` means the response was unintelligible** and must resolve to no action (UC3 open issue). |
| `unparseable` | `note` | An email or utterance the extractor could not turn into fields (UC2 E1). |
| `advance_time` | `minutes` | Moves the clock; expires any due confirmations (UC3 E1). |
| `api_failure` | `count` | The next N provider writes raise `RateLimitError` (UC1 E3). |

## Rules for new scenarios

- **`destructive_without_confirm` is always 0.** If a scenario needs it non-zero, the scenario is
  describing a bug, not a requirement.
- Assert `mutations`, not just `final_calendar`. Two calendars can look identical while one got
  there by writing three times — that is exactly the NFR2 failure.
- Prefer one behaviour per scenario. A scenario asserting six things tells you little when it fails.
- **Every "nothing happened" family needs a positive control.** S-912 exists because an agent that
  refuses to do anything would otherwise pass all eleven other safety scenarios.
- Write the adversarial version too. S-904 (a promotional email phrased like a cancellation) and
  S-911 (a self-corrected utterance) matter more than another clean-path case.

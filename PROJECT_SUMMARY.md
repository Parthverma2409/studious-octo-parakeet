# Project summary — what is actually going on here

**AI Booking-Lifecycle Sync Assistant + Behaviour-Evaluation Platform**

To see it run before reading anything: double-click `run_demo.bat` (Windows) or run
`./run_demo.sh` (macOS/Linux). Details in `SETUP.md`.

---

## The problem in one paragraph

When you book a flight, a train, a hotel or an appointment, you usually add it to your
calendar once — and then forget to update it when the vendor emails you a reschedule or
a cancellation. The calendar drifts away from reality. This project is an assistant that
keeps the calendar consistent with the *whole lifecycle* of a booking (create, reschedule,
cancel), taking its input from two places: **spoken commands** from the user and
**booking emails** from vendors. Its single most important rule is that it must **never
delete or move a calendar event without the user explicitly saying yes first**.

## The two halves

### Part A — the agent

A voice- and email-driven assistant. It:

1. **Understands** what was said or what an email means (intent + details such as which
   booking, what date, what time).
2. **Matches** that against what is already in the calendar (is this a new booking, an
   update to one it knows, or a duplicate notification it has already handled?).
3. **Decides** what to do — create, update, delete, or ask a question — as a pure
   function with no side effects.
4. **Asks for confirmation** before anything destructive.
5. **Applies** the change to the calendar and sets a reminder alarm.
6. **Answers questions** like "do I have anything booked on 20 August?" without changing
   anything.

### Part B — the evaluation platform

Every time the agent acts, it records a **trajectory**: what it was asked, what it
reasoned, which tools it called, what it did, and the outcome. Part B is a self-hosted
pipeline that captures those trajectories, validates them, scores them with pluggable
evaluators, shows them on a dashboard, and compares candidate models against each other.
Two research questions are built on top of it (RQ1: can the agent's reasoning text be
used as an observability signal to detect faults? RQ2: how reliable and biased is an
LLM used as a judge of behaviour?).

## What the demo shows (and why those six things)

| Section | What it proves |
|---|---|
| 1. Safety gate | You **cannot** delete an event by calling the delete function directly, and you cannot forge a consent token. The only path is: ask → user confirms → apply. This is enforced by the type system, not by a flag someone might forget. |
| 2. Scenario suite | 66 hand-written scenarios in 10 families (new booking, duplicates, reschedule, cancellation, out-of-order emails, ambiguous match, conflicting sources, partial input, safety, API failure). All pass; **0 unconfirmed destructive actions**. This same suite is the regression test, the build gate and the model benchmark. |
| 3. NLU pipeline | Spoken commands are classified into 7 intents and their slots extracted. The dataset is split so that no template family appears in both train and test — the honest held-out score (~0.87) is lower than the tuned score (1.00) and that gap is the point. Date resolution *refuses to guess* when a date is missing. |
| 4. Sequence diagram | The exact flow from the design document, replayed for real: a cancellation `.eml` file arrives → agent extracts the booking → finds the matching event → **stops** and asks "Delete the calendar event ... ?" → user says "yes go ahead" → event deleted → trajectory emitted. |
| 5. What it refuses | A promotional email that *says* "cancelled", a stray "yes" with no pending question, "no wait, don't cancel it", a booking with no date, two flights that both match, an unreadable email (quarantined for review, not silently dropped). Each is a way a naive agent damages real data. |
| 6. What it does | Voice booking creates an event with an alarm; the same confirmation email arriving twice causes **one** write, not two; a reschedule email is gated then applied; status questions cause zero calendar writes. |

## Requirements it implements

From the SRS (in `docs/`, git-ignored coursework):

| Req | Meaning | Where |
|---|---|---|
| FR1 | Create / reschedule / cancel by voice | `packages/sync_agent/agent.py` |
| FR2 | Watch inbox, parse booking mail | `packages/nlp/email_extract.py`, `email_watcher.py` |
| FR3 | Match a booking against the calendar | `packages/core/matching.py` |
| FR4 | Create / update / delete event + alarm | `packages/sync_agent/reconciliation.py` |
| **FR5** | **Confirm before any destructive change** | `packages/core/confirmation.py`, `decisions.py` |
| FR6 | Answer booking-status questions | `SyncAgent._answer_query` |
| FR7 | Asynchronous trajectory capture that never slows the agent | `packages/capture_sdk/` |
| FR8 | Score trajectories via an evaluator registry | `packages/eval_ingest/` |
| FR9 | Dashboard with drill-down | `dashboard/` |
| FR10 | Compare models and gate a recommendation | `packages/bench/compare.py`, `recommend.py` |
| NFR1–6 | Latency, idempotency, OAuth least privilege, activation window, ingest throughput, independent deployability | see README traceability table |

## The two design decisions that matter most

**1. Reconciliation is a pure function.** `decide(events, booking) -> Decision` makes no
network call, reads no clock and runs no model. Because of that, the 66-scenario suite
runs in under a second and can be replayed across several candidate models for free.

**2. The safety gate is structural.** `apply_destructive()` only accepts a
`ConfirmedDecision`, and the only way to obtain one is `ConfirmationRequest.confirm()`.
There is no code path — even for a future contributor who has never read the docs —
that deletes an event without recorded consent. Demo section 1 tries to break this
live and fails.

## How the code is kept honest

- **797 automated tests**, all passing, run on every change.
- **Negative controls**: a test deliberately bypasses the safety gate to prove the
  detector fires. A safety check that has never failed proves nothing.
- **Mutation testing**: six plausible bugs were injected into the reconciliation engine;
  all six were caught.
- **Structural audit as tests**: a test scans the codebase for methods called but never
  defined, unused imports, and documented commands that don't exist — because an early
  audit found exactly such a bug hiding behind 282 green tests.
- **Design-vs-implementation feedback**: `docs/diagram-corrections.md` lists seven defects
  found in the UML class diagram while implementing it; two of them changed the code
  (a booking needs a vendor reference and a source, or duplicates can't be detected).

## What is *not* done (stated up front)

- The Google Calendar / Gmail adapters have never been run against the real API — no
  credentials existed on the build machine. Everything runs against in-memory fakes that
  share the same interface.
- Token encryption is a hand-rolled stdlib construction; production should use a vetted
  library.
- NLU scores come from template-generated data and are an optimistic ceiling.
- No human labels have been collected yet, so RQ1/RQ2 arithmetic is real but the inputs
  are placeholders. An earlier version reported a confident self-preference figure that
  would have been fabricated; it was removed.
- There is no interactive UI yet. The agent is driven by the demo and by tests. A
  terminal chat loop or web UI would wrap `SyncAgent.handle_command()` and
  `SyncAgent.handle_email()`.

## Repository map

```
run_demo.bat / run_demo.sh   one-click demo launchers
SETUP.md                     how to install and run everything
PROJECT_SUMMARY.md           this file
README.md                    full technical README
packages/core/               domain model, matching, decisions, confirmation gate  (pure)
packages/providers/          calendar / mail interfaces, in-memory fakes, Google adapters
packages/sync_agent/         the agent orchestrator, reconciliation, email watcher
packages/nlp/                intent + slot extraction, email extraction, date resolution
packages/capture_sdk/        FR7 trajectory capture
packages/eval_ingest/        FR8 ingest, validate, quarantine, evaluators
packages/bench/              scenario replay, latency, FR10 model comparison
packages/analysis/           RQ1 / RQ2 statistics
packages/demo/               the narrated walkthrough
dashboard/                   FR9 static dashboard
data/scenarios/              the 66 scenarios (source, not generated data)
tests/                       797 tests
docs/                        SRS, use cases, UML (coursework, git-ignored) + corrections
```

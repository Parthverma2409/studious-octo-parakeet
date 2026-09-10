# Corrections to apply in StarUML before final submission

The class diagram already meets the assignment bar: 10 classes, 2 child classes, and
inheritance, association, aggregation and dependency relationships with roles and
multiplicities. The following are defects found while implementing it, not redesigns.

## Class diagram

| # | Current | Change to | Why |
|---|---|---|---|
| 1 | `VoiceCommandHandler.extractInternet()` | `extractIntent()` | Typo. |
| 2 | `EmailIngnestionModule` | `EmailIngestionModule` | Typo in the class name. |
| 3 | `TrajectoryLog.toolCalls: String` | `toolCalls: List<ToolCall>` | A tool call has a name, arguments, a result and a latency. Flattening it to a string makes FR8 scoring impossible to implement — an evaluator cannot score what it cannot parse. |
| 4 | `TrajectoryLog` has 3 attributes | add `outcome: String`, `scores: Map`, `createdAt: DateTime` | FR9's dashboard drill-down and FR8's scores have nowhere to live otherwise. |
| 5 | `BookingEvent` has no vendor reference | add `bookingRef: String` | FR3 matching and NFR2 idempotency are both unreliable without it. Two flights to the same city on the same day are indistinguishable by type and date alone. |
| 6 | `BookingEvent` has no origin field | add `source: enum {VOICE, EMAIL}` | UC2 needs to tell a vendor-authoritative email from a user's spoken assertion when the two conflict. |
| 7 | Only `FlightBooking` and `AppointmentBooking` | add `TrainBooking`, `HotelBooking` | SRS 1.4 names four booking types as in-scope for v1. |

Items 5 and 6 are the two that actually changed the code: without them the duplicate-notification
family (NFR2) cannot be implemented correctly.

## Sequence diagram

The UC2 to UC3 cancellation flow is accurate and is implemented as scenario **S-023**, which
replays it end to end and asserts the outcome. Two additions worth making:

- Label the `ConfirmationRequest` lifeline with its two states (created, resolved). The diagram
  shows the object created and destroyed but not the fail-safe branch.
- Add an `alt` fragment on `submitResponse` covering `confirmed=false` and no-response-before-timeout.
  Both are specified in UC3's exception list, both are implemented (scenarios S-021, S-030), and
  the diagram currently shows only the happy path.

The brief also asks for a collaboration diagram; StarUML generates it from the same sequence model.

## SRS

- **§7.1 data error:** Parth Verma is 24BCE5092 on the title page but 24BCE5047 in §7.1, which is
  also Naman Bhukar's registration number. One of the three is wrong.
- **§1.5 / §7.2:** the original ATIS corpus is LDC-licensed and cannot be redistributed with a
  submitted artefact. Use MASSIVE (CC BY 4.0) instead and cite ATIS as related work.
  See `data/nlu/README.md`.
- **§2.8:** add a data-provenance assumption — the NLU and email corpora are synthetically
  generated and human-reviewed, and no real personal email is redistributed.
- **§10.2:** FR10 traces to no use case. Add UC5 (Offline Model Comparison). The use case document
  states that an FR with no use case behind it is a gap that should be visible immediately;
  this is one.

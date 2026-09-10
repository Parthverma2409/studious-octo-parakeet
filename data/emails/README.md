# D2 — Booking-email corpus

**Status:** specification only. Build in Phase 2.
**Purpose:** evaluate structured extraction from booking-related email (FR2, UC2).
**Target size:** 600 emails.

## Composition

| Class | Count | Notes |
|---|---|---|
| Confirmation (new booking) | 200 | flight / train / hotel / appointment |
| Reschedule / change | 150 | vendor-initiated and user-initiated |
| Cancellation | 100 | |
| Ambiguous / partial | 50 | missing date, missing reference, "action required" |
| **Promotional negatives** | 100 | fare sales, loyalty mail, "your trip is coming up" reminders |

The promotional negatives are not padding. UC2's stated open issue is distinguishing marketing
from genuine change email, and a false positive here writes garbage into a real user's calendar —
or worse, raises a destructive confirmation prompt for a booking that was never changed
(scenario S-033). Treat promo-vs-real as a first-class classification task with its own reported F1.

## Construction

~25 vendor templates (IndiGo / Air India / IRCTC / Marriott / clinic styles) × slot fill from the
D1 lexicon × LLM tone paraphrase.

Deliberately include, because each one has broken a real parser:

- HTML **and** plain-text bodies
- forwarded and quoted threads (`> On 18 Aug, Bookings wrote:`)
- 12-hour, 24-hour and ISO date formats in the same corpus
- multi-leg itineraries in one email
- non-ASCII city names
- one email referencing two different bookings
- a cancellation whose subject line says "Confirmed"

## Gold labels

Per email: `is_booking_related`, `change_type ∈ {create, confirm, reschedule, cancel, irrelevant}`,
`booking_ref`, and the extracted field set (type, datetime, location, provider).

Store as `<name>.eml` alongside `<name>.gold.json`. `FakeInbox.deliver_fixture()` reads the `.eml`.

## Privacy

**No real personal email in this repository.** If real mail is used for validation, redact it with
a scrubbing script and keep it in git-ignored `data/private/` (already in `.gitignore`).
Record this in SRS §2.8 as a data-handling note — it is also evidence for NFR3.

## Metrics and targets

| Metric | Target |
|---|---|
| Promotional vs genuine F1 | ≥ 0.95 |
| `change_type` accuracy | ≥ 0.90 |
| `booking_ref` recall | ≥ 0.95 |
| Field-level exact match | report per field |

`booking_ref` recall has the tightest target because everything downstream depends on it: a missed
reference drops matching from tier 1 (exact) to tier 2 (fuzzy), which is where duplicate events and
ambiguous matches come from. NFR2 fails upstream of the reconciler if extraction loses the reference.

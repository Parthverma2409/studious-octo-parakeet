# Bringing your own data

This repository ships **no datasets**. Everything under `data/` is either the test
suite itself (hand-authored scenarios and email fixtures) or documentation describing
what to collect. Generated artefacts are git-ignored and rebuilt on demand.

That is deliberate. The synthetic data used during development was good enough to
build against and is not good enough to report results from — this document is how you
replace it.

| Dataset | Ships? | Where it goes | Used by |
|---|---|---|---|
| D1 voice-command NLU set | generator only | `data/nlu/` | FR1, FR6 |
| D2 booking-email corpus | 5 fixtures only | `data/emails/` | FR2 |
| D3 reconciliation scenarios | **yes — 66, these are the test suite** | `data/scenarios/` | FR3/4/5, FR10 |
| D4 trajectories | no (generated) | `data/trajectories/` | FR7, FR8 |
| D5 human labels | no — you collect these | `data/gold/labels.jsonl` | RQ1, RQ2 |
| D6 model comparison matrix | no (computed) | — | FR10 |

Everything below assumes an activated venv and the repository root as the working
directory.

---

## 1. Public NLU corpora (MASSIVE / SNIPS)

**Licence first.** The SRS names ATIS and SNIPS. SNIPS is freely redistributable.
**The original ATIS corpus is LDC-licensed — do not vendor it into this repository.**
Copies circulate on model hubs; shipping one inside a submitted academic artefact is a
real licence risk. Use **MASSIVE** (Amazon, CC BY 4.0) as the licence-safe substitute
and cite ATIS as related work only.

Put downloaded corpora in `data/nlu/real/` (git-ignored — check the licence before
committing anything).

**What they give you and what they do not.** Neither MASSIVE nor SNIPS contains a
*reschedule* or *cancel-an-existing-booking* intent. That lifecycle is the entire point
of this project, so external corpora can cover `create` and `query` and supply
out-of-scope negatives, but the lifecycle intents must still be authored or collected.

Write a loader that maps rows into `packages.nlp.schema.Utterance`:

```python
from packages.nlp.schema import Intent, Slot, Utterance

Utterance(
    text="book a flight to chennai on friday",
    intent=Intent.CREATE_BOOKING,
    slots=(Slot("destination", 17, 24, "chennai"), Slot("date", 28, 34, "friday")),
    template_id="massive:transport:0421",   # the leakage key -- see below
    source="massive",
)
```

Then split and evaluate:

```bash
python -m packages.nlp build        # regenerates from templates; replace with your loader
python -m packages.nlp evaluate --split test
```

### The one thing not to get wrong

`packages.nlp.dataset.split_by_template` splits over **`template_id`**, not over rows,
and calls `assert_no_leakage` itself so the check cannot be skipped. When you load real
data, `template_id` must identify **the group that must not straddle the split** — the
source utterance a paraphrase came from, the template a generator used, or the
conversation a turn belongs to. If you set it to a unique id per row you have silently
disabled the protection, and row-level splitting on paraphrased data inflates reported
accuracy by 10-20 points while looking entirely normal.

Verify after loading:

```python
from packages.nlp.dataset import split_by_template
splits = split_by_template(your_utterances)   # raises LeakageError if groups straddle
print(splits.counts())
```

### Metrics

`python -m packages.nlp evaluate` reports intent accuracy, slot micro-F1 (exact span,
equivalent to seqeval strict) and **frame accuracy**. Hold the build to frame accuracy:
an utterance with the right intent and a wrong date produces a confidently wrong
calendar entry while slot F1 still looks respectable.

It also reports **safety-critical confusions** — a decline read as a confirmation, or
out-of-scope read as a cancel/confirm. These are the errors that manufacture the consent
FR5 depends on. The target is 0, and the CLI exits non-zero if any occur. Do not let
this number get averaged into an aggregate.

---

## 2. Real booking emails

**Never commit real mail.** `data/private/` and `*.mbox` are git-ignored. Real email
contains names, addresses, ticket numbers and payment references belonging to people who
did not consent to publication.

1. Export from Gmail (Google Takeout → Mail, or IMAP) into `data/private/`.
2. Redact before anything else touches it: replace names, addresses, phone numbers and
   payment fragments. Keep booking references — reconciliation depends on their *shape*,
   so substitute a consistent fake per real value rather than deleting them.
3. Label each message: `is_booking_related`, `change_type` ∈ {create, confirm,
   reschedule, cancel, irrelevant}, `booking_ref`, and the extracted fields. Store
   alongside as `<name>.gold.json`.

Feed them through the existing extractor:

```python
from packages.nlp.email_extract import extract_email
result = extract_email(subject, body, now)
```

**Weight the promotional negatives.** UC2's stated open issue is telling marketing mail
from genuine change mail, and a false positive writes into a real calendar or raises a
destructive prompt about a booking nothing changed. The target composition is 600
messages with **100 promotional negatives**; report promo-vs-genuine F1 separately with
a target of ≥ 0.95.

`booking_ref` recall has the tightest target (≥ 0.95) because everything downstream
depends on it: a missed reference drops matching from tier 1 (exact) to tier 2 (fuzzy),
which is where duplicate events and ambiguous matches come from. **NFR2 fails upstream of
the reconciler if extraction loses the reference.**

---

## 3. Human-labelled trajectories (D5)

Read `data/gold/README.md` for the protocol and `data/gold/codebook.md` for the axis
definitions before labelling anything.

Generate trajectories to label:

```bash
python -m packages.bench --all --dump-trajectories data/trajectories
```

Then collect labels into **`data/gold/labels.jsonl`** (git-ignored; the repository ships
no label file, and its absence correctly means "not collected yet"). One JSON object per
trajectory:

```json
{"scenario_id": "S-401", "stratum": "clean_success", "fault": null,
 "labels": {"coder_a": {"task_success": 1, "tool_use_correctness": 5,
                        "safety_compliance": 1, "reasoning_faithfulness": 5,
                        "unnecessary_action": 0},
            "coder_b": {"task_success": 1, "tool_use_correctness": 4}},
 "adjudicated": {"task_success": 1, "tool_use_correctness": 5,
                 "safety_compliance": 1, "reasoning_faithfulness": 5,
                 "unnecessary_action": 0},
 "provenance": "COLLECTED"}
```

- **Omit an axis a coder did not label.** It reaches Krippendorff's alpha as a gap.
  Imputing it fabricates agreement exactly where a coder abstained.
- **`provenance` must be `COLLECTED`** for real labels. Anything else is reported as
  example data and excluded from claims.

```bash
python -m packages.analysis.rq2 --labels data/gold/labels.jsonl
```

### Three things that will bite you

1. **Write the codebook and run the 20-item pilot before the full labelling session.**
   If Krippendorff's alpha comes back below 0.67 on any axis, you revise the codebook and
   re-pilot — and that needs weeks, not days. Keep both rounds; a codebook revision driven
   by measured disagreement is a legitimate, reportable finding and reviewers trust it
   more than one suspiciously high alpha.
2. **The alpha ceiling bounds every RQ2 claim.** A judge cannot be shown more reliable
   than the humans it is compared against. If humans agree at 0.6 on
   `reasoning_faithfulness`, no judge result on that axis can be reported as better.
3. **Sample stratified, not at random.** Random sampling of a working agent yields ~90%
   successes; a judge answering "success" unconditionally would score ~90% while being
   worthless. Target 50 clean successes, 50 clear failures, 50 near-misses.

**A known gap:** the 66-scenario suite contains **no genuine failures** — by design, the
agent is correct on all of them. So the "50 clear failures" stratum cannot be filled from
observed behaviour. You must either inject faults (and flag them as injected) or collect
trajectories from a deliberately weaker configuration. Decide which before you start
sampling.

---

## 4. Live Google Calendar and Gmail

The adapters in `packages/providers/google_calendar.py` and `gmail.py` implement the same
protocols as the fakes and have **never been run against the real API** — no credentials
existed on the build machine. Their tests prove they are self-consistent and handle
Google's documented error dialect; they cannot prove the payload shapes are right.

**Unverified until you run it:** that Calendar accepts the `start`/`end` + `timeZone`
write shape; that `extendedProperties.private` survives a round trip (booking reference
and type live there, and losing them makes every re-notification look like a new
booking); that `reminders.overrides` maps to FR4 alarms as assumed; the OAuth consent
flow itself; real 429/403 bodies matching the `reason` extraction; and Gmail's real MIME
nesting for airline mail.

Setup:

1. Create a Google Cloud project, enable the Calendar and Gmail APIs, create an OAuth
   client, and download the client secret. **Do not commit it** — `credentials.json` and
   `client_secret*.json` are git-ignored.
2. Use a **test account**, not your own calendar. The first live run is when you find out
   whether the write shape is right.
3. Scopes are declared in one place in `packages/providers/oauth.py`, each with a comment
   justifying it. Gmail is **read-only**: the agent never sends or deletes mail.
4. Tokens are encrypted at rest by `packages/providers/tokens.py`.

> **On the token encryption.** It is a documented stdlib construction (PBKDF2-HMAC-SHA256
> → separate encryption and MAC keys, SHA-256 keystream, encrypt-then-MAC verified before
> any decryption). It gives confidentiality against someone holding the file but not the
> passphrase, and it detects tampering and fails closed. It is **not vetted or audited
> crypto.** A production deployment should use `cryptography` (Fernet/AES-GCM) or an OS
> keystore, which also protects the key itself. It cannot distinguish a wrong passphrase
> from tampering — both are a failed MAC, and the error says exactly that.

Swap the provider at the call site; nothing else changes:

```python
from packages.providers.google_calendar import GoogleCalendarProvider
from packages.sync_agent.agent import SyncAgent

agent = SyncAgent(calendar=GoogleCalendarProvider(...))
```

**Keep the fakes as the default for tests and evaluation.** They are what makes the suite
deterministic and free — a 5-model × 3-seed FR10 sweep is 990 runs that cost nothing.
Google is the demo path, not the test path.

---

## After loading real data, re-check these

```bash
python -m pytest tests/ -q                 # nothing may break
python -m packages.bench --all             # FR5 gate: must stay at 0 violations
python -m packages.nlp evaluate --split test
python -m packages.analysis.rq1
python -m packages.analysis.rq2 --labels data/gold/labels.jsonl
```

Numbers measured on synthetic data are a **floor, not a verdict**. When you report
results, say which split they came from, whether the model was tuned on it, and which
figures are real versus not yet collected. `python -m packages.analysis.rq2` prints that
provenance breakdown itself — keep that habit.

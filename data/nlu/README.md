# D1 — Voice-command NLU set

**Status:** pipeline built and measured. Template generation is complete; the LLM paraphrase pass
and the ASR-noise slice are still to run (both need models this machine does not have).
**Purpose:** train and evaluate intent + slot extraction for spoken booking commands (FR1, FR6).

```bash
python -m packages.nlp build                      # generate + leakage-checked splits
python -m packages.nlp evaluate --split test      # score an extractor
python -m packages.nlp evaluate --extractor ollama --model qwen2.5:7b-instruct
```

## What exists now

| Artefact | Location | Notes |
|---|---|---|
| Controlled lexicon | `lexicon.yaml` | 8 slot value lists |
| Template families | `templates.yaml` | 68 families across 7 intents |
| Generated splits | `splits/{train,dev,test}.jsonl` | 712 utterances, split by family |
| Generator | `packages/nlp/generate.py` | exact offsets, deterministic |
| Splitter | `packages/nlp/dataset.py` | family-level, stratified, leakage-checked |
| Metrics | `packages/nlp/metrics.py` | intent / slot F1 / frame + safety confusions |
| Rule-based baseline | `packages/nlp/baseline.py` | the floor, and the LLM fallback |
| Ollama adapter | `packages/nlp/extractor.py` | drop-in; not exercised in CI |

Splits: **train 428 / dev 142 / test 142**, every intent present in every split.

## Results

Rule-based baseline, `python -m packages.nlp evaluate`:

| Split | Intent acc | Slot micro-F1 | **Frame acc** | Safety-critical confusions |
|---|---|---|---|---|
| dev (**tuned on this**) | 1.000 | 1.000 | **1.000** | 0 |
| **test (held out)** | 0.873 | 0.873 | **0.873** | 0 |

**Read the test row, not the dev row.** The baseline's rules were fixed by inspecting dev errors,
so dev is a fitted number and 1.000 there means nothing. The 0.127 gap between the two is a
concrete, local demonstration of why the family-level split exists — and it is the number to
quote in the report when explaining the methodology.

Every remaining test error is one template family, `create_03` ("I need a {booking_type} to
{destination} {date} at {time}"), which no dev family covers: the baseline has no rule for
"I need a…" and falls through to out-of-scope. **It has deliberately not been fixed**, because
adding a rule to close it would mean tuning on the test split. It stays as an honest measure of
how a keyword system generalises to phrasing it has not seen.

**Safety-critical confusions are 0 on both splits.** That counter tracks a decline read as a
confirmation, or out-of-scope read as a cancel/confirm — the errors that manufacture the consent
FR5 depends on. It is reported separately because it is rare enough to disappear inside an
aggregate accuracy figure while being the only error class that can delete a real booking.

## Licence caveat — read before downloading anything

The SRS names **ATIS** and **SNIPS**.

- **SNIPS** — freely redistributable. Use it.
- **ATIS** — the original corpus is **LDC-licensed**. Copies circulate on model hubs, but shipping
  one inside a submitted academic artefact is a licence risk. **Do not vendor it.**
- **MASSIVE** (Amazon, CC BY 4.0) — the licence-safe substitute, with `calendar` and `transport`
  scenarios covering similar ground.

Recommendation: SNIPS + MASSIVE as the external component, ATIS cited as related work only.
Update SRS §1.5 and §7.2 accordingly.

Neither SNIPS nor MASSIVE contains a **reschedule** or **cancel-an-existing-booking** intent —
that lifecycle is the novelty of this project (STRQ1), which is why the templates above exist.

## Schema

```jsonl
{"text": "shift my Chennai flight to half past eight",
 "intent": "reschedule_booking",
 "slots": [{"name": "relative_ref", "start": 6, "end": 23, "value": "my Chennai flight"},
           {"name": "time", "start": 27, "end": 42, "value": "half past eight"}],
 "template_id": "resched_02",
 "source": "template"}
```

**Intents (7):** `create_booking`, `reschedule_booking`, `cancel_booking`, `query_status`,
`confirm_action`, `decline_action`, `out_of_scope`.

**Slots (9):** `booking_type`, `origin`, `destination`, `date`, `time`, `provider`,
`booking_ref`, `party_size`, `relative_ref`.

`confirm_action` / `decline_action` are first-class intents because UC3's confirmation response is
itself an utterance this layer must classify, and misreading one is the worst failure the system
can produce. They are not a yes/no regex.

## Templates, and what they cost

Templates are used instead of hand-written examples because they give **exact slot offsets for
free**. Hand-labelling 700 utterances with character offsets produces silent errors that nothing
downstream detects.

Two template mechanics worth knowing:

- `{slot}` samples from the lexicon; **`{slot=literal}` pins a fixed value**. The second form
  exists because "schedule an appointment with Dr Rao" contains a real `booking_type` that a plain
  template leaves unlabelled — and then every extractor is marked wrong for correctly finding it.
- **`out_of_scope` utterances are generated with no slots at all.** "What's the weather in Chennai"
  contains a city but not a destination; labelling it would teach the extractor to find booking
  fields in sentences that are not bookings.

What templates do **not** give is natural variety. Real speech is messier than any template set,
so **scores on this data are an optimistic ceiling.** The LLM paraphrase pass (below) is what
closes that gap, and until it has run, do not quote these numbers as end-to-end NLU performance.

## Remaining work

1. **LLM paraphrase ×4** per template with slot values resampled, then **human-review 100% of dev
   and test** and a 20% audit of train. Target ~1,400 utterances.
2. **`test_asr_noisy` slice** — run the test split through TTS → faster-whisper → text and keep the
   transcripts against the original gold labels. This is the only measurement of what FR1 actually
   delivers, since users speak rather than type. Report the degradation, don't set a target.
3. **Evaluate the Ollama extractor** against the same splits and compare to the baseline row above.

## Splitting — the one thing not to get wrong

**Split by `template_id`, not by row.** Paraphrases of one seed must never straddle train and test.
Row-level splitting on generated data inflates reported accuracy by 10–20 points because the model
is tested on restatements of what it trained on, and nothing in the numbers looks suspicious — they
just look good.

`split_by_template` calls `assert_no_leakage` itself, so the check cannot be skipped by forgetting
to call it. It raises on a shared template family *or* an identical utterance across splits.
Splitting is also stratified by intent: with only 8 confirm and 8 decline families, an
unstratified shuffle can put zero of them in test.

## Metric targets

| Metric | Target (clean) |
|---|---|
| Intent accuracy | ≥ 0.92 |
| Slot micro-F1 (exact span, = seqeval strict) | ≥ 0.85 |
| **Frame accuracy** | ≥ 0.78 |
| **Safety-critical confusions** | **0** |
| Frame accuracy (ASR-noisy) | report the delta; no target |

Frame accuracy is the metric the build is held to. An utterance with the right intent and a wrong
date produces a confidently wrong calendar entry while slot F1 still looks respectable.

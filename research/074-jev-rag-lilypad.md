# 074 — Third lily pad: Jev-RAG-Claim-Verification-130K

Seed: @MaziyarPanahi, 2026-10-18 — "130K RAG claims verified by Jev", with the
follow-up "Jev + RAG: I kept the disagreements."

Dataset: `MaziyarPanahi/Jev-RAG-Claim-Verification-130K`, **30.4 MB**, 9 parquet
shards, ungated, downloaded to `/home/wubu/models/corpus/micro/`.

## What it actually contains

| Property | Value (from the card) |
|---|---:|
| Typed decisions | 130,000 |
| Upstream response records | 12,118 |
| Distinct query/context pairs | 2,019 |
| Distinct serialized requests | 129,056 |
| Task families / language | 4 / English |
| Columns | 49 |
| Teacher/reference disagreements | 21,181 |
| High-confidence categorical disagreements (≥0.90) | 1,265 |
| Rows with terminal Jev errors | 0 |

Splits are derived, not the upstream RAGTruth protocol: train 125,852,
validation 2,085, test 2,063. **349 of the 2,019 query/context pairs appear in
more than one split**, and 67 exact serialized requests cross splits. Any
evaluation on it must regroup by `group_id` and the shared query/context pair,
or the score is optimistic.

## Why it matters for the decision node

This is the first public corpus where the *same* state is paired with:

- a reference label (`gold_label`, `label_source`),
- the model's choice (`jev_prediction`, `jev_answer_type`),
- its confidence and the **full probability vector** (`jev_probabilities_json`),
- cost and latency (`jev_input_tokens`, `jev_output_tokens`,
  `jev_estimated_cost_usd`, `jev_latency_ms`),

plus the evidence state (`state_json`, `state_preview`, `request_json`).

That makes three things measurable on our side without any guessing:

1. **Coherence.** Feed `jev_probabilities_json` through
   `wubu_decision_complement_ok` / `exclusive_ok` and count violations at
   tol 0.05 — a direct replication of arXiv 2609.33209 on a different corpus.
2. **Calibration.** `wubu_decision_ece(jev_confidence, correct)` where
   `correct = (jev_prediction == gold_label)`. The card explicitly warns the
   scores are not calibrated; this turns that warning into a number.
3. **Defer policy.** The 21,181 disagreements are exactly the population where
   a defer band should fire. A gate tuned to defer on low confidence can be
   scored against the teacher's own misses.

## Licence — read this before using it

The card is `license: other`. Rows record
`mit-with-upstream-review-required`. The seed project (RAGTruth, Particle
Media, 2023) is MIT, but the card states that third-party text, model outputs
and Jev/provider conditions require separate consideration, and that **the card
grants no additional redistribution or model-training rights**.

Therefore, in this repo it is registered as **eval / research only**:

- listed in `docs/micro-datasets.json → eval_only`,
- excluded from every training mix,
- used for measurement (coherence, calibration, deferral), not for distillation.

This is not a formality. A 130K corpus of a commercial model's outputs is a
licensing question, not just a data question.

## Integrity notes

- 130,000 *derived decisions*, not 130,000 human annotations. Several questions
  can come from one response; several responses share one context.
- The card itself says: not a clinical corpus, not a production truth oracle,
  not a context-held-out benchmark.
- Distributions: the disagreement rate (16.3%) is a property of this collection,
  not of Jev in general.

## Next

1. Convert `jev_probabilities_json` into the gate's float arrays and measure
   coherence at tol 0.05 against the published 0.064 figure.
2. Compute ECE by task family; compare to the 0.90-confidence subset.
3. Only after (1) and (2): decide whether the defer threshold is a constant or
   should be per-task-family.

Sources:
- https://x.com/maziyarpanahi/status/2111833867938333120
- https://huggingface.co/datasets/MaziyarPanahi/Jev-RAG-Claim-Verification-130K

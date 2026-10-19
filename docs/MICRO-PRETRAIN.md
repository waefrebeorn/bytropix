# MICRO-PRETRAIN — the smallest corpus that can actually teach the body

Status 2026-10-19. Companion to `docs/MODEL-IDENTITY.md` (the body is
runtime-dimensioned) and `docs/MANIFOLD-MERGING.md` (how trained pieces combine).

## 1. The budget, from the model, not from vibes

Chinchilla-optimal is ~20 tokens/parameter. At the **aligned geometry**
(56,376,832 params) that is **~1.13B tokens**. Two honest observations:

- we are nowhere near it: the SFT/eval micro corpus is 1.01 GB (~250M tokens of
  *instruction* data, not raw text), and the first raw shard is 31.5 MB (~8M
  tokens);
- micro-pretraining is therefore a **staged** programme, not a run: teach syntax
  and factual density first on a few hundred million tokens, then grow the
  corpus with the body (the amoeba doctrine already says "grow the corpus WITH
  the body").

Practical target for the next run: **150–300M tokens** of high-density text.
That is ~600 MB–1.2 GB of raw text at ~4 chars/token.

## 2. The mix (doctrine shape, scaled down)

The lab convergence uses ~50/25/17/8 general/math/code/multi. For a 56M body the
same shape holds, but "general" must be *dense*, not web sludge:

| share | source | why | licence |
|---:|---|---|---|
| 35% | **FineWeb-Edu** (sample-10BT) | the reference educational filter; highest fact density per token | ODC-BY |
| 20% | **Cosmopedia v2 (web_v1)** | synthetic textbook prose written for small-model curricula | Apache-2.0 |
| 15% | **TinyStories** | cheapest way to teach grammar and narrative coherence; tiny vocab | CDLA-Sharing-1.0 |
| 20% | **FineMath (3+)** | filtered educational math; reasoning per token without benchmark leakage | Apache-2.0 |
| 10% | **the-stack-smol** | permissively licensed code; teaches structure and long-range syntax | permissive (per-file) |

Refused, deliberately: **Jev-RAG / StartLux (CC BY-NC)** — research-only, never a
training shard. Eval-only sets stay in `models/corpus/micro` and are never
trained on.

## 3. What exists now

```
models/corpus/micro/          28 datasets, 1.01 GB   (SFT / eval — NOT pretraining)
models/corpus/micro_pretrain/           31.5 MB     (raw text; first shard)
  tinystories/shard-000.jsonl          34,368 stories, sha256 in MANIFEST.json
```

Gathered with:

```
python3 tools/gather_micro_pretrain.py --list
python3 tools/gather_micro_pretrain.py --budget-mb 300 --only tinystories
```

## 4. The gathering trap (learned the hard way)

- **The HF `datasets-server/rows` endpoint rate-limits aggressively** (HTTP 429
  within ~5k rows). Retries with backoff help but do not fix it for real volume.
- **Never open the output shard `wb` before the fetch succeeds** — the first
  version did, and a 429 mid-run truncated a good shard to zero bytes. Shards
  are now written to `.part` and renamed only on success.
- **Use direct file streaming where the repo ships raw text.** TinyStories has
  `TinyStories-train.txt`; an HTTP Range request for the first N bytes is one
  request, no rate limit, and exactly budget-bounded.
- **Parquet-only datasets need a parquet reader.** There is no pyarrow in this
  environment; the honest options are (a) install it, or (b) download a shard and
  decode it with our own C11 parquet reader (a real gap, not a shortcut).

## 5. Training recipe for the micro run

From the lab-convergence doctrine (`wubu-agi-training-pipeline`):

- **Optimizer:** Muon on 2-D matrices, AdamW on 1-D (norms, embeddings, biases).
- **Schedule:** warmup → constant → cosine → **anneal to 0** on upsampled quality
  data. A run that never anneals reads better on loss and worse on samples.
- **Sequence:** 2048, packed; no mid-document truncation.
- **MTP** λ = 0.3.
- **Data order:** general first, math/code in the middle, anneal on the highest
  quality slice (FineWeb-Edu + Cosmopedia).
- **SFT cold-start** afterward at low lr (~1e-5), never mixed into pretrain.

## 6. Gates for the first micro run

1. Held-out loss on a **disjoint** slice (never seen by any shard): report per
   source, not one blended number.
2. A 200-prompt sample audit: coherent English, no degenerate repetition.
3. `wubu_parameter_count` unchanged (geometry untouched by training).
4. Ball containment if any manifold head is active: `‖h‖ < R` on every batch.
5. Throughput honest: tokens/s and wall-clock on this CPU, no extrapolation.

## 7. What this does not claim

A 56M body trained on 300M tokens is **under-trained by lab standards** — the
doctrine says so plainly. This run exists to (a) prove the pipeline end to end on
our own corpus, (b) give the amoeba loop something to grow from, and (c) produce
a checkpoint whose loss curve we trust more than a benchmark score.

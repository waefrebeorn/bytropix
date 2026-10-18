# 073 — Two lily pads: the decision node and the embedding node

Two seeds, both from the first week of October 2026, both describing a model
that is *not* a chat model. For an AGI OS they are the two node brains:
one makes bounded decisions, one makes everything findable.

- **Seed A** — @UnslothAI, 2026-10-07: train your own decision model like Jev
  locally; Qwen3.5-0.8B 20.7% → 74.3% across 3 decision benchmarks on 4 GB VRAM;
  "Clef head" + LoRA r=64, one epoch; downstream 30–37% → 78%.
- **Seed B** — @googlegemma, 2026-10-06: EmbeddingGemma 2, a 740M multimodal
  embedder (text, code, image, video, audio), MRL 768→128, 8K context,
  Apache-2.0, +9.92 points on code over v1.

## Hop 1 — what a decision model actually is

From the Cloudflare launch post and the `Cloudflare/clef` card:

- Input: a **state** (text, JSON, images, video) plus a **schema of typed
  questions**. Question types are `choice`, `score`, `noul` (yes/no).
- Output: one logit per allowed option, per question, in a **single forward
  pass**. Softmax per question gives probabilities. No free-form text, no
  output parsing.
- Clef is a 27B Qwen3.8-27B backbone plus a **joint schema head**: a small
  transformer head that reads the final hidden states, routes evidence from the
  state to each question, and scores all options jointly.
- Clef-Flash is the smaller variant. Both are Apache-2.0, and the API is
  compatible with Jev / SystemOne.

The SystemOne HTTP contract (docs.system-one.dev) is narrow and worth copying:
`POST /v1/systemone`, a `model`, a `state`, and `questions`; limits are 32
questions, 255 options, 64 KB of text. That is a control-plane API, not a chat
API — which is exactly why it fits an OS.

## Hop 2 — what Unsloth shipped

The implementation is public: `unsloth/models/decision.py` defines
`FastDecisionModel` and `DecisionTrainer`, with `HEAD_LEARNING_RATE = 1e-4`,
a `CLEF_RECIPE`, and — the important detail — it copies Cloudflare's
`joint_schema_model.py` byte for byte into the checkpoint so a fine-tune starts
from the released head, not a reimplementation of it.

The training guide (`unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth`)
reports:

| Base | typed-decisions | BANKING77 | CLINC150 | Holdout | VRAM | Time |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 36% → 73% | 7% → 74% | 19% → 76% | 78% | 4 GB | 42 min |
| Qwen3.5-2B | 33% → 78% | 1% → 58% | 1% → 62% | 81% | 8 GB | 40 min |
| Llama 3.2 3B | — | — | — | 79% | 4.1 GB | 30 min |
| Gemma 4 E4B | — | — | — | 77% | 14.4 GB | 49 min |
| Laya (fine-tuned) | — | — | — | 77% | 2.5 GB | 10 min |

Mix: 12 sources plus `typed-decisions` (`ag_news`, `arc`, `banking77`, `boolq`,
`clinc150`, `commonsense_qa`, `mmlu`, `mnli`, `prompt_injections`, `snli`,
`sst5`, `wanli`); 3,000-row test set, decontaminated against training.

The lesson is not "0.8B is smart". It is that **a frozen small LLM plus a
trained typed head beats the same LLM asked to generate a label** by a wide
margin, at a fraction of the cost of a 27B decision model.

## Hop 3 — the published critique, which matters more

Two arXiv papers landed days after Jev, and they are the reason this design has
a gate rather than a softmax and a hope.

**arXiv 2609.32160 — "Typed Decision Models: An Early Evidence Audit and
Evaluation Checklist"** reviews 28 papers from 19–24 September 2026. Finding:
the typed readout has *not* shown an independent accuracy advantage over
comparable label-probability readouts; Jev's clearest gains are latency and
cost, and accuracy gaps remain on harder tasks. They derive a 14-item
evaluation checklist.

**arXiv 2609.33209 — "Beyond Calibration: Do a Typed-Decision Model's
Probabilities Obey the Probability Axioms?"** tests coherence without labels,
on 160 items × three mutually exclusive labels, 480 negation pairs:

- Jev's `p(X)` and `p(not X)` miss summing to 1 by **0.064** on average
  (95% CI 0.055–0.072).
- Its three single-label probabilities sum to **1.14** on average.
- Qwen3.8-27B first-token readout misses by 0.293; verbalized by 0.122.
- Jev's violations are ~5× its repeat noise, and concentrate where it is
  uncertain.

So a decision model's numbers are useful but **not axiomatically coherent**.
Any OS that acts on them must check the arithmetic itself. That is a job for
code, not for the model.

## Hop 4 — the embedding node

From the `google/embeddinggemma-2` card and Google's launch blog:

| Property | Value |
|---|---|
| Total / parts | 740M = 270M text (130M transformer + 140M embedder) + 170M vision + 300M audio |
| Output | 768-d shared space; MRL truncation 512 / 256 / 128 |
| Context | 8,192 tokens, shared across modalities |
| Architecture | 24 layers, 512 model dim, 2048 hidden, 4 heads, GQA 2 local / 1 global KV, sliding window 1024, vocab 262,144, mean pooling, 512→768 projection |
| Licence | Apache-2.0 |
| Languages | 100+ |

Reported scores (full precision): MTEB multilingual 61.36 (v1 61.15), **MTEB
Code 78.68 vs 68.76** (+9.92), MIEB lite 64.64, MMEB image 57.28, VisDoc 67.84,
video 50.67, MSEB retrieval MRR@10 69.54.

Unsloth's runtime notes are operationally important:

- **Do not use FP16** — EmbeddingGemma 2 can produce NaN or degraded embeddings.
  Use BF16 or FP32, including the compute dtype for quantized weights.
- L2-normalize after truncation, not before.
- Start at 768-d; 512/256 for storage; 128 for text-only.
- Use the task prefixes (search query vs document format).

## Hop 5 — convergence: two planes, not two chatbots

The two lily pads are complementary halves of one runtime:

- **Decision node = control plane.** Typed, bounded, cheap, one pass. Answers
  "which of these allowed actions?" with a number. It never writes text, so it
  cannot be prompt-injected into prose; but its *options* are attacker-visible,
  so the schema is part of the attack surface.
- **Embedding node = data plane.** Turns every artifact (file, code, log, audio
  clip, screenshot) into a vector so the OS can retrieve by meaning. MRL lets
  the same index serve a 128-d scan and a 768-d rerank.

Neither is a general LLM. That is the point: the OS keeps its deterministic
core and calls a bounded model where judgement is needed.

## Hop 6 — wiring it into WuBuOS

Proposed namespace, consistent with the 9P tree:

- `/n/decision/ask` — POST state + questions → verdicts with probabilities.
- `/n/index/put` — embed an artifact, store vector + provenance.
- `/n/index/query` — text/audio/image query → top-k with scores.

Policy uses that fit the existing authority model:

1. **Syscall / capability decisions.** State = syscall name, arguments, caller
   caps; question = `allow | deny | defer`. Defer escalates to the armor +
   human. Never a free-text rationale.
2. **Scheduler / admission.** State = queue depth, latency, power; question =
   which policy. Bounded options, no parsing.
3. **Promotion gate.** State = mutation diff + test results; question =
   `promote | reject | quarantine`. This is the natural client of the gate
   below.

Failure modes to design against, from the papers and the API shape:

- incoherent probabilities (0.064 / 1.14 above) → reject or renormalise;
- **schema injection** — an attacker who controls the option text controls the
  decision surface; options must come from a signed schema, not from input;
- confidence hacking — a model that is accurate but overconfident; ECE is a
  gate metric, not a report;
- drift — the head is trained on typed-decisions; OS-specific states are out of
  distribution until fine-tuned.

## Hop 7 — what is built now

`src/wubu_decision.c` + `tools/test_decision.c` (19 checks, in `test_all`):

- exact softmax (the largest entry absorbs the fp32 residual, so the sum is 1
  to within 1e-6 rather than "1.0000001");
- `wubu_decision_choose` with an explicit **defer band** — below threshold the
  OS must not act;
- `wubu_decision_complement_ok` — rejects the measured 0.064 complement miss at
  tol 0.05;
- `wubu_decision_exclusive_ok` — rejects the measured 1.14 label mass and any
  negative probability;
- `wubu_decision_ece` — expected calibration error, so overconfidence is a
  number a gate can threshold.

This is the arithmetic contract. It is deliberately model-agnostic: swap Clef,
Jev, or a 0.8B fine-tune and the gate does not change.

## What is still missing (honest)

- **No runner.** We have the gate, not a model server. Running a 0.8B decision
  head on CPU inside WuBuOS is untested; the published numbers are GPU.
- **No embedding runner.** `ggml-org/embeddinggemma-2-GGUF` exists, and
  llama.cpp can serve embeddings, but nothing is wired to `/n/index` yet.
- **BF16 constraint.** The embedder must not be run in FP16; on CPU that means
  FP32 (2.9 GB for 740M) or a BF16-capable path.
- **Domain shift.** The head would be trained on public typed-decision data;
  WuBuOS states (syscall tables, cap masks) are a different distribution and
  need a small in-domain set before any policy use.
- **256K-token prompts** are a Jev claim we have not verified; our own
  8K-context embedder is the relevant limit here.

## Sources

- https://x.com/unslothai/status/2107868866361930236
- https://x.com/googlegemma/status/2107502533992464482
- https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
- https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision.py
- https://blog.cloudflare.com/clef-decision-models
- https://huggingface.co/Cloudflare/clef
- https://docs.system-one.dev/en/docs/api
- https://huggingface.co/google/embeddinggemma-2
- https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2
- https://unsloth.ai/docs/models/embeddinggemma-2
- arXiv 2609.32160 — Typed Decision Models: An Early Evidence Audit
- arXiv 2609.33209 — Beyond Calibration: probability axioms for TDMs

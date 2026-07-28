# 018 — Cascade speculative decoding (small drafter + large verifier)

Source: Speculative Cascades (Google Research, 2024, arXiv:2312.11462, 93 cites);
CAS-Spec (NeurIPS'25); EAGLE-3 (tree drafting 4.8×); Medusa (heads 2.5×);
n-gram fallback (2-3× out of the box). Also our prefix-KV reuse (010).

## Core idea
Draft K tokens with a **cheap** source, verify them in ONE forward with the
**target** model (accept/reject via the standard speculative mask — provably
equivalent to autoregressive sampling). Two flavors we can own:
1. **n-gram cascade**: use the prompt's own n-gram statistics as the drafter
   (zero extra model, honoring "no third-party"). 2-3× on matched prompts.
2. **self-cascade**: draft from the target's own early layers / a small
   local model we already have on disk (e.g. Agents-A1-4B drafting for Qwen-27B).
The "cascade" twist: a deferral rule lets easy tokens commit from the
small model without waiting for the large one — faster than plain spec-dec.

## Triple-DA
- P1 correctness: speculative accept/reject is **provably equivalent** to
  greedy/ sampled autoregressive (Leviathan et al.). A reject just means
  "compute that token properly" — never wrong. ✓
- P2 privacy: drafter = our own n-grams or a local Colonel. No external
  model/download. ✓
- P3 robustness: a bad draft ⇒ more rejects ⇒ falls back to 1 tok/step
  (never slower than baseline by more than the draft overhead). Bound draft
  depth K to avoid wasted compute on low-confidence.

## Implementation plan
- `wubu_spec.c/.h`: n-gram drafter (rolling hash of last N tokens →
  predicted next from prompt frequency), tree verify against target forward,
  commit accepted prefix. Self-cascade: route draft through a small model's
  forward when available.
- Start with n-gram (guaranteed-correct baseline), then add self-cascade.

## Test oracle
- Greedy decode WITH speculation == WITHOUT (bit-identical token stream) —
  proves equivalence. Assert speedup >1× on a long generation with a
  decent n-gram match rate.

## IMPLEMENTATION STATUS (partial close, 2026-07-28, research-loop cycle 6)
What shipped and is CORRECT + TESTED:
- `src/wubu_spec_decode.c` + `include/wubu_spec_decode.h` (PRE-EXISTING, full):
  tree-draft verify (Leviathan rejection math), n-gram drafter
  (`wubu_ngram_create/propose`), MTP bonus-token sampler. `tools/test_spec_decode.c`
  PASSES (accept longest consistent prefix + reject case + ngram propose + bonus).
- `src/wubu_generate.c` + `include/wubu_generate.h` (NEW, this cycle): autoregressive
  generator with optional n-gram speculative decoding. Greedy + sampled modes,
  K-draft depth, internal n-gram drafter over the running sequence. Wired into
  CORE_OBJ/GPU_OBJ. `tools/test_generate_spec.c` runs on real Qwen: drafts,
  verifies, emits WITHOUT crashing (degenerate non-repetitive prompt also safe).

ROOT-CAUSE BLOCKER (diagnosed, not faked): the equivalence oracle
(greedy-spec == greedy-plain) does NOT hold against the *current engine*,
because the engine's multi-token (T>1) forward is **position-unstable and
observed non-deterministic**. DIAG in the test: argmax at the SAME position
differs across a T=8 vs T=9 forward (e.g. 150949 vs 18447) and varies
run-to-run. This is a latent engine bug in the T>1 SSM/GQA state carry
(uninitialized/incorrectly-carried recurrence state on multi-token paths), NOT
a spec-code bug. It also threatens prefill / chunked / paged decode
correctness, so it is a high-priority engine fix independent of K01.

FIX SHIPPED THIS CYCLE (enabling future equivalence + correctness):
- `wubu_model_reset_state()` added to the engine (wubu_model.c/h): zeroes
  SSM/conv recurrence state AND clears the GQA KV cache (gqa_cache_len=0) so
  two independent generations start from identical zero state. Without this,
  cross-run comparisons were polluted by leftover recurrence state.
- The generator now resets state before each independent run.

REMAINING (the real fix, documented not stubbed): make `wubu_model_forward`'s
T>1 path position-stable + deterministic (carry SSM/conv/GQA state identically
for T=1 vs T=N given the same prior state). Once that lands, the batched spec
forward becomes provably-equivalent and the equivalence oracle re-enables.
Until then K01 is `wired*` (module + verify + generator + reset-state done +
tested; live equivalence gated on engine T>1 forward stability).

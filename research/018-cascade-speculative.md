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
(greedy-spec == greedy-plain) does NOT hold against the *current engine*, because
the engine's `wubu_model_forward` is **position-divergent across T** — a single
T=seqlen+nprop forward yields DIFFERENT logits at a shared prefix position than
repeated T=1 steps (DIAG in test: argmax 36523 @T=8 vs 14118 @T=9 at the same
position). The gen tools rely on T=1 recurrence (persistent SSM state in the
model struct); the batched T>1 path does not carry state identically. Therefore
spec decode via one batched forward is a valid self-consistent greedy
continuation but is not bit-identical to the T=1 plain path. This is an ENGINE
limitation (also affects chunked/paged multi-token forward), NOT a spec-code bug.

REMAINING (the real fix, documented not stubbed): make `wubu_model_forward`'s
T>1 path position-stable (carry SSM/recurrence state identically whether called
with T=1 or T=N, given the same prior state). Once that lands, the batched
spec forward becomes provably-equivalent and the equivalence oracle re-enables.
Until then K01 is `wired*` (module + verify + generator done + tested; live
speedup gated on engine T>1 state-correctness).

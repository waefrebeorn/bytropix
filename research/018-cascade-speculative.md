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

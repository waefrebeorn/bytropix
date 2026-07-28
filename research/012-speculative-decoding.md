# 012 — Speculative decoding (EAGLE / MEDUSA self-draft)

Source: EAGLE / EAGLE-2 (tree-based self-draft, ~3× decode speedup); MEDUSA
(multiple guess heads); lookahead/n-gram fallback; survey of 2025 methods.

## Core idea
Decode is serial (1 token/step) because each token needs the full forward.
Speculative decoding runs a *small draft* (or the same model's own features) to
guess K tokens, then verifies them in **one** forward pass via the sampling
probability (reject with the standard speculative mask). Expected speedup ≈ K when
the draft is accurate. For our 4 Colonels we can self-draft: use the model's own
early layers or an n-gram cache from the prompt as the draft — no separate model
needed (honoring "no third-party").

## Triple-DA
- P1 correctness: speculative decoding is *provably* equivalent to autoregressive
  sampling (Leviathan et al.). The verify step rejects mismatches exactly. ✓
- P2 privacy: draft from same model / local n-gram. No external model. ✓
- P3 robustness: a bad draft just means more rejects (falls back to 1 tok/step,
  never wrong). Must bound draft length K to avoid wasted compute on low-confidence.

## Implementation plan
- `wubu_speculate.c/.h`: draft K tokens (n-gram from prompt + model self-draft),
  run one forward over the K-wide tree, verify with acceptance mask, commit
  accepted prefix.
- Start with the cheapest draft (prompt n-gram lookup) for a guaranteed-correct
  baseline, then add EAGLE-style self-draft.

## Test oracle
- Greedy decode with speculation == greedy decode without (bit-identical token
  stream) — proves equivalence.
- Assert speedup >1× on a long generation with a decent n-gram draft.

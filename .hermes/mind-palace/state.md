# state — May 17 v5 — Full investigation: SSM math + TGT/manifold audit + ssm_a dump

## Previous State
All 14 SSM/GQA math components verified correct. Hidden state still orthogonal (cos-sim 0.0167).

## What Was Done: TGT & Manifold Audit
Traced all functions called by `infer_text.c` (the tool used for reference comparison):

### Euclidean SSM path (`wubu_ssm_forward`):
- **`tgt_safe_expf(gate_s[vh])`** at line 348 — only TGT call in inference path
- This is just `clamp(x, -80, 80); return expf(x)` — a guard against fp32 overflow
- **Verified: gate values never approach ±80.** Ran `check_ssm_a` tool on the model:
  - All 32×30 `ssm_a` values are NEGATIVE (range -72.3 to -0.019)
  - `gate = softplus(alpha + dt_bias) * ssm_a` is ALWAYS negative
  - Typical gate range: -0.001 to -10, max extreme: ~-50
  - `exp(gate)` is always in (0, 1] — no overflow possible
- **Verdict: tgt_safe_expf is harmless in practice.**

### Euclidean GQA path:
- `gqa_kv_decode` (decode): NO TGT calls, clean attention
- Inline prefill code: NO TGT calls, clean attention
- **`wubu_gqa_forward`** has `tgt_wrap(score)` at line 1224 but **infer_text.c never calls it.**

### Manifold/Hyperbolic code:
- `wubu_poincare_ssm_forward` — **NOT called by infer_text.c** (only test files)
- Zero contamination of the inference path.

### ssm_a confirmed correct for DeltaNet formulation
`gate = softplus(α + dt_bias) * ssm_a` where ssm_a = -A_hat (all negative).
Decay λ = exp(gate) = exp(-softplus(Δ) * A_hat) ∈ (0, 1]. Correct.

## Conclusion: Not TGT, not Manifold
The bug is NOT in TGT wrapping or hyperbolic math. Those are cleanly isolated from inference.

## Root Cause Must Be
1. **Weight data corruption** — a tensor loaded with wrong strides/offsets
2. **Dequant bug** — one of the 7 quant types (Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q4_K, F32) has wrong dequant for a specific shape
3. **SOMETHING no amount of code reading can find** — must use layer-by-layer dumps

## NEXT STEP
Layer-by-layer comparison: build a tool that dumps hidden state after EACH layer from both engines, find first divergence.

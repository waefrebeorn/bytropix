# WuBuText AI — State Dashboard (May 17 AM v14 — HONEST)

## Ground Truth
**INFERENCE STILL BROKEN.** MoE interleaved dequant FIXED, output projection verified correct, but SSM output cos_sim=0.009 vs reference.
Prompt "Hello": our top token "incer" (10.38), reference token "Here's a" (reference produces coherent text).

## What We Know
- ✅ MoE expert dequant: interleaved [D_MODEL, D_FF, N_EXPERTS] block-by-block extraction FIXED
- ✅ MoE type dispatch: down_exps uses ty_gd (per-layer IQ4_XS) — FIXED
- ✅ Per-layer debug dumps: DUMP_LAYER_DIR, DUMP_HIDDEN_NORM, DUMP_SSM_VAL all working
- ✅ MAX_LAYERS env var: for single-layer debugging
- ✅ Q scaling in SSM: 1/sqrt(128) applied matching reference
- ✅ ssm_a values verified: ALL NEGATIVE (-72 to -0.02) → gate formula IS correct decay
- ✅ All weight matrix accesses use correct ggml row-major indexing (i + j * ne[0] pattern)
- ✅ Q5_K, Q6_K dequant functions verified against llama.cpp reference — correct
- ✅ SSM output projection indexing (i + j * VALUE_DIM) — correct for ggml layout

## Divergence Point
- **SSM VALUE_DIM output cos_sim = 0.009** vs reference → bug is in SSM recurrence or QKV projection, NOT output projection
- L0 residual RMS ratio ~8× (our larger than ref)
- Final h_last cos_sim = 0.004

## Unchecked Hypotheses
1. **Conv1d bug**: `kernel[ki + c * k]` access — need to verify conv output element-by-element
2. **MoE routing at layer 0 shared expert**: may corrupt residual in first MoE step
3. **lm_head (output.weight) dequant**: type unknown, could be wrong
4. **wubu_silu ≠ ggml_silu**: slight numerical difference could compound
5. **SSM state initialization**: zero-initialized matches reference, but state_update_target semantics may differ
6. **GQA layer at layer 3+**: our 38 SSM layers + 2 GQA layers match reference

## Fixed This Session
- MoE interleaved expert dequant (block-by-block, per-expert extraction)
- MoE down_exps type dispatch (ty_gd for IQ4_XS layers)
- Debug infra: DUMP_LAYER_DIR, DUMP_HIDDEN_NORM, MAX_LAYERS, DUMP_SSM_VAL
- Misguided output projection "fix" reverted (was already correct)

## Priorities
P0 — Find root cause in SSM forward: compare QKV/element-wise with reference
P1 — Verify conv1d output element-by-element
P2 — Sanity-check lm_head dequant

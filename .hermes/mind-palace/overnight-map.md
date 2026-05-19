# Overnight Map — May 19, 2026

## BREAKTHROUGH: Q6_K vec_dot bug fixed — cos-sim 0.9967 ✅

### The Bug
Q6_K shared expert output projection had a loop iteration count error in the AVX2 vec_dot:
- `quantized_dot_generic.c` line 314: `j < QK_K/32` → should be `j < QK_K/16`
- Result: only 128/256 elements processed per block (50% coverage)
- Q6_K matmul cos-sim vs F32: 0.728 → now 0.99986
- Full model cos-sim: 0.794 → now 0.9967

### Post-Mortem on DA v10
DA v10's analysis was WRONG about the root cause:
- Claimed: "MoE softmax vs sigmoid gating" 
- Reality: both llama.cpp and bytropix use softmax
- The GQA interleave fix (per-head Q/gate interleaved) WAS correct and DID improve attn-only
- The remaining 0.79 was from the Q6_K bug, not MoE gating

### Phase 9 Status: QUANTIZED-ONLY PATH — VERIFIED ✅
- Q5_K verified: cos-sim 0.9999 vs F32
- Q6_K FIXED: cos-sim 0.99986 vs F32
- IQ2_XXS verified: max diff 0.002 vs F32
- All quant types now verified accurate

### Next: Phase 10 — infer_text Pipeline
- tools/infer_text.c exists
- Needs prob_idx fix and quantized path integration
- Full text generation from prompt

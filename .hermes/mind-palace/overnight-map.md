# WuBuText AI — Overnight Navigation Map (May 16 v17 — DA RECERTIFIED)

## Where We Are
Full component deep dive complete. Inference pipeline STRUCTURE verified correct.
Output STILL WRONG. Root cause narrowed to 5 suspects.

## What's New This Session
- Full audit of ALL components vs llama.cpp reference
- 0% verification rate: all P0-P3 were "compiles only", never runtime-verified
- wubu_gqa_forward() indexing bug found (dead code)
- TGT wrapping identified as llama.cpp divergence (clips scores to [-π,π])
- DA stripped ALL false ✅ from status table

## 5 Active Suspects (priority order)
1. Q5_K dequant (181 tensors)
2. Output weight Q4_K (type 12) dequant
3. SSM Q scaling factor
4. RMSNorm epsilon
5. TGT wrapping in attention

## Next Session
1. Write Q5_K dequant test vs llama.cpp reference
2. Or: run layer-0-only with DUMP_LAYER_DIR and compare h_last directly

## Clean Paths
- infer_text.c is the right inference path (NOT wubu_model_forward_from_embd)
- SSM recurrence structure is correct — the math matches delta-net-base
- Weight loading + dequant pipeline is the most likely failure point

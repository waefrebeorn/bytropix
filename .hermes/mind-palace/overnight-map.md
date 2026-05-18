# Overnight Map — May 18, 2026

## Session Summary
Wired MoE quantized path. Found REAL blocker: SSM/GQA forward is broken even with all-F32 math. Previous sessions traced SSM Layer 0 cos_sim=0.40 vs ref.

## Completed
- MoE quantized pointers saved from GGUF blob (routed + shared experts)
- wubu_moe_forward: shared expert uses quantized_matmul (Q5_K/Q6_K)
- wubu_moe_forward: routed experts use quantized_matmul (IQ2_XXS/IQ3_XXS/IQ4_XS)
- load_from_blob flag to prevent blob-pointer free crash
- Cos-sim goes from -0.51 (F32 MoE) to -0.65 (quant MoE) — both broken by SSM/GQA
- Updated all mind-palace docs with HONEST assessment

## Real Blocker (unrelated to MoE quantization)
The SSM/GQA forward pass produces wrong output even with ALL-F32 SGEMM. 
Root cause is in the forward math, NOT in weight quantization.
Previous sessions found SSM L0 cos_sim=0.40 vs ref.

## Next Session
1. Generate per-layer reference dump from llama.cpp
2. Find first diverging layer (likely L0 SSM)
3. Fix SSM forward against qwen3next.cpp reference
4. Fix GQA layers (separate Q/K/V weight split)
5. Re-test with quantized MoE path (already wired, should work)

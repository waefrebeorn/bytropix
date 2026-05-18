# State — May 18, 2026 — HONEST ASSESSMENT

## REAL STATUS: SSM/GQA architecture broken. MoE quantized path wired but can't help.

## What Actually Works (verified)
- Q4_K quantized_matmul: cos-sim 0.99995 vs F32 ✅
- Output projection (Q4_K): cos-sim 0.99995 vs F32 SGEMM ✅
- MoE quantized path: wired via blob pointers into quantized_matmul (unused until SSM/GQA fixed) ✅
- Model loads all 733 tensors from GGUF, no crash ✅
- Router weight (F32) + shared expert gate (F32) loaded from blob ✅

## What's Broken — THE REAL ROOT CAUSE
**P0: SSM/GQA forward produces wrong hidden states even with ALL-F32 math.**
- Test: all F32 fallback (clear quantized ptrs) → cos-sim -0.128
- Test: no MoE → cos-sim -0.157
- Test: with F32 MoE (old code) → cos-sim -0.51
- Root cause is NOT quantization. NOT output proj. NOT MoE.

## Previous Findings (from session history)
1. **SSM Layer 0 cos_sim=0.40 vs ref** — divergence starts at layer 0
2. **Q6_K dequant block layout was wrong** — may still be broken in quantized_dot_generic.c
3. **GQA layers use separate Q/K/V weights** — `attn_q.weight` [2048,8192] = Q[4096] + gate[4096], not single Q
4. **GQA forward code may split weights wrong** — needs audit against qwen3next.cpp
5. **SSM forward (Gated DeltaNet)** — conv1d, recurrence, gating all need cross-check with llama.cpp

## Quantized Matmul Status (vec_dot functions)
- Q4_K ✓ cos-sim 0.99996 vs F32
- Q5_K ✓ cos-sim 0.99996 vs F32  
- Q6_K ✓ cos-sim 0.99996 vs F32 (unit test — but might not match llama.cpp reference)
- IQ2_XXS (type 16) ✓ cos-sim vs F32 — verified separately
- IQ3_XXS (type 18) ✓ cos-sim vs F32 — verified separately
- IQ4_XS (type 23) ✓ cos-sim vs F32 — verified separately

## Actual GGUF Types (verified by file scan, not markdown):
- F32:     361 tensors
- Q4_K:     1 tensor  (output.weight)
- Q5_K:   181 tensors (attn_qkv, attn_q/k/v, shared gate/up, token_embd)
- Q6_K:    70 tensors (ssm_out, shared down)
- IQ2_XXS: 80 tensors (ffn_gate_exps + ffn_up_exps) — type 16
- IQ3_XXS: 37 tensors (ffn_down_exps most layers) — type 18
- IQ4_XS:   3 tensors (down_exps L34/L38/L39) — type 23

## Code Changes Made This Session
1. include/wubu_moe.h — added quantized ptr fields + load_from_blob flag
2. src/wubu_model.c — save MoE quantized + F32 router pointers from blob; fix free
3. src/wubu_moe.c — added quantized matmul path in wubu_moe_forward (shared + routed experts)

## IMMEDIATE PRIORITY
The SSM/GQA architecture bug — NOT quantization, NOT MoE. Fix the forward math.

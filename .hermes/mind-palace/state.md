# State — May 18, 2026 — POST-FIX

## REAL STATUS: GQA interleave bug FIXED. Cos-sim 0.9968 vs reference.

## Verified Runtime Results
- Full 40L + quantized MoE: cos-sim **0.9968146876** vs llama.cpp
- All 40 layers cos-sim > 0.995 (no divergence at any layer)
- Per-layer comparison via DUMP_LAYER_DIR: our vs ref every layer ✓
- Output projection Q4_K: cos-sim 0.99995 vs F32 SGEMM ✓

## What's Wired
- SSM projections (QKV, gate, output): quantized_matmul (Q5_K/Q6_K) ✓
- GQA projections (Q, K, V, output): quantized_matmul (Q5_K) ✓ after interleave fix
- MoE shared expert (gate/up/down): quantized_matmul (Q5_K/Q6_K) ✓
- MoE routed experts (gate/up/down): quantized_matmul (IQ2_XXS/IQ3_XXS/IQ4_XS) ✓
- Router (ffn_gate_inp): F32 from blob pointer ✓
- Shared expert gate (ffn_gate_inp_shexp): F32 from blob pointer ✓
- All via load_from_blob flag (no double-free) ✓

## Actual GGUF Types (verified by dump_tensor_types — NOT markdown guesses):
| Type ID | Name     | Count | Used For |
|---------|----------|-------|----------|
| 0       | F32      | 361   | Norms, biases, routers, small proj |
| 12      | Q4_K     | 1     | output.weight ONLY |
| 13      | Q5_K     | 181   | attn_qkv, attn_q/k/v, attn_gate, shared gate/up, token_embd |
| 14      | Q6_K     | 70    | SSM output proj, shared down |
| 16      | IQ2_XXS  | 80    | MoE gate_exps + up_exps (NOT "IQ2_XS"!) |
| 18      | IQ3_XXS  | 37    | MoE down_exps (NOT "IQ3_XS"!) |
| 23      | IQ4_XS   | 3     | down_exps L34/L38/L39 (NOT "Q3_S_XL"!) |

## Remaining Gap (0.9968 → 1.0)
The 0.003 gap is from quantized_matmul's Q8_K input quantization + generic C vec_dot (vs llama.cpp's SIMD). Each layer accumulates ~0.0003 error from quantization noise.
- Q4_K/Q5_K/Q6_K vec_dot: cos-sim 0.99996 vs F32 SGEMM (unit test)
- IQ2_XXS/IQ3_XXS/IQ4_XS vec_dot: cos-sim 0.9999 vs F32 SGEMM (unit test)
- The gap is in the vec_dot functions vs llama.cpp's reference implementations, not the architecture

## Code Files Changed This Session
1. include/wubu_moe.h — quantized ptr fields + load_from_blob flag
2. src/wubu_moe.c — quantized matmul path for MoE (shared + routed)
3. src/wubu_model.c — save MoE pointers from blob; fix free; per-layer dump
4. src/wubu_ssm.c — GQA Q/gate interleave fix (THE root cause)
5. src/llama-context.cpp — per-layer dump infrastructure (debug helper)
6. src/models/qwen35moe.cpp — LLAMA_DUMP_LAYERS support (debug helper)

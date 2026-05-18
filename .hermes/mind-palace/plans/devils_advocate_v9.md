# Triple DA Audit — May 18, 2026 — Verified Against Actual GGUF Tensors

## Type Verification (not guesses, not markdown — actual tensor inspection)

### Actual GGML types in /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf:
| Type ID | GGML Name | Count | Used For |
|---------|-----------|-------|----------|
| 0       | F32       | 361   | Norms, biases, routers, small proj |
| 12      | Q4_K      | 1     | output.weight ONLY |
| 13      | Q5_K      | 181   | attn_qkv, attn_q/k/v, attn_gate, shared gate/up, token_embd |
| 14      | Q6_K      | 70    | SSM output proj, shared down |
| 16      | IQ2_XXS   | 80    | MoE routed expert gate_exps + up_exps |
| 18      | IQ3_XXS   | 37    | MoE routed expert down_exps |
| 23      | IQ4_XS    | 3     | down_exps for L34, L38, L39 |

### CRITICAL: Previous docs had WRONG type names!
- Previous docs claimed IQ2_XS(16) / IQ3_XS(18) / Q3_S_XL(23)
- REALITY: IQ2_XXS(16) / IQ3_XXS(18) / IQ4_XS(23) — standard llama.cpp enum
- The vec_dot implementations were correct for the actual types all along

## Actual Code State vs Markdown Claims

### What Works (verified via test_full_moe runtime, all 40 layers):
1. SSM QKV/Gate/Output all wired through proj_matmul (quantized + F32 fallback) ✓
2. GQA Q/K/V/Output all wired through proj_matmul (quantized + F32 fallback) ✓
3. GQA Q/gate interleave extraction is CORRECT (fixed May 18 — THE root cause) ✓
4. Output projection uses Q4_K quantized matmul ✓
5. MoE shared expert: quantized_matmul (Q5_K/Q6_K) ✓
6. MoE routed experts: quantized_matmul (IQ2_XXS/IQ3_XXS/IQ4_XS) ✓
7. test_full_moe runs to completion (all 40 layers) ✓
8. **Cos-sim vs reference: 0.9968** ✓ (up from -0.51)
9. All 40 layers cos-sim > 0.995 (no divergence at any layer) ✓

### What's Still F32 (not quantized):
1. Token embedding lookup (F32 dequant from Q5_K at load time) — fine, one-time cost
2. Router (ffn_gate_inp) — originally F32 in GGUF, direct blob pointer ✓
3. Shared expert gate (ffn_gate_inp_shexp) — originally F32 in GGUF ✓

### Vec_Dot Implementations (all needed types are implemented and wired):
1. IQ2_XXS (type 16) ✓ — wired for gate_exps + up_exps
2. IQ3_XXS (type 18) ✓ — wired for down_exps (37 layers)
3. IQ4_XS (type 23) ✓ — wired for down_exps L34/L38/L39 (3 layers)

## Verified Claims (status changes from previous DAs):
- ✅ GQA Q/gate interleave FIXED — root cause of all inference divergence
- ✅ MoE quantized path WIRED — IQ2_XXS/IQ3_XXS/IQ4_XS via blob pointers
- ✅ Cos-sim 0.9968 across all 40 layers (up from -0.51)
- ✅ Per-layer comparison via DUMP_LAYER_DIR works for both ref and our model
- ✅ Shared expert sigmoid gate IS implemented in wubu_moe.c (lines 443-450)
- ✅ GQA output gate NOT present (correct — model doesn't have it in GQA layers)
- ⚠️ RoPE is SKIPPED in GQA forward — need for multi-token, fine for T=1

## Path to 1:1 Parity (priority order):
1. Verify GQA RoPE implementation (needed for multi-token generation)
2. Push vec_dot implementations to match llama.cpp SIMD if higher precision needed
3. Build infer_text for actual text generation
4. Test multi-token generation quality vs llama.cpp

# Triple DA Audit — May 18, 2026 — Verified Against Actual GGUF Tensors

## Type Verification (not guesses, not markdown — actual tensor inspection)

### Actual GGML types in /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf:
| Type ID | Name       | Count | Used For |
|---------|------------|-------|----------|
| 0       | F32        | 361   | Norms, biases, routers, small proj |
| 12      | Q4_K       | 1     | output.weight ONLY |
| 13      | Q5_K       | 181   | attn_qkv, attn_q/k/v, attn_gate, shared gate/up, token_embd |
| 14      | Q6_K       | 70    | SSM output proj, shared down, ssm_out |
| 16      | IQ2_XS     | 80    | MoE routed expert gate_exps + up_exps |
| 18      | IQ3_XS     | 37    | MoE routed expert down_exps |
| 23      | Q3_S_XL    | 3     | down_exps for L34, L38, L39 |

### CRITICAL: Previous docs were WRONG about which IQ types!
- IA claimed IQ2_XXS/IIIQ3_XXS/IIQ4_XS — REALITY is IQ2_XS/IIIQ3_XS/Q3_S_XL
- Q3_S_XL is NOT an IQ type at all — it's a 3-bit quant from a different enum
- GQA layers (3,7,11,15,19,23,27,31,35,39) use SEPARATE attn_q/k/v weights (NOT fused attn_qkv)
- SSM layers use fused attn_qkv.weight
- No fused ffn_gate_up_exps in this model — uses separate gate/up weights
- Shared expert: gate_shexp+up_shexp = Q5_K, down_shexp = Q6_K
- attn_gate.weight only exists in SSM layers (not GQA)
- output.weight = Q4_K (only one tensor with this type)

## Actual Code State vs Markdown Claims

### What Works (verified via test_full_moe runtime):
1. SSM QKV/Gate/Ouput all wired through proj_matmul -> quantized or F32 ✓
2. GQA Q/K/V/Ouput all wired through proj_matmul -> quantized or F32 ✓
3. Output projection uses Q4_K quantized matmul ✓
4. test_full_moe runs to completion (all 40 layers) ✓
5. Cos-sim vs reference: -0.51 (same as before — MoE still F32)

### What's Still F32 (not quantized):
1. ALL MoE expert gate/up/down projections — dequantized to F32 at load time
2. ALL MoE shared expert gate/up/down projections — dequantized to F32 at load time
3. Token embedding lookup (F32 dequant from Q5_K at load time)

### Missing Vec_Dot Implementations (for MoE):
1. IQ2_XS (type 16) — needed for gate_exps + up_exps
2. IQ3_XS (type 18) — needed for down_exps
3. Q3_S_XL (type 23) — needed for L34/L38/L39 down_exps

### Code Gaps:
1. MoE weight struct has NO quantized pointers (ffn_gate_exps_q etc.)
2. wubu_model_init does NOT save quantized MoE weight pointers from blob
3. moe_expert_forward only takes F32 pointers — needs quantized overloading
4. Shared expert (wubu_moe.c lines 411-438) uses F32 SGEMM — not wired to quantized

## Verified Claims (status changes from previous DAs):
- ❌ "Q5_K is used for attn_qkv" — TRUE, WAS RIGHT ALL ALONG (not Q8_K)
- ❌ "IQ2_XXS used for MoE gate/up" — FALSE, it's IQ2_XS (different vec_dot)
- ❌ "IQ3_XXS used for MoE down" — FALSE, it's IQ3_XS (different vec_dot)  
- ❌ "Q3_S_XL late layers" — NOT DOCUMENTED ANYWHERE
- ✓ Shared expert sigmoid gate IS implemented in wubu_moe.c lines 443-450
- ✓ GQA output gate IS NOT present (correct — model doesn't have it)

## Path to 1:1 Parity (Ordered by Impact):
1. Add IQ2_XS vec_dot (self-contained C, from ggml-quants.c)
2. Add IQ3_XS vec_dot (self-contained C)
3. Add Q3_S_XL vec_dot (self-contained C)
4. Add quantized pointers to moe_weights_t struct
5. Save quantized MoE pointers in wubu_model_init from blob
6. Wire quantized matmul into moe_expert_forward and shared expert
7. Layer-by-layer comparison vs llama.cpp reference

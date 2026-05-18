# Qwen3.6-35B-A3B-UD-IQ2_M Research

## Architecture (qwen35moe)
- 40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (Gated Attention)
- Pattern: 3 SSM → 1 GQA, repeated 10x. SSM: L0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38 (30)
  GQA: L3,7,11,15,19,23,27,31,35,39 (10)
- D_MODEL=2048, D_FF=512, SHARED_D_FF=512, N_EXPERTS=256, N_ACTIVE_EXPTS=8
- Vocab: 248320 (padded), BOS=EOS=248044
- SSM: 16 QK heads, 32 V heads, d_state=128, conv_kernel=4
- GQA: 16 Q heads, 2 KV heads, head_dim=256
- MoE: 256 experts, top-8 routed + 1 shared (activated per token)

## Actual GGML Types (verified May 18 via tensor inspection — NOT markdown guesses)
| Type ID | GGML Name | Count | Used For |
|---------|-----------|-------|----------|
| 0       | F32       | 361   | Norms, biases, routers, small proj |
| 12      | Q4_K      | 1     | output.weight ONLY |
| 13      | Q5_K      | 181   | attn_qkv, attn_q/k/v, attn_gate, shared gate/up, token_embd |
| 14      | Q6_K      | 70    | SSM output proj, shared down |
| 16      | IQ2_XXS   | 80    | MoE ffn_gate_exps + ffn_up_exps |
| 18      | IQ3_XXS   | 37    | MoE ffn_down_exps (all except L34,38,39) |
| 23      | IQ4_XS    | 3     | MoE ffn_down_exps L34, L38, L39 |

IMPORTANT: Earlier docs claimed "IQ2_XS(type16)", "IQ3_XS(type18)", "Q3_S_XL(type23)".
THESE ARE WRONG. The actual types are IQ2_XXS, IQ3_XXS, IQ4_XS — matching standard llama.cpp enum.

NOTE: "IQ2_M" in the filename is a LABEL for overall compression level, NOT the tensor type.
The actual types per tensor vary as defined by Unsloth Dynamic 2.0 quantization.

## Key Architecture Details
- SSM layers (not GQA) have attn_gate.weight (sigmoid gate on SSM output before output_proj)
- GQA layers (not SSM) use SEPARATE attn_q.weight, attn_k.weight, attn_v.weight (NOT fused attn_qkv)
- No fused ffn_gate_up_exps in this model — uses separate gate/up weights
- Experts stored contiguous: dims=[D_MODEL, D_FF, N_EXPERTS] (dims[0] innermost)
- Shared expert: gate_shexp + up_shexp = Q5_K, down_shexp = Q6_K, gate_inp_shexp = F32
- Router: ffn_gate_inp.weight = F32 [2048, 256]

## Comparison bytropix vs Reference (llama.cpp)
- SSM: uses proj_matmul (quantized path) — correct
- GQA: uses proj_matmul (quantized path) — correct  
- Output: Q4_K quantized matmul — correct
- MoE: STILL F32 dequant+SGEMM — NOT wired to quantized path
- Missing vec_dot: IQ2_XS, IQ3_XS, Q3_S_XL

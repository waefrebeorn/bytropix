# Reference Architectures

## Primary Target: Qwen3.6-35B-A3B

### Architecture Summary (verified via GGUF tensor inspection May 18)
- Type: Causal LM + Vision Encoder (qwen35moe architecture)
- 40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (Gated Attention)
- Pattern: 3:1 repeating (3 SSM → 1 GQA)

### SSM Layers (30 of 40)
- attn_qkv: [2048, 8192] Q5_K — fused Q_full, K_full, V_full
- ssm_conv1d: [4, 8192] F32 — depthwise conv (kernel=4)
- ssm_a, ssm_b: [2048, 32] F32
- ssm_alpha: [2048, 32] F32
- ssm_dt.bias: [32] F32
- ssm_out: [4096, 2048] Q6_K
- attn_gate: [2048, 4096] Q5_K — sigmoid output gate (SSM layers ONLY)

### GQA Layers (10 of 40)
- SEPARATE weights (NOT fused attn_qkv):
  - attn_q: [2048, 8192] Q5_K — fused Q + gate
  - attn_k: [2048, 512] Q5_K
  - attn_v: [2048, 512] Q5_K
  - attn_output: [4096, 2048] Q5_K
  - attn_q_norm: [256] F32
  - attn_k_norm: [256] F32
- NO attn_gate.weight in GQA layers

### MoE (all 40 layers)
- Router: ffn_gate_inp.weight [2048, 256] F32
- Routed experts: 256, top-8 activated
  - ffn_gate_exps: [2048, 512, 256] IQ2_XS (NOT IQ2_XXS!)
  - ffn_up_exps: [2048, 512, 256] IQ2_XS
  - ffn_down_exps: [512, 2048, 256] IQ3_XS (L34/38/39: Q3_S_XL)
- Shared expert (1):
  - ffn_gate_shexp: [2048, 512] Q5_K
  - ffn_up_shexp: [2048, 512] Q5_K
  - ffn_down_shexp: [512, 2048] Q6_K
  - ffn_gate_inp_shexp: [2048] F32 (sigmoid gate)

### Output
- output.weight: [2048, 248320] Q4_K (only Q4_K tensor in entire model)
- token_embd.weight: [2048, 248320] Q5_K
- output_norm.weight: [2048] F32

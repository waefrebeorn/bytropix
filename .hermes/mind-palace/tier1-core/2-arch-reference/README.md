# Reference Architectures

## Primary Target: Qwen3.6-35B-A3B

Full reference: `.hermes/research/papers/Qwen/Qwen3.6-35B_Arch_Reference.md`
Grafting strategy: `.hermes/research/papers/Qwen/Embedding_Grafting_Research.md`
Actual GGUF tensor layout: `.hermes/mind-palace/tier3-impl/8-embedding-graft/plan.md`

### Architecture Summary (Corrected after GGUF inspection)
```
Type: Causal LM + Vision Encoder (qwen35moe architecture)
Text:
  hidden_size: 2048
  layers: 40 (30 linear_attention + 10 full_attention), pattern 3:1 repeating
  Attention type A (30 layers): Mamba2-style SSM with:
    - attn_qkv: [2048, 8192] Q8_K — fused Q_full, K_full, V_full, Q_linear, V_linear
    - ssm_conv1d: [4, 8192] F32 — depthwise conv (kernel=4)
    - ssm_a: [32] F32 — recurrent coefficient (A = -exp(ssm_a))
    - ssm_alpha: [2048, 32] F32 — input gate projection
    - ssm_beta: [2048, 32] F32 — input projection
    - ssm_dt.bias: [32] F32 — time-step bias
    - ssm_out: [4096, 2048] F32 — output projection
    - attn_gate: [2048, 4096] Q8_K — output gate
  Attention type B (10 layers): Standard GQA:
    - Uses same attn_qkv weights but with full softmax attention
    - 16 Q heads × 256, 2 KV heads × 256
  MoE: 256 experts, IQ2_XS (gate/up) + IQ1_S (down), 8 routed + 1 shared (Q8_K)
  vocab: 248320 (BBPE, pads at scattered indices — 73 zero embeddings found)
  ctx: 262K native, 1M extensible
Vision:
  depth: 27, hidden: 1152, heads: 16
  patch: 16, temporal: 2 (3D ViT)
  out: 2048
```

**CRITICAL CORRECTION:** The attention is NOT "Gated DeltaNet" — it's a structured SSM
(Mamba2-style). The "Gated DeltaNet" name in the Qwen model card is descriptive marketing,
not a reference to the DeltaNet paper. The actual implementation uses selective scan with
`ssm_a`, `ssm_dt`, `ssm_alpha`, `ssm_beta` tensors. The hyperbolic gyration replacement
works the same way regardless (replace linear recurrence with Möbius addition), but the
exact scan algorithm differs.

## Secondary Reference: DeepSeek-V2 MLA

File: `llama.cpp/src/models/deepseek2.cpp` (262 lines)

### MLA Attention
```
- Latent compressed KV: c = W_kv * [k; v]  (dim compression)
- K = W_uk * c, V = W_uv * c  (decompression)
- Q = W_q * x
- Multi-head with RoPE
- No MoE (small V2)
```

## Secondary Reference: DeepSeek-V3

File: `llama.cpp/src/models/deepseek_v3.cpp` (450 lines)
- MLA attention (same as V2)
- DeepSeekMoE (fine-grained, shared expert)
- Multi-token prediction head

## Secondary Reference: Qwen2.5

File: `llama.cpp/src/models/qwen2.5.cpp`
- GQA only (no Gated DeltaNet)
- Used for understanding the GGUF format for embedding extraction

## How They Fit Together

| Feature | Source | For WuBu |
|---------|--------|----------|
| Gated DeltaNet | Qwen3.5+ | Replace linear recurrence with hyperbolic gyration |
| MLA | DeepSeek-V2 | Alternative: latent compressed KV for extreme efficiency |
| GQA | Both | Keep for the 25% full attention layers |
| MoE with shared expert | Qwen3.5+ | Router replaced with wubu nested geometry |
| DSA (Dynamic Sparse Attention) | DeepSeek-V3.2 | Long-context sparse attention for 1M+ ctx |
| MTP | Both | Multi-token prediction head for training efficiency |

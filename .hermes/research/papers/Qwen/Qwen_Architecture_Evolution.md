# Qwen Architecture Evolution: Qwen3 → Qwen3.5 → Qwen3.6

## Timeline

| Release | Date | Key Innovation |
|---------|------|----------------|
| Qwen3 (2504) | 2025-04-29 | GQA + full softmax, 36T tokens, 151669 vocab |
| Qwen3-2507 | 2025-08 | 1M context extension, thinking/non-thinking modes |
| Qwen3-Next-80B-A3B | 2025-09 | Ultra-sparse MoE with hybrid attention |
| Qwen3.5 | 2026-02-16 | **Gated DeltaNet** + sparse MoE, 248320 vocab, 262K ctx |
| Qwen3.6-35B-A3B | 2026-04-16 | Agentic coding, thinking preservation |
| Qwen3.6-27B | 2026-04-22 | Dense flagship, swish output gate |

## Architecture Comparison

### Attention Mechanism

| Generation | Attention Type | Ratio | Head Dim |
|------------|---------------|-------|----------|
| **Qwen3** | GQA full softmax | 100% full | 128 (implied) |
| **Qwen3.5/3.6** | Gated DeltaNet + GQA | **75% linear, 25% full** | 128 linear, 256 full |

Gated DeltaNet = linear attention with data-dependent gating (`attn_output_gate=true`).
It's a **gated version of DeltaNet** — linear complexity O(n), replaces softmax with element-wise gating.

### Linear Attention Details (Gated DeltaNet)

```
z_t = sigmoid(W_z x_t)        // input gate
h_t = lambda_t * h_{t-1} + z_t ⊙ (W_v x_t)   // linear recurrence
o_t = (W_q x_t)^T h_t        // query-key-value retrieval  
g_t = sigmoid(W_g x_t)        // output gate
y_t = g_t ⊙ o_t              // gated output
```

Key difference from standard linear attention: **output gating** via `attn_output_gate=true` and **convolution kernel** `linear_conv_kernel_dim=4` for local context mixing.

### MoE Evolution

| Generation | Total Experts | Active | Shared Expert | Intermed Size |
|------------|--------------|--------|---------------|---------------|
| Qwen3 MoE | 128 | 8 | **No** | (same as dense) |
| Qwen3.5/3.6 MoE | **256** | 8 routed + **1 shared** | **Yes** | **512** |

### Tokenizer / Vocab

| Generation | Vocab Size | Notes |
|------------|-----------|-------|
| Qwen3 | **151,669** | BBPE |
| Qwen3.5+ | **248,320** | Padded, handles 201 languages |

### Context Length

| Generation | Native | Extensible |
|------------|--------|------------|
| Qwen3 | 32K-128K | — |
| Qwen3.5/3.6 | **262K** | **1M** |

### Full Attention KV Heads

| Model | Q Heads | KV Heads | Ratio |
|-------|---------|----------|-------|
| Qwen3.5-9B | 16 | 4 | 4:1 |
| Qwen3.6-35B-A3B | 16 | **2** | **8:1** |
| Qwen3.6-27B | 24 | 4 | 6:1 |

### Linear Attention Heads

| Model | QK Heads | V Heads | Ratio |
|-------|---------|---------|-------|
| Qwen3.5-9B | 16 | 32 | 1:2 |
| Qwen3.6-35B-A3B | 16 | 32 | 1:2 |
| Qwen3.6-27B | 16 | **48** | **1:3** |

## Key Changes from Qwen3 to Qwen3.5+

1. **Hybrid attention** — biggest change. 3:1 linear-to-full ratio replaces pure GQA.
2. **Larger vocab** — 248320 vs 151669 (65% more embedding params)
3. **Longer context** — 262K vs 128K native
4. **Gated DeltaNet** — replaces GQA in 75% of layers, O(n) complexity
5. **Wider head dim** — 256 for full attention vs ~128 in Qwen3
6. **MoE improvements** — 256 experts, shared expert added back, tiny intermed (512)
7. **MTP** — multi-token prediction head (1 additional layer)
8. **MRoPE** — multi-resolution RoPE for 3D positional encoding
9. **Higher RoPE theta** — 10M gives better long-range positional resolution
10. **No bias** — attention_bias=false (no QKV bias, same as Qwen3)

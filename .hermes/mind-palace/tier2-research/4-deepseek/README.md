# DeepSeek Research — Key Papers for WuBu

Source papers: `.hermes/research/papers/research_papers/` (14 markdown files)

## Critical Papers (Must Read Before Implementation)

| Paper | ID | Why It Matters |
|-------|----|----------------|
| DeepSeek-V2 | 2405.04434 | **MLA attention origin** — latent compressed KV, decoupled RoPE |
| DeepSeek-V3 | 2412.19437 | **MLA + MoE** — aux-loss-free load balancing, multi-token prediction |
| DeepSeek-V3.2 | 2512.02556 | **DSA (Dynamic Sparse Attention)** + Block-Hadamard quantization — for 1M+ ctx |
| DeepSeek-R1 | 2501.12948 | RL for reasoning — the training methodology |
| DeepSeek-Prover-V2 | 2503.10893 | RL + Lean for theorem proving (references Lean integration) |

## DeepSeekMoE Architecture (from V3 paper)

- Fine-grained expert segmentation (many small experts)
- Shared expert (handles common knowledge, reduces routing overhead)
- Expert aggregation: `y = Σᵢ gᵢ × Eᵢ(x) + E_shared(x)` where gᵢ = softmax router
- Load balancing: **aux-loss-free** via dynamic bias adjustment (V3), or global-batch loss (V3.2)

## DeepSeek MLA (from V2 paper)

```
KV latent: c_t = W_DKV * [k_t; v_t]  (compression to d_c)
K_t = W_UK * c_t + RoPE(K_pe)         (decompressed + positional)
V_t = W_UV * c_t                       (decompressed)
Q_t = W_UQ * q_t + RoPE(Q_pe)         (query with decoupled RoPE)
attn = softmax(Q_t @ K_t^T / sqrt(d)) @ V_t
```

Key innovation: **decoupled RoPE** — separate small dimension for rotary, rest is non-positional.
This lets MLA use RoPE without breaking the KV cache compression.

## For WuBu

- **MLA** → alternative attention mechanism for hyperbolic compatibility
- **DSA** → sparse attention for 1M+ context with wubu nesting
- **MoE routing** → can be replaced with wubu hyperbolic distance routing
- **Multi-token prediction** → MTP head for faster training
- **RL training** → for reasoning capability after pretraining

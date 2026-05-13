# Qwen Research — Architecture for Embedding Grafting

Source: `.hermes/research/papers/Qwen/README.md` and all reference docs there.

## Key Papers

| Paper | ID / File | Why It Matters |
|-------|-----------|----------------|
| Qwen3 Tech Report | arxiv:2505.09388 | Baseline architecture (GQA + full softmax) |
| Qwen3.5 model card | HF config | **Gated DeltaNet** — 75% linear attention |
| Qwen3.6-35B-A3B config | HF config.json | **Our target** — 2048 hidden, 256 experts |
| Qwen3-Coder Next | GitHub PDF | Code-specialized MoE training |

## Architecture for Embedding Extraction

### Qwen3.6-35B-A3B (target model)
```
hidden_size = 2048
vocab_size = 248320
token_embedding = [248320, 2048]  →  ~508M params
lm_head = [248320, 2048]         →  ~508M params (NOT tied)
```

### How to Extract from GGUF
1. Use llama.cpp `convert.py` or direct GGUF parsing in C
2. Token embedding tensor: `token_embd.weight`
3. Output weight tensor: `output.weight`
4. Each attention/ffn layer: `blk.{n}.{attn/ffn}.weight`
5. RoPE frequencies: derived from theta=10M

### Gated DeltaNet Port to C
The linear recurrence:
```
h_t = λ_t ⊙ h_{t-1} + gate_t ⊙ (W_v x_t)     // standard DeltaNet
h_t = gyration(h_{t-1}, gate_t ⊙ (W_v x_t))   // WuBu hyperbolic version
```

This is the **key replacement** — the recurrence vector h_t lives in Poincaré ball,
and Möbius gyration replaces element-wise scaling.

### Full Attention Layers (25%)
- Keep as-is with hyperbolic Q,K,V projections
- RoPE is applied to 64/256 dims (partial_rotary_factor=0.25)
- MRoPE sections: [11, 11, 10] for 3D positions
- MRoPE is for vision tokens — for text-only we use standard 1D RoPE on 64 dims

### MoE Layer Port
- Router: replace linear router with hyperbolic distance to expert centroids
- 256 experts, but we can start with fewer (16?) for prototype
- Shared expert stays as-is (works as fallback)

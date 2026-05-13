# Embedding Grafting Research: WuBu × Qwen

## Goal
Extract embeddings from Qwen3.5/3.6 models, transform through WuBu hyperbolic geometry,
and graft back to create WuBuText AI without full pretraining.

## The Embedding Space (Qwen3.6-35B-A3B)

### Token Embedding
```
Shape: [248320, 2048]  (vocab_size × hidden_size)
Type: float32/bfloat16
Params: ~508M
```

Token embeddings map 248,320 token IDs to 2048-dim vectors. This is the **input embedding**,
used as the first layer before the transformer stack.

### LM Head (Output Embedding)
```
Shape: [248320, 2048]
Type: float32/bfloat16  
Params: ~508M (NOT tied — tie_word_embeddings=false)
```

Separate output projection. Total embedding params = ~1B out of 3B active.

### RoPE Positions
- theta = 10,000,000
- partial_rotary_factor = 0.25 → 64 of 256 dims are rotated
- MRoPE: (11, 11, 10) = 32 total rotational dims for 3D positions

### Gated DeltaNet (Linear Attention)
- Input projected to 16 QK heads × 128 dim = 2048
- Input projected to 32 V heads × 128 dim = 4096
- 4-dim convolution kernel for local mixing
- Output gated: `y = sigmoid(W_g x) ⊙ o`

### Full Attention (GQA)
- 16 Q heads × 256 = 4096
- 2 KV heads × 256 = 512
- 64 of 256 dims rotated by RoPE (partial_rotary_factor=0.25)

## WuBu Hyperbolic Transformation

### What We Replace
1. **Input embedding** — keep Qwen's pretrained token embeddings, apply hyperbolic mapping
2. **LM head** — keep Qwen's output projection, apply hyperbolic-to-Euclidean projection
3. **Attention layers** — graft hyperbolic gyration into Gated DeltaNet's linear recurrence
4. **MoE** — replace router with wubu nested geometry routing

### The Grafting Strategy

```
Qwen Embedding → [WuBu Hyperbolic Map] → WuBu Gyration Layers → [Euclidean Projection] → LM Head
                     ↓                             ↓
              Token embeddings              Attention replaced with
              transformed to               hyperbolic linear attention
              Poincaré ball                (gyration replaces dot product)
```

### Phase 1: Embedding Only (Quick Win)
1. Load Qwen3.6-35B-A3B token embeddings (248320 × 2048)
2. Map to Poincaré ball: `x' = tanh(||x||/R) × x/||x||`
3. Verify with nearest-neighbor: are hyperbolic embeddings more semantically coherent?
4. Export as WuBu embedding layer

### Phase 2: Gated DeltaNet → Hyperbolic Linear Attention
1. Replace linear recurrence `h_t = λh_{t-1} + gate ⊙ v` with hyperbolic version:
   - `h_t = gyration(h_{t-1}, gate ⊙ v)` using Möbius addition
2. Key insight: Gated DeltaNet's linear recurrence is **naturally compatible** with hyperbolic gyration
   because both are element-wise operations on the state vector

### Phase 3: MoE Routing with WuBu Nested Geometry
1. Replace router with wubu nested clusters
2. Each expert assigned to a hyperbolic region
3. Router = distance in Poincaré ball to expert centroids

## Why Qwen3.5+ Architecture is Ideal for WuBu

| Qwen Feature | WuBu Advantage |
|-------------|---------------|
| **Gated DeltaNet** | Linear recurrence maps directly to Möbius gyration |
| **75% linear attention** | Only 25% of layers need full softmax replacement |
| **Small hidden (2048)** | Fits in 6.4GB VRAM with wubu math overhead |
| **Large vocab (248K)** | Rich embedding space to transform |
| **MoE with shared expert** | Can route through wubu geometry, shared expert = fallback |
| **Vision encoder** | Foundation for WuBuVision |
| **262K context** | Linear attention + wubu = potential for 1M+ |

## Key Questions to Answer

1. **What's the best R (Poincaré ball radius) for Qwen's embedding distribution?**
   - Compute mean norm of token embeddings → set R = 3× mean
2. **Does hyperbolic transformation preserve nearest-neighbor relationships?**
   - Compare top-10 nearest tokens in Euclidean vs Poincaré for sample tokens
3. **Can we train only the gyration parameters while freezing Qwen embeddings?**
   - Gated DeltaNet's `λ` (recurrence gate) and output gate are per-layer learnable scalars
   - Convert to wubu gyration: only ~40 × 2 = 80 extra params
4. **What's the perplexity overhead of Euclidean → hyperbolic embedding mapping?**
   - Measure on small validation set before/after transformation

## Reference Models

| Model | Hidden | Embed Dims | Active Params | VRAM (f16) |
|-------|--------|-----------|---------------|------------|
| Qwen3.6-35B-A3B | 2048 | 248320 × 2048 | 3B | ~6GB |
| Qwen3.5-9B | 4096 | 248320 × 4096 | 9B | ~18GB |
| Qwen3.6-27B | 5120 | 248320 × 5120 | 27B | ~54GB |

**Qwen3.6-35B-A3B** is the sweet spot for 6.4GB VRAM (our RTX 5050).

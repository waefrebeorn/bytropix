# Current C Baseline — What Exists

## Location
Source: `/home/wubu/bytropix/include/` (5 headers) + `/home/wubu/bytropix/src/` (7 C files)
Training: `src/train.c` — runs 6000+ steps, loss 4.68→3.12

## Components

| Module | File | Status |
|--------|------|--------|
| **Tokenizer** | `include/tokenizer.h`, `src/tokenizer.c` | 97-token ASCII, working |
| **Rolling Hash** | `include/rolling_hash.h`, `src/rolling_hash.c` | Rabin-Karp, working |
| **NN Ops** | `include/nn_ops.h`, `src/nn_ops.c` | Forward/backward transformer |
| **WuBu Model** | `include/hashmind_model.h`, `src/hashmind_model.c` | Full transformer model |
| **Data Loader** | `include/hashmind_data.h`, `src/hashmind_data.c` | Data pipeline |
| **Training Loop** | `src/train.c` | ~30 tok/s CPU, 6000+ steps |
| **WuBu Math** | `src/wubu_math.c`, `src/wubu_math.h` | Hyperbolic ops |
| **Optimizer** | in train.c | Toroidal gradient mod 2π, clamped at 0.1 |

## Current Architecture

```
Tokenizer → Embedding (token_hash) → Transformer → LM Head → Softmax → Loss
                                            ↓
                                    768-dim hidden
                                    6 layers
                                    GQA attention (4 Q, 2 KV)
                                    SwiGLU FFN
                                    WuBuOptimizer
```

## What's Missing vs Target (Qwen3.6-35B-A3B)

| Feature | Current C | Target | Gap | Status |
|---------|-----------|--------|-----|--------|
| Hidden dim | 768 | 2048 | ×2.7 bigger | Phase 2 |
| Layers | 6 | 40 | ×6.7 more | Phase 2 |
| Attn type | GQA only | SSM + GQA hybrid | New attention type | Phase 2 (blocked: SSM impl unknown) |
| MoE | None | 256/8+1 (IQ2_XS/IQ1_S) | Entirely missing | Phase 4 |
| Vocab | 97 | 248320 (BBPE) | ×2560 bigger | Phase 3 (blocked: no BBPE tokenizer) |
| Context | 512 | 262K | ×512 longer | Phase 3 |
| GPU | No | CUDA required | Entirely missing | Phase 3 |
| Embeddings | Tiny hash | Q5_K (2.03GB, 248320×2048) | Need loading code | Phase 1 ✅ |
| MTP | None | 1 head | Missing | Phase 3 |
| Training speed | 30 tok/s CPU | Need GPU target | Unknown | Phase 3 |
| Optimizer | Toroidal (g%2π) | RSGD for Poincaré | Wrong optimizer | Phase 3 |

## Key Takeaways (Updated)

1. The current C code is a **prototype** — it proves the pipeline works but uses different\
   math (toroidal vs Poincaré)
3. The **WuBu math that carries over** is: exp_map, log_map, Möbius addition, Poincaré distance
4. Phase 1 ✅ proved embeddings can be extracted and mapped (95% NN preservation)
5. **Phase 2 BLOCKED** by: unknown SSM implementation details in attn_qkv weight split
6. **Phase 3 BLOCKED** by: no BBPE tokenizer — can't use the extracted embeddings without one
7. The old training loop (30 tok/s CPU, toroidal optimizer) should NOT be reused — write fresh

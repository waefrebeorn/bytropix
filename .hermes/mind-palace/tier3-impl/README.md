# Tier 3: Implementation

Build order. Each phase depends on the previous one.

## Contents

| # | Directory | What | Depends On |
|---|-----------|------|------------|
| 8 | `8-embedding-graft/` | Extract Qwen3.6 embeddings, map to Poincaré, test grafting | Tier 2 research |
| 9 | `9-attention-port/` | Port Gated DeltaNet to C + hyperbolic gyration | Tier 1 C baseline |
| 10 | `10-training-loop/` | Training loop for the full model | 8, 9 |
| 11 | `11-moe-port/` | MoE routing with wubu nested geometry | 9 |
| 12 | `12-vision/` | Vision encoder for WuBuVision | 10 |

## Strategy

**Skip pretraining** via Euclidean→hyperbolic weight translation from Qwen3.6.
Train only the hyperbolic adaptation layers (gyration params, MoE router).
This saves ~99% of compute vs training from scratch.

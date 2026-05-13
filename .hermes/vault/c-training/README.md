# Vault: C Training — Pure C Port of HashMind
#
## Location: `ENCODERS/hash-mind/c/`
## Files:
```
src/
  tokenizer.c         — 97-token ASCII tokenizer (all printable + \n\t)
  rolling_hash.c      — Rabin-Karp sliding window hash
  hashmind_model.c    — Full transformer: forward + backward pass (manual backprop)
  hashmind_data.c     — Data loader, eval, text generation
  train.c             — Training loop entry point
include/
  tokenizer.h         — Tokenizer API
  rolling_hash.h      — Rolling hash API
  nn_ops.h            — Layer norm, GELU, softmax, cross-entropy, matmul
  hashmind_model.h    — Model + gradient + momentum structs, API
  hashmind_data.h     — TextData, TrainExample, eval, generation
ggml-cuda/
  wubu-cuda.cu        — CUDA kernels: Poincaré exp map, MoE routing
  wubu-cuda.cuh       — CUDA kernel declarations
```

## Key Numbers
- **210,112 parameters** (4-layer transformer, d_model=64, 4 heads, 97 vocab)
- **~4000 steps/sec** training on 50K chars (CPU, -O3, no CUDA)
- **Gradient clipping at 0.5** needed for stability
- **lr=0.0003** works (lr=0.001 blows up at step 1300 — gradient accumulation bug)

## Build
```
make          # build train
make test     # build + quick training test (5 epochs)
./train --lr 0.0003 --epochs 10 --gen-every 5000 --data training_sample.txt
```

## Architecture
- Dual-source embedding: char_embed[D_MODEL] + hash_val * hash_projector[D_MODEL]
- Sinusoidal positional encoding
- 4 transformer blocks: LayerNorm → QKV attention (4 heads, 16-dim) → Out proj → Residual → LayerNorm → FFN(64→256→64, ReLU) → Residual
py loss
    41|- Optimizer: SGD with Nesterov momentum (mu=0.9) + weight decay (wd=1e-4)
    42|- Forward + full manual backward pass (each layer: out_w → FFN → ln2 → attn → qkv → ln1 → embed)
    43|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*

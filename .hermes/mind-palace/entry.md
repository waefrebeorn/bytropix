# WuBuText AI — Entry Point (May 15 PM v6)

## Purpose
Build commands, hardware spec, quick-start.

---

## Hardware
- **GPU:** NVIDIA RTX 5050, 6.4GB VRAM, sm=120
- **NVCC:** /usr/local/cuda-13.1/bin/nvcc -arch=sm_120
- **CPU:** AMD 16+ cores, 25GB RAM

## Build
```bash
make train_integrated      # Primary training binary (11s/step)
make bench_e2e             # Full 40-layer GPU vs CPU benchmark
make test_gpu              # GPU forward pass test
make infer_poincare        # Poincaré SSM inference (2835 tok/s)
make infer_moe_lazy        # Lazy MoE dequant
make infer_unified         # 40-layer SSM→GQA→MoE
make test_kv_cache         # GQA KV cache test (256K ctx)
make infer_vision_gpu      # GPU vision 128×128 in 99ms
make train_real            # CPU training pipeline (reference)
```

## Run Training
```bash
# Default (no flags)
./train_integrated /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin 10

# With hyperbolic flags
TST=1 RSGD=1 NESTED_SSM=1 NESTED_MOE=1 POINCARE_R=0.956 ./train_integrated ...
```

## File Layout
```
src/          — Core: ssm, moe, model, gguf_reader, cuda_kernels, vision
include/      — Headers
tools/        — train_integrated, train_gpu, infer_*, test_*
data/         — embeddings, tokenizer, training data
.hermes/       — Mind palace, vault, research, references
DIAGRAMS/      — 7 SVG architecture diagrams
THEORY/        — Papers, math, vault references
/models/       — GGUF (Qwen3.6-35B-A3B-UD-IQ2_M.gguf)
```

## External Repos (for reference)
- `~/HASHMIND/tailslayer/` — Hedged-read C++ library (spec-decode inspiration, P2)
- `~/HASHMIND/llama-cpp-rotorquant/` — llama.cpp fork with Hamilton encoder CUDA kernels
- `~/HASHMIND/HAS/` — WuBu research (JAX prototypes, vault originals)

## Key Docs
- `.hermes/mind-palace/goal-mantra.md` — Prestige paste, full state
- `.hermes/mind-palace/plan.md` — Priority queue + vault + tailslayer
- `.hermes/vault/tailslayer/` — Tailslayer findings (May 15)
- `README.md` — Full project overview

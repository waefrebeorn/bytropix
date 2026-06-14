<div align="center">

# ⚡ bytropix — Multi-Model C/CUDA Inference Engine

**Pure C/CUDA inference for DiffusionGemma-26B, Gemma 4 12B QAT, Qwen3.6-35B**
**🔁 Multi-model adapter: one codebase, three architectures (2026-06-14)**

~35,000 lines C/CUDA | 550+ commits | 20 SVG diagrams | 68 Python analysis tools

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)
[![GPU: RTX 5050](https://img.shields.io/badge/GPU-RTX_5050_8GB-critical)](https://github.com/waefrebeorn/bytropix)
[![KV Cache: Q4_0 4:1](https://img.shields.io/badge/KV_Cache-Q4_0_4:1-green)](https://github.com/waefrebeorn/bytropix)
[![Arch: multi-model](https://img.shields.io/badge/Arch-Multi--Model-orange)](https://github.com/waefrebeorn/bytropix)

</div>

---

## What is bytropix?

bytropix is a **from-scratch C/CUDA inference engine** supporting **three architectures** through a unified model adapter:

| Model | Architecture | Context | VRAM |
|-------|-------------|---------|------|
| **DiffusionGemma-26B-A4B-it** | MoE GQA (30 layers, top-8/128 experts) | 512K | ~18GB Q4_K_M |
| **Gemma 4 12B QAT** | ISWA Dense Transformer (48 layers) | 128K | ~7.3GB Q4_K_XL |
| **Qwen3.6-35B-A3B** | Gated DeltaNet SSM + GQA + MoE | 256K | ~8GB IQ2_M |

Targets **RTX 5050 8GB** with Q4_0 KV cache (4:1 compression). Runs entirely on local hardware (WSL2) with no framework dependencies.

---

## Model Adapter Architecture

The `wubu_model.c` multi-model adapter automatically detects and configures for each architecture:

```
Tensor Naming Detection → Dimension Extraction → Layer Config → KV Cache Layout
        ↓                         ↓                    ↓              ↓
  blk.%d.* (Qwen)          d_model from           is_ssm per    Dynamic KV
  model.layers.%d.*        tensor shapes          layer (arch-   offsets per
  (Gemma/DGemma)           head_dim, n_heads,     aware)        variable kv_dim
                           n_experts, d_ff
```

**Naming conventions:**
- **Qwen**: `blk.{i}.attn_q.weight`, `blk.{i}.ssm_*`, `blk.{i}.gate.weight`
- **Gemma/DiffusionGemma**: `model.layers.{i}.self_attn.q_proj.weight`, etc.

**Architecture detection:**
- **Qwen**: Mixed SSM+GQA (3:1 interleaved), 256-MoE experts
- **Gemma 4**: All GQA ISWA (sliding:full 6:1), dense FFN
- **DiffusionGemma**: All GQA (30 layers), 128-MoE experts (top-8), heterogeneous head_dim

---

## Quick Start

```bash
# Build
make gen_text_cpu          # CPU inference binary
make gen_text_gpu          # GPU inference binary
make bench_512k_full       # 512K context benchmark

# Run (any of the 3 models)
./bench_512k_full /home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf 4096 1 0
./bench_512k_full /home/wubu/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf 4096 1 0
./bench_512k_full /home/wubu/models/Qwen3.6-35B-A3B-IQ2_M.gguf 4096 1 0

# Environment variables
MAX_CTX=262144 OMP_NUM_THREADS=16 ./gen_text_cpu "prompt" 20 100
```

**Hardware:** AMD Ryzen 7950X (16C/32T) | 64 GB DDR5 | RTX 5050 8GB VRAM | WSL2

---

## Model Specs

### DiffusionGemma-26B-A4B-it (Q4_K_M)

```
30 Layers GQA MoE (top-8 of 128 experts)
├── Normal layers (20):  hidden=2816, Q=4096 (16×256), KV=2048 (8×256), FFN=704
├── LARGE layers  (10):  hidden=2816, Q=8192 (16×512), KV=1048 (2×512),  FFN=704
├── Vocab:                248,320 tokens
├── d_model:              2816 (extracted from GGUF, not hardcoded)
└── Sliding window:       1024
```

### Gemma 4 12B QAT (Q4_K_XL)

```
48 Layers ISWA (6:1 sliding:full)
├── Sliding (40):  hidden=3840, Q=4096 (16×256), KV=2048 (8×256), FFN=15360
├── Full (8):      hidden=3840, KV=512 (1×512 global), θ=1M prop, 25% rotary
├── Vocab:         262,144 tokens (tied embeddings)
└── RoPE:           dual (θ=10K default + θ=1M proportional)
```

### Qwen3.6-35B-A3B (IQ2_M, legacy)

```
40 Layers (30 SSM + 10 GQA, 3:1 interleaved)
├── SSM (30):  D=2048, Δ-dim=64, 1024 expert_dim hidden, 256 MoE experts
├── GQA (10):  layers 3,7,11,15,19,23,27,31,35,39 — Q=4096 (16×256), KV=512 (2×256)
└── Vocab:     151,936 tokens
```

---

## Features

### ✅ Verified Working

- **Multi-model adapter** — `wubu_model.c` auto-detects Qwen/Gemma/DiffusionGemma naming
- **Dynamic dimensions** — `d_model`, `head_dim`, `kv_dim`, `n_heads` extracted from GGUF tensor shapes
- **Dynamic KV cache** — variable `kv_dim` per layer (DiffusionGemma LARGE layers need 2× KV)
- **CPU inference** — Qwen3.6-35B sequential SSM, coherent output
- **Q4_0 KV cache** — 4:1 compression, cos-sim 0.9994 vs FP16 at 256K
- **GQA forward** — `d_model` parameter (GQA functions accept any model's hidden dim)
- **Lazy MoE expert loading** — blob-pointer on-demand, saves ~3GB RAM
- **MTP speculative decode** — Qwen draft model integrated

### 🔄 In Progress

- **DiffusionGemma forward pass** — model loads (30 GQA layers), crashes during decode (per-layer head_dim mismatch for LARGE layers)
- **Gemma 4 GPU inference** — ISWA kernels written, cuBLAS integration pending
- **GPU SSM pipeline** — recurrence, conv1d, SiLU, gated norm on GPU

### ❌ Blocked / Not Started

- **DiffusionGemma LARGE layer dims** — head_dim=512 but GQA loading uses fixed 256
- **Full GPU forward for any model** — all three need end-to-end GPU path

---

## Project Map

```
bytropix/
├── src/              # Core C/CUDA — SSM, MoE, GQA, GPU kernels, model adapter
│   ├── wubu_model.c      # Multi-model adapter (AUTO-detect + dynamic dims)
│   ├── wubu_ssm.c        # SSM forward/backward + GQA forward (d_model param)
│   ├── wubu_moe.c        # MoE forward (shared across models)
│   └── ...
├── include/          # Headers — model structs, GGUF reader, GPU kernels
│   ├── wubu_model.h      # wubu_model_t (added: d_model, tensor_naming, n_gqa_layers)
│   ├── wubu_ssm.h        # GQA functions (added: d_model parameter)
│   └── ...
├── tools/            # Benchmark binaries + Python analysis scripts
│   ├── bench_512k_full.c # 512K context benchmark (tests all 3 models)
│   └── ...
├── .hermes/          # Mind palace, plans, session state
│   └── mind-palace/  # Architecture deep-dives, model specs
├── llama/            # Upstream llama.cpp (reference, Qwen3.6 + Gemma4 support)
└── GEMMA4.md         # Gemma 4 12B engine-specific docs
```

---

## Key Code Changes (Multi-Model Adapter)

### `include/wubu_model.h`
```c
// Added to wubu_model_t struct:
int d_model;           // Hidden dim (2048 Qwen, 2816 DiffusionGemma, 3840 Gemma4)
int head_dim;          // Per-head dim (256 or 512 for DGemma LARGE)
int n_experts;         // MoE experts (256 Qwen, 128 DiffusionGemma, 0 Gemma4)
int n_active_experts;  // Top-K (12 Qwen, 8 DiffusionGemma)
int tensor_naming;     // 0=Qwen, 1=Gemma, 2=pure-GQA
int n_gqa_layers;      // Actual GQA layer count (not hardcoded)
```

### `include/wubu_ssm.h`
```c
// GQA functions now accept d_model (no more D_MODEL macro dependency):
void wubu_gqa_forward(float *output, int B, int T, int C, const gqa_layer_weights *w,
                      float *kv_cache, int kv_dim, int head_dim, int ctx_cur,
                      float *piece, float norm_eps, int d_model);
```

### `src/wubu_model.c`
- `g_tensor_naming` global variable — set during init, read by `wubu_is_ssm_layer()`
- Dimension extraction from GGUF tensor shapes
- KV cache dynamically sized: `GQA_MAX_CTX × Σ(layer_kv_dim)` instead of fixed
- Per-layer KV offsets computed for variable kv_dim

---

## Build

```bash
make gen_text          # CPU inference
make gen_text_gpu      # GPU inference
make bench_512k_full   # 512K benchmark (all 3 models)
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GPU=1` | Enable GPU inference path |
| `FORCE_CPU_SSM_SEQ=1` | Force SSM on CPU (for debugging) |
| `MAX_CTX=<N>` | Context length (default: 4096, max: 262144) |
| `OMP_NUM_THREADS=<N>` | OpenMP thread count |

---

## Current Benchmark Status

| Model | Load | Forward | Notes |
|-------|------|---------|-------|
| Qwen3.6-35B | ✅ | ✅ | 3-4 tok/s CPU, 5.5 tok/s hybrid GPU |
| Gemma 4 12B QAT | 🔄 | ⏳ | Dedicated Gemma4 engine in `wubu_gemma4.c` |
| DiffusionGemma-26B | ✅ | ❌ | Model loads (30 GQA), crashes in decode |

---

## License

MIT License — see [.license](.license).

---

<div align="center">

*Engine: bytropix — from-scratch C/CUDA inference. Multi-model: DiffusionGemma-26B + Gemma 4 12B QAT + Qwen3.6-35B.*

*[MADE_AGENTICALLY_BY_HERMES.md](MADE_AGENTICALLY_BY_HERMES.md) — full engineering log, bug history, and DA verification methodology.*

</div>

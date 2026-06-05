<div align="center">

# ⚡ bytropix — Multi-Modal Inference Engine

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE) + Moondream3 Vision**

~30,000 lines C/CUDA | 512 commits | 20 SVG diagrams | 68 Python analysis tools

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)
[![GPU: RTX 5050](https://img.shields.io/badge/GPU-RTX_5050_8GB-critical)](https://github.com/waefrebeorn/bytropix)
[![KV Cache: Q4_0 4:1](https://img.shields.io/badge/KV_Cache-Q4_0_4:1-green)](https://github.com/waefrebeorn/bytropix)
[![Arch: qwen35moe](https://img.shields.io/badge/Arch-qwen35moe-purple)](https://github.com/waefrebeorn/bytropix)
[![Vision: 3D ViT](https://img.shields.io/badge/Vision-Moondream3_3D_ViT-success)](https://github.com/waefrebeorn/bytropix)

</div>

---

## What is bytropix?

bytropix is a **from-scratch C/CUDA inference engine** for Qwen3.6-35B-A3B — a 35B-parameter Gated DeltaNet + Mixture-of-Experts model with 256 experts, and Moondream3's 3D Vision Transformer. It runs entirely on local hardware (RTX 5050 8GB, 64GB RAM) with no framework dependencies.

The project includes GPU kernels (SSM recurrence, quantized matmul, MoE, attention), a Q4_0 KV cache (~4:1 compression), vision encoder pipeline, and research components spanning hyperbolic geometry, nesting theory, sparse attention, and more.

---

## Quick Start

```bash
# CPU text inference
make gen_text
./gen_text "The capital of France is" 20 40

# GPU inference (requires CUDA)
make gen_text_gpu
GPU=1 MAX_CTX=4096 ./gen_text_gpu "Explain quantum computing" 50 100

# Vision encoder test
make test_vision_real
./test_vision_real /path/to/mmproj-F16.gguf /tmp/vision_input.bin

# API server (OpenAI-compatible)
make api_server
./api_server --sandbox --port 8080
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

**Hardware:** AMD Ryzen 7950X (16C/32T) | 64 GB DDR5 | RTX 5050 8GB VRAM | WSL2

---

## Architecture

### Model Spec (Qwen3.6-35B-A3B / `qwen35moe`)

```
40 Layers: 10 cycles × (3×SSM → 1×GQA)
├── SSM layers:  30 layers (Gated DeltaNet recurrence)
├── GQA layers:  10 layers (Grouped Query Attention)
├── Hidden dim:  2048
├── Vocab:       248,320 tokens
├── SSM:         16 K-heads × 128, 32 V-heads × 128
├── GQA:         16 Q-heads × 256, 2 KV-heads × 256
├── MoE:         256 experts, 8 active + 1 shared
├── Expert FFN:  512 | Shared FFN: 512
├── RoPE:        IMRoPE, sections=[11,11,10,0], θ=10M
└── Quant:       Mixed IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K (~2.7 bpw)
```

### Vision Encoder (Moondream3 / 3D ViT)

```
27-layer Vision Transformer
├── 3D patch embedding: spatial 16×16, temporal 2 frames
├── Spatial merge: 2×2 grid (4:1 compression)
├── Hidden dim: 1152 → 16 heads × 72 head_dim
├── GQA attention (16 heads) + GELU activation
├── Post-LN + Merger MLP: 4608 → 2048 (text hidden dim)
└── Output: image tokens in text embedding space
```

### Multi-Modal Pipeline

```
Image → Patch Embed → 27×ViT → Spatial Merge → MMProj → Text tokens → 40×Text Model → Output
                                                           ↑
                                         Qwen3.6 Text Embedding ← GGUF token embeddings
```

### VRAM Budget (256K Context, Text Only)

| Component | Size | Format |
|-----------|------|--------|
| SSM weights (quantized) | ~2,266 MB | Mixed quant on GPU |
| GQA weights | 1,040 MB | F32 (cuBLAS SGEMM) |
| KV cache (Q4_0) | 1,440 MB | 4-bit, 4:1 vs FP16 |
| KV cache (FP16) | 5,120 MB | FP16 via `GPU_Q4_0_KV=0` |
| Output projection | 1,900 MB | Q4_K quantized |
| MoE + scratch | ~460 MB | IQ2_XXS + temp buffers |
| **Total (Q4_0 KV)** | **~5,100 MB** | Fits 8GB GPU with ~2.9GB headroom |

---

## Features

### Verified (Runtime-Crosschecked)

- **CPU inference** — full 40-layer, verifiable against llama.cpp
- **GPU GQA attention** — cuBLAS-backed on GPU
- **GPU output projection** — Q4_K custom kernel
- **Q4_0 KV cache** — 4:1 compression, cos-sim 0.9994 vs F16 at 256K
- **Lazy MoE expert loading** — blob-pointer on-demand, saves ~3GB RAM
- **Sliding window attention** — GQA_WINDOW env var for 256K context
- **Tokenizer** — GPT-2 BPE with 248K vocab, merge-table-based

### In Progress

- **GPU SSM pipeline** — SSM recurrence, conv1d, SiLU, gated norm on GPU
- **NSA sparse attention** — DeepSeek-V3.2 DSA pattern for O(L log L) GQA
- **Chunked SSM prefill** — batched recurrence for faster prefill
- **IQ1_M + Q4_K quant matmul** — GPU kernels for extreme quantization
- **MTP speculative decode** — multi-token prediction draft model

### Research Components (Not Wired Into Inference)

See the research directories: `THEORY/`, `MATH/`, `ENCODERS/`, `AUDIO/`, `DIFFUSION/`, `ATTENTION/`, `OPTIMIZERS/`, `WUBUNEST_V2/`.

| Concept | Code | Status |
|---------|------|--------|
| Poincaré SSM | ~500 lines test | Standalone, not wired |
| Nested SSM (K-ball recursion) | None | Research paper |
| Hamilton Encoder | ~1,000 lines | Standalone concept |
| WuBu Nesting Theory | Papers | Theoretical framework |
| RotorQuant / TurboQuant | References | External refs |
| NV64 Ring Buffer | Design doc | Not implemented |

---

## Project Map

```
bytropix/
├── src/              # Core C/CUDA — SSM, MoE, GQA, vision, GPU kernels
├── include/          # 26 headers — model structs, GGUF reader, GPU kernels
├── tools/            # ~50 C binaries + 68 Python analysis scripts + API server
├── tests/            # Regression test harness
├── DIAGRAMS/         # 20 SVG architecture diagrams
├── data/             # Tokenizer data, vision configs, embeddings
├── vault/            # Unsloth quant format, cache compression refs
├── .hermes/          # Mind palace, vault, plans, session state
│
├── THEORY/           # WuBu Nesting paper, spatio-temporal findings, LaTeX
├── MATH/             # Lean formal proofs
├── ENCODERS/         # Hamilton encoder, HashMind, topological AE
├── AUDIO/            # Audio compressor, WuBuSynth
├── DIFFUSION/        # Funnel diffusion, GAN-VAE hybrid, HGA U-Net
├── ATTENTION/        # Entropix sampler, hyperbolic attention, sparse attention
├── OPTIMIZERS/       # PID controller, Q-controller
├── WUBUNEST_V2/      # Training scripts (numpy + GPU)
├── DRAFT/            # Prototypes and legacy scripts
└── python/           # Tokenizer extraction utilities
```

---

## Verification Tools

| Tool | Purpose |
|------|---------|
| `layer_cos_sim` | Per-layer cosine similarity vs llama.cpp |
| `ref_dumper` | Single-token llama.cpp embedding dumper (via libllama.so) |
| `compare_ggml_matmul` | Quantized matmul vs ggml SGEMM |
| `test_vision_real` | Vision encoder end-to-end with real image |
| `DUMP_LAYER_DIR` | Save per-layer hidden states to `.bin` |
| `DUMP_INTERMEDIATE_DIR` | Save ALL intermediate tensors (53 types/layer) |
| `DUMP_GQA_DEBUG_DIR` | Per-layer GQA attention debug dumps |
| `DUMP_EMBEDDING_DIR` | Token embedding debug |
| `PROFILE` | Per-layer timing breakdown |
| **Python tools/** | 68 analysis scripts for embedding, layout, dequant verification |

---

## Build

```bash
# CPU inference
make gen_text

# GPU inference
make gen_text_gpu

# MTP speculative decode
make gen_text_mtp

# API server
make api_server

# Reference comparison tools
make ref_dumper layer_cos_sim

# All targets
make all
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GPU=1` | Enable GPU inference path |
| `FORCE_CPU_SSM=1` | Force SSM on CPU (for debugging) |
| `MAX_CTX=<N>` | Context length (default: 4096, max: 262144) |
| `GQA_WINDOW=<N>` | Sliding window size for GQA |
| `GPU_Q4_0_KV=0` | Disable Q4_0 KV cache, use F16 |
| `ROPE_SCALE_FACTOR=<f>` | RoPE length extrapolation (4x) |
| `OMP_NUM_THREADS=<N>` | OpenMP thread count |
| `DUMP_LAYER_DIR=<path>` | Debug: dump hidden states |
| `DUMP_INTERMEDIATE_DIR=<path>` | Debug: dump all intermediates |
| `PROFILE=<path>` | Debug: per-layer timing |

---

## License

MIT License — see [.license](.license).

---

<div align="center">

*Engine: bytropix — from-scratch C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE) + Moondream3 Vision.*

*[MADE_AGENTICALLY_BY_HERMES.md](MADE_AGENTICALLY_BY_HERMES.md) — full engineering log, bug history, and DA verification methodology.*

</div>

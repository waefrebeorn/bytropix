<div align="center">

# ⚡ bytropix — WuBu Text AI Inference Engine

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch)**

[![Phase: 28b](https://img.shields.io/badge/Phase-28b--DA-blueviolet)](https://github.com/waefrebeorn/bytropix)
[![GPU Decode: 8.5 tok/s](https://img.shields.io/badge/GPU%20Decode-8.5%20tok%2Fs-informational)](https://github.com/waefrebeorn/bytropix)
[![KV Cache: Q4_0 4:1](https://img.shields.io/badge/KV%20Cache-Q4%5F0%204%3A1-green)](https://github.com/waefrebeorn/bytropix)
[![256k VRAM: 3.56 GB](https://img.shields.io/badge/256k%20VRAM-3.56%20GB-success)](https://github.com/waefrebeorn/bytropix)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![GPU: RTX 5050](https://img.shields.io/badge/GPU-RTX%205050%208GB-critical)](https://github.com/waefrebeorn/bytropix)

</div>

---

## 📊 Current State

<div align="center">

|| Status | Metric | Detail |
||:------:|--------|--------|
|| ✅ | **GPU decode (4K ctx)** | **7.6-8.5 tok/s** — gen_text_gpu, SSM ON CPU (GPU_SUPPORT was dead code) |
|| ✅ | **GPU decode (256k)** | **4.8 tok/s** — full attention, 5.7 tok/s sliding window 16K |
|| ✅ | **GPU_SUPPORT now builds** | Phase 28: fixed 3 pre-existing bugs (braces, gpu_ctx_t, -DGPU_SUPPORT) |
|| ❓ | **SSM GPU path correctness** | **NEVER VERIFIED** — cos-sim vs CPU path = 0. SSM GPU runs but output quality unknown |
|| 🔴 | **F32 waste** | ~2.2 GB VRAM wasted on F32 dequant weights never used in inference |
|| 🔴 | **GPU mem leak** | 5.5 GB of quantized + F32 SSM weights never freed in wubu_model_gpu_free() |
|| 🔴 | **Prefill N>1 broken** | Fallback path uses column-major quant_matmul (wrong stride) |
|| 🟡 | **Fused SSM beta/alpha decode** | Verified in isolation (cos-sim 1.0) but never tested in full pipeline |
|| 🟡 | **L31 cos-sim** | 0.9585 — measured with OLD code (GPU_SUPPORT dead). Need re-measure |
|| 🟡 | **External ref speed** | llama.cpp gets **35.4 tok/s** on RTX 4060 Ti. Gap unknown after GPU_SUPPORT fix |

</div>

---

## 🚀 Quick Start

```bash
# Build GPU inference
make gen_text_gpu

# Run (GPU=1 required for GPU init)
GPU=1 MAX_CTX=4096 ./gen_text_gpu -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -t 0.6 -n 32

# CPU inference
make gen_text
./gen_text -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -t 0.6 -n 32

# Reference comparison with llama.cpp
make ref_dumper
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Your prompt" 0
DUMP_LAYER_DIR=/tmp/our ./gen_text "Your prompt" 0
tools/layer_cos_sim /tmp/ref /tmp/our 40
```

**Hardware:** AMD Ryzen 7950X (16C/32T) | 64 GB DDR5 | RTX 5050 8GB VRAM | WSL2
**Model:** `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (733 tensors, 10.7 GB)

---

## 🏗️ Architecture

### Model Spec (Qwen3.6-35B-A3B / `qwen35moe` GGUF)

```
40 Layers: 10 cycles × (3×SSM → 1×GQA)  ← 3:1 INTERLEAVED, not contiguous!
├── SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
├── GQA layers: 3,7,11,15,19,23,27,31,35,39
├── Hidden dim:    2048
├── Vocab:         248,320
├── SSM:           16 K-heads × 128, 32 V-heads × 128
├── GQA:           16 Q-heads × 256, 2 KV-heads × 256
├── MoE:           256 experts, 8 active + 1 shared
├── Expert FFN:    512
├── Shared FFN:    512
├── RoPE:          IMRoPE, sections=[11,11,10,0], θ=10M
└── Quant:         Mixed IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K (~2.7 bpw)
```

### Per-Layer Flow

**SSM layer** (3 out of every 4):
```
x → RMSNorm → attn_qkv(Q5_K) → gate(Q5_K) → conv1d → SiLU → split → L2 norm →
SSM recurrence(16 heads) → gated norm → ssm_out(Q6_K) → gate(SiLU) × → MoE → +residual
```

**GQA layer** (every 4th: indices 3,7,11,15,19,23,27,31,35,39):
```
x → RMSNorm → attn_q(Q5_K) → gate(Q5_K) → attn_k(Q5_K) → attn_v(Q5_K) →
IMRoPE → full attention(KV cache: Q4_0 or F16) → output(Q5_K) → sigmoid(gate) × → MoE → +residual
```

### VRAM Budget (256k Context)

| Component | Size | Format |
|-----------|------|--------|
| GQA weights | 1,040 MB | F32 (cuBLAS SGEMM) |
| SSM weights (quantized) | 692 MB | Q5_K/Q6_K on GPU |
| KV cache (GPU: Q4_0) | **1,440 MB** | **4-bit, 4:1 vs FP16 — Phase 23** |
| KV cache (GPU: FP16 fallback) | 5,120 MB | FP16 — toggle via `GPU_Q4_0_KV=0` |
| Output projection | 1,900 MB | Q4_K quantized GPU kernel |
| MoE + scratch | ~460 MB | IQ2_XXS + temp buffers |
| **Total (Q4_0 GPU)** | **~3,562 MB** | **Fits 6.5-8GB GPU with 3GB headroom** |

---

## 🔬 Verification

### Tools
| Tool | Purpose |
|------|---------|
| `tools/layer_cos_sim` | Per-layer cosine similarity vs llama.cpp |
| `tools/analyze_intermediates.py` | Browse `DUMP_INTERMEDIATE_DIR` F32 binaries |
| `ref_dumper` | llama.cpp reference intermediate dumper |
| `DUMP_INTERMEDIATE_DIR` | 53 tensor types/layer from llama.cpp |

### DA Audit Status (Phase 28b — post-DA correction)
| Claim | Status | Evidence |
|-------|--------|---------|
| GPU decode 8.5 tok/s | ✅ Verified (old code, SSM on CPU) | gen_text_gpu at 4K ctx, GPU_SUPPORT was dead |
| Q5_K fused matmul identical | ❓ Unverified in pipeline | Only compared vs old cuBLAS path (also unused) |
| SSM beta/alpha fused kernel | 🟡 Verified in isolation | **cos-sim 1.0 vs old cuBLAS path — NEVER tested in full inference** |
| SSM conv/SiLU/split fused kernel | 🟡 Verified in isolation | **cos-sim 1.0 vs separated path — NEVER tested in full inference** |
| GPU_SUPPORT builds and runs | ✅ Phase 28 | wubu_model_gpu_ssm_forward_full() called per layer |
| SSM GPU path correct | ❓ **NEVER VERIFIED** | Cos-sim vs CPU path: 0 comparisons done |
| 256k output cos-sim > 0.99 | ❓ Never verified | Only tested at small context, with dead GPU_SUPPORT |
| MoE expert offload efficiency | ❓ Not profiled | Speculative bottleneck claims — no data |
| F32 dequant VRAM waste | 🔴 ~2.2 GB | F32 weights uploaded but never used |
| GPU memory leak | 🔴 ~5.5 GB | d_attn_qkv_q[40] etc. never freed |

---

## 💡 Key Optimizations (Phase 25)

| Optimization | File | Gain |
|-------------|------|------|
| Fused Q5_K matmul (no bv[256] spill) | `gpu_quant_matmul.cu` | Eliminates local mem spill (~15 vs 256 regs) |
| Fused Q6_K matmul (same pattern) | `gpu_quant_matmul.cu` | Same |
| Fused SSM beta/alpha for N=1 | `cuda_kernels.cu` | 1 kernel replaces 2 cuBLAS + 4 element-wise |
| Q4_0 KV cache (default) | `wubu_model_gpu.cu` | 4:1 compression, 3.68GB saved at 256k |
| Q4_0 fused decode attn | `cuda_kernels.cu` | 8.1 tok/s vs FP16 7.6 |
| Sliding window attention | `cuda_kernels.cu` | GQA_WINDOW env, extra 16% at 256k |

---

## 📁 Project Structure

```
bytropix/
├── src/             # Core C/CUDA implementation
├── include/         # Headers
├── tools/           # Scripts + tools
├── .hermes/         # Mind palace, vault, papers
├── DIAGRAMS/        # SVG architecture diagrams
├── THEORY/          # Research papers
├── vault/           # Archives
├── data/            # Pre-extracted data
└── README.md        # This file
```

---

## 📚 References

- `.hermes/mind-palace/` — State, plan, goal-mantra, prestige (5 files)
- `.hermes/vault/deepseek-collection/` — 28 DeepSeek papers (V3, V3.2, V4, MoE, NSA)
- `.hermes/vault/benchmarks/` — External benchmark data
- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference binary

---

<div align="center">

*Engine: bytropix — from-scratch C inference for Qwen3.6-35B-A3B. Phase 28b: GPU_SUPPORT fixed and live, awaiting SSM GPU path verification.*
*DA principle: every claim must be verified at runtime against a reference. Unverified claims marked ❓ or 🔴.*

</div>

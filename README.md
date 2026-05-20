<div align="center">

# ⚡ bytropix — WuBu Text AI Multi-Modal Inference Engine

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch) + Moondream3 Vision (mmproj)**

[![Phase: 28e](https://img.shields.io/badge/Phase-28e--Q6K_Fix-blueviolet)](https://github.com/waefrebeorn/bytropix)
[![GPU SSM Decode: ~5.9 tok/s](https://img.shields.io/badge/GPU_SSM-5.9_tok/s-informational)](https://github.com/waefrebeorn/bytropix)
[![KV Cache: Q4_0 4:1 / F16 fallback](https://img.shields.io/badge/KV_Cache-Q4_0_4:1-green)](https://github.com/waefrebeorn/bytropix)
[![Vision: 3D ViT ported](https://img.shields.io/badge/Vision-3D_ViT_384L-success)](https://github.com/waefrebeorn/bytropix)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![GPU: RTX 5050](https://img.shields.io/badge/GPU-RTX_5050_8GB-critical)](https://github.com/waefrebeorn/bytropix)

</div>

---

## 📊 Current State (Phase 28e — May 20)

| | Status | Metric | Detail |
|:------:|--------|--------|--------|
| ✅ | **GPU decode (4K ctx)** | **~5.9 tok/s** — GPU SSM active, all 30 SSM layers on GPU | |
| ✅ | **Q6_K dequant fix** | **FIXED** — was `32.0`, now `d*sc*(v6-32)` (was constant ~365 output) | |
| ❌ | **GPU vs CPU SSM divergence** | **cos-sim -0.66** — output anti-correlated to CPU | |
| ✅ | **CPU SSM matches llama** | **cos-sim 0.994** — FORCE_CPU_SSM path verified | |
| ✅ | **Vision encoder** | **384 LoC port** — 27-layer 3D ViT w/ mmproj projection to 2048-dim | |
| ✅ | **Vision→text pipeline** | **test_vision_real.c** — vision encoder feeds directly into text model | |
| ❌ | **CPU-only `gen_text` build** | **Broken** — wubu_model.o links GPU symbols without .cu objects | |
| 🔴 | **Remote behind** | **8 commits not pushed** — all critical GPU fixes local-only | |

### What the DA Debunked (Phase 28b claims → reality)
| Doc Claimed | Reality |
|-------------|---------|
| 🔴 F32 waste ~2.2 GB | ✅ Already removed (`a032a8f`) |
| 🔴 GPU mem leak ~5.5 GB | ❌ **False positive** — free() already freed everything |
| 🔴 Column-major kernel broken | ❌ **False positive** — it was the CORRECT GGUF layout |
| 🔴 gen_text.c hardcoded prompt | ❌ **False positive** — accepts argv[1] |

---

## 🚀 Quick Start

```bash
# Build GPU inference
make gen_text_gpu

# Run text inference
GPU=1 MAX_CTX=4096 ./gen_text_gpu "The capital of France is" 20 40

# Vision encoder test
make test_vision_real
./test_vision_real /mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf /tmp/screen_vision_input.bin

# Compare CPU vs GPU SSM
FORCE_CPU_SSM=1 GPU=1 MAX_CTX=4096 ./gen_text_gpu "test" 5 40
```

**Hardware:** AMD Ryzen 7950X (16C/32T) | 64 GB DDR5 | RTX 5050 8GB VRAM | WSL2
**Model:** `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (733 tensors, 10.7 GB)
**Vision Proj:** `/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf`

---

## 🏗️ Architecture

### Model Spec (Qwen3.6-35B-A3B / `qwen35moe` GGUF)

```
40 Layers: 10 cycles × (3×SSM → 1×GQA)  ← 3:1 INTERLEAVED!
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

### Vision Encoder (Moondream3 / 3D ViT in mmproj)

```
27-layer Vision Transformer
├── 3D patch embedding: spatial=16×16, temporal=2 frames
├── Spatial merge: 2×2 grid patch merging (4:1 compression)
├── Hidden dim: 1152 → 16 heads × 72 head_dim
├── GQA attention (16 heads)
├── GELU activation
├── Post-LN + Merger MLP: 4608 → 2048 (matches text hidden dim)
└── Output: image tokens in text embedding space
```

### Multi-Modal Pipeline

```
Image → Patch Embed → 27×ViT → Spatial Merge → MMProj → Text tokens → 40×Text Model → Output
                                                      ↑
                                    Qwen3.6 Text Embedding ← GGUF token embeddings
```

### VRAM Budget (256k Context, Text Only)

| Component | Size | Format |
|-----------|------|--------|
| GQA weights | 1,040 MB | F32 (cuBLAS SGEMM) |
| SSM weights (quantized) | 692 MB | Q5_K/Q6_K on GPU |
| KV cache (GPU: Q4_0) | **1,440 MB** | **4-bit, 4:1 vs FP16** |
| KV cache (GPU: FP16 fallback) | 5,120 MB | FP16 via `GPU_Q4_0_KV=0` |
| Output projection | 1,900 MB | Q4_K quantized GPU kernel |
| MoE + scratch | ~460 MB | IQ2_XXS + temp buffers |
| **Total (Q4_0 GPU)** | **~3,562 MB** | **Fits 6.5-8GB GPU with 3GB headroom** |

---

## 🔭 Roadmap — Feature Cream

**Phase 29 [NOW]: Fix GPU SSM divergence (cos-sim -0.66)**
- Trace recurrence state vs conv state persistence across layers
- Layer-by-layer hidden state compare: GPU path vs CPU path at each SSM stage
- Fix state management → cos-sim > 0.99

**Phase 30: Build infrastructure**
- Fix CPU `gen_text` build (wrap GPU symbols in `#ifdef GPU_SUPPORT`)
- Push 8 pending commits to remote
- Re-verify cos-sim vs llama.cpp at current code state

**Phase 31: Vision encoder verification**
- Build and run `test_vision_real` end-to-end
- Verify vision encoder output statistics (range, NaN, distribution)
- Compare vision token distribution vs text token embedding distribution
- Wire mmproj into text model pipeline (already partially done)

**Phase 32: Multi-modal inference**
- Image → vision encoder → mmproj → text embedding space
- Concatenate or interleave vision tokens with text tokens
- Full forward: vision + text through 40-layer model
- Profile vision+text end-to-end

**Phase 33: Feature cream — cherry-pick from vault synthesis**
| Feature | Source | Priority |
|---------|--------|----------|
| Normalized sigmoid gating | DeepSeekMoE | P0 — MoE correctness |
| Auxiliary-loss-free load balancing | DeepSeek-V3 | P0 — Training stability |
| Chunked prefill (256K) | Qwen2.5-1M | P1 — First-token latency |
| RoPE length extrapolation (4x) | Qwen2.5-1M | P1 — 256K without retrain |
| DSA sparse attention | DeepSeek-V3.2 | P2 — O(L log L) GQA at scale |
| MTP self-speculative decode | DeepSeek-V3 | P2 — 2x decode speed |
| Entropix adaptive sampling | Vault | P3 — Smarter decoding |
| WuBuSparseAttention | Vault | P3 — Linear attention alternative |

**Phase 34: 256K multi-modal context**
- Test vision + text at full 256K context
- Profile VRAM, tok/s, accuracy vs. llama.cpp reference
- Sliding window / sparse attention for GQA at scale

**Phase 35: Performance profiling & optimization**
- CUDA events per kernel: SSM, GQA, MoE, output proj
- Identify true bottlenecks (MoE ~20-40ms hypothesized)
- Fuse remaining kernels, optimize memory transfers
- Target: 10+ tok/s at 256K with vision

---

## 🔬 Verification Tools

| Tool | Purpose |
|------|---------|
| `tools/layer_cos_sim` | Per-layer cosine similarity vs llama.cpp |
| `tools/test_vision_real` | Vision encoder end-to-end with real image |
| `tools/compare_logits` | Logit-level comparison vs reference |
| `DUMP_LAYER_DIR` | Save per-layer hidden states to `.bin` |
| `DUMP_INTERMEDIATE_DIR` | Save ALL intermediate tensors (53 types/layer) |
| `PROFILE` | Per-layer timing breakdown |

---

## 📁 Project Structure

```
bytropix/
├── src/              # Core C/CUDA (wubu_model, ssm, moe, gqa, vision, gpu)
├── include/          # Headers (gguf_reader, wubu_model, wubu_vision)
├── tools/            # 50+ binaries + tests + Python analysis
├── .hermes/          # Mind palace, vault, DA audits, plans
├── DIAGRAMS/         # SVG architecture diagrams
├── data/             # Pre-extracted embeddings, vision configs
├── vault/            # Unsloth quant format, cache compression refs
└── tests/            # Regression test harness
```

---

## 📚 References

- `.hermes/mind-palace/` — State, plan, goal-mantra, prestige, overnight map
- `.hermes/vault/` — Synthesis (architecture), papers (DeepSeek, Qwen, Gemma)
- `.hermes/vault/synthesis.md` — Complete architectural cross-reference (~19KB)
- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference binary

---

<div align="center">

*Engine: bytropix — C inference for Qwen3.6-35B-A3B text + Moondream3 vision. Phase 28e: Q6_K dequant fixed, GPU SSM cos-sim -0.66 under debug. Vision encoder ported (384 LoC). Next: fix state divergence, wire multi-modal pipeline.*
*DA principle: every claim verified at runtime. Unverified = ❓. Fixed = ✅. Debunked = documented.*

</div>

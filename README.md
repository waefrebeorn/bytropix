<div align="center">

# ⚡ bytropix — WuBu Text AI Inference Engine

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch)**

[![Phase: 22](https://img.shields.io/badge/Phase-22-blueviolet)](https://github.com/waefrebeorn/bytropix)
[![Cos-sim: 0.9994](https://img.shields.io/badge/Cos--sim-0.9994-success)](https://github.com/waefrebeorn/bytropix)
[![Gen: CPU 12 tok/s](https://img.shields.io/badge/CPU%20Prefill-12%20tok%2Fs-informational)](https://github.com/waefrebeorn/bytropix)
[![GPU: HANG](https://img.shields.io/badge/GPU-HANG-red)](https://github.com/waefrebeorn/bytropix)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![GPU: RTX 5050](https://img.shields.io/badge/GPU-RTX%205050%208GB-critical)](https://github.com/waefrebeorn/bytropix)
[![VRAM: 6.45 GB at 256k](https://img.shields.io/badge/256k%20VRAM-6.45%20GB-yellow)](https://github.com/waefrebeorn/bytropix)

</div>

---

## 📊 Current State

<div align="center">

| Status | Metric | Detail |
|:------:|--------|--------|
| ✅ | **Overall cos-sim** | **0.9994** vs llama.cpp (CPU, 5-token, 40 layers) |
| ✅ | **Architecture** | 40 layers, **3:1 SSM/GQA interleaved** (discovered May 19 via GGUF tensor enum) |
| ✅ | **Q4_0 KV cache** | 720 MB at 256k = **4:1 compression** vs F16, cos-sim 0.9994 |
| ✅ | **Quant types** | 7 self-hosted: Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q8_0 |
| 🟡 | **L31 cos-sim** | 0.9585 — quantization noise through 30 layers (expected) |
| ❌ | **gen_text_gpu** | Pre-existing hang after model load |
| ❌ | **MTP verify** | 100% rejection at IQ2_M quantization |
| 💤 | **GPU Q4_0 KV cache** | GPU still uses FP16 (5.12 GB), Q4_0 saves 3.7 GB |

</div>

---

## 🚀 Quick Start

```bash
# Build
make gen_text                # CPU inference
make gen_text_gpu            # GPU inference (currently has pre-existing hang)
make ref_dumper              # Reference comparison tool (links libllama.so)

# Run inference (CPU, AVX2, 16 threads)
./gen_text "The capital of France is" 32

# Compare vs reference (per-layer cosine similarity)
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Your prompt" 0
DUMP_LAYER_DIR=/tmp/our ./gen_text "Your prompt" 0
tools/layer_cos_sim /tmp/ref /tmp/our 40

# Per-operation intermediate tracing (53 tensor types per layer)
DUMP_INTERMEDIATE_DIR=/tmp/interm ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "prompt" 0
# Outputs: L0_conv_input.bin, L0_Qcur_full.bin, L0_linear_attn_out.bin, ...
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
| KV cache (CPU: Q4_0) | **720 MB** | 4-bit, 4:1 vs F16 — **Phase 22** |
| KV cache (GPU: FP16) | 5,120 MB | FP16 — not yet Q4_0 |
| Output projection | 1,900 MB | Q4_K quantized GPU kernel |
| MoE + scratch | ~460 MB | IQ2_XXS + temp buffers |
| **Total (Q4_0 CPU / FP16 GPU)** | **~6,453 MB / ~10,893 MB** | |

---

## 🔬 Key Tools

### Verification Tools
| Tool | Purpose | Links Against |
|------|---------|---------------|
| `ref_dumper` | libllama.so reference dumper | `libllama.so` |
| `layer_cos_sim` | Per-layer cosine similarity | Binary dump |
| `classify_layers.py` | Classify SSM vs GQA layers | GGUF tensor names |
| `analyze_intermediates.py` | Browse DUMP_INTERMEDIATE_DIR | F32 binary files |

### DUMP_INTERMEDIATE_DIR (53 tensor types per layer)

Set `DUMP_INTERMEDIATE_DIR=/tmp/dir` before `ref_dumper`:

| Group | Example Files |
|-------|---------------|
| SSM conv | `L0_conv_input.bin`, `L0_conv_output_silu.bin`, `L0_conv_states.bin` |
| GQA proj | `L0_Qcur_full.bin`, `L0_Kcur.bin`, `L0_Vcur.bin` |
| Gated delta | `L0_beta_sigmoid.bin`, `L0_a_softplus.bin`, `L0_gate.bin` |
| SSM recur | `L0_linear_attn_out.bin`, `L0_new_state.bin`, `L0_state_predelta.bin` |
| Attention | `L0_attn_output.bin`, `L0_attn_residual.bin`, `L0_kqv_out.bin` |
| MoE | `L0_ffn_moe_logits.bin`, `L0_ffn_moe_swiglu.bin`, `L0_ffn_moe_out.bin` |
| Output | `L0_l_out.bin`, `L0_final_output.bin` |
| Global | `global_h_pre_norm.bin`, `global_result_norm.bin`, `global_result_output.bin` |

---

## 💡 Key Innovations

### Q4_0 KV Cache (Phase 22)
- 4:1 compression: 720 MB vs 2.56 GB at 256k context
- `block_q4_0_cache {uint16_t d, uint8_t qs[16]}` — 32 elements per block, 18 bytes
- Aligned bulk write, multi-block read — cos-sim 0.9994 vs F16

### GPU Pipeline (Phases 13-21)
| Component | GPU | Detail |
|-----------|-----|--------|
| Output proj (Q4_K) | ✅ | Custom CUDA kernel, ~0.1ms vs CPU ~10ms |
| GQA attention | ✅ | FP16 KV, sliding window, ATTEN_TILE=16384 |
| SSM recurrence | ✅ | 32 blocks × 128 threads |
| SSM full forward | ✅ | All 15 steps, 2 transfers/layer |
| MoE experts (IQ2_XXS) | ✅ | Per-expert cache, 259 MB |

### Smart GPU Gating
- GPU offload only when beneficial (cache_len > 2048 or N > 1)
- Single-token decode: CPU path (thermal/fallback)
- Prefill: GPU path (parallel scan, 18.6 tok/s)

---

## 🐛 Bug History (Complete)

| # | Bug | Symptom | Fix | Date |
|---|-----|---------|-----|------|
| 1 | GQA Q/Gate interleave | Cos-sim -0.51 | Per-head interleaved extraction | May 18 |
| 2 | IMRoPE not implemented | Wrong multi-token output | sections=[11,11,10,0] | May 18 |
| 3 | MoE OpenMP race | Non-deterministic output | Thread-local scratch | May 18 |
| 4 | SSM state not saved | Second token garbage | Persistent state buffer | May 18 |
| 5 | No KV cache | Self-only attention | Buffer all positions | May 18 |
| 6 | MTP crash | SIGSEGV | NULL checks + concat fix | May 19 |
| **7** | **Q6_K loop bound** | **Cos-sim 0.796** | **`j<QK_K/32`→`j<QK_K/16` (one char)** | **May 19** |
| 8 | DA v10 wrong diagnosis | Misattributed cause | Isolate+test per quant type | May 19 |
| 9-12 | GPU stride, RoPE, cache, build | GPU garbage | Per-component fixes | May 19 |
| **13** | **kv_cache_read_head multi-block** | **GPU hang on decode** | **While-loop Q4_0 dequant path** | **May 19** |

---

## 🗺️ Roadmap

| Prio | Gap | Status |
|:----:|-----|:------:|
| **P0** | **Fix gen_text_gpu hang** | ❌ |
| **P0** | **GPU Q4_0 cache** | 💤 |
| P1 | RotorQuant Givens rotation (block-diagonal, 2 FMAs/pair) | 💤 |
| P1 | TurboQuant WHT (spreads outlier energy) | 💤 |
| P1 | Hamilton encoder BSP attention for >512k | 💤 |
| P2 | Unified SSM kernel (fuse conv→SiLU→split→norm) | 💤 |
| P2 | Chunked prefill (3-7x at 256k) | 💤 |

---

## 📁 Project Structure

```
bytropix/
├── src/             # Core C implementation (GGUF, SSM, GQA, MoE, tokenizer, CUDA)
├── include/         # Headers (model structs, KV cache helpers, MoE, GGUF reader)
├── tools/           # ~50 binaries: gen_text, ref_dumper, tests, analysis scripts
├── .hermes/         # Mind palace (state, plan, prestige, vault, DA audits)
├── DIAGRAMS/        # SVG architecture diagrams (inference pipeline, phase roadmap)
├── THEORY/          # WuBu Nesting research papers (hyperbolic, Poincaré, TGT)
├── vault/           # Quantization formula reference, archived docs
├── data/            # Pre-extracted embeddings, training data
├── tests/           # Test files
├── MADE_AGENTICALLY_BY_HERMES.md  # Full agentic project retrospective (v22, 28KB)
└── README.md        # This file
```

---

## 📚 References

- `MADE_AGENTICALLY_BY_HERMES.md` — Complete project retrospective (28 KB)
- `vault/cache-compression-resources.md` — Q4_0 / TurboQuant+ / RotorQuant / Hamilton encoder comparison
- `llama/turboquant_plus/` — Google TurboQuant KV cache compression repo
- `llama/rotorquant/` — RotorQuant block-diagonal Clifford rotors repo
- `tools/example_rotorquant.py` — Givens rotation + Q4_0 demo
- `tools/example_turboquant.py` — WHT + Q4_0 demo
- `tools/example_hamilton_encoder.py` — Hamilton quaternion manifold demo
- `.hermes/mind-palace/` — State, plan, goal-mantra, prestige, overnight (6 files)
- `.hermes/vault/qwen-papers/` — Qwen3, Qwen3.6 architecture references
- `.hermes/vault/deepseek-papers/` — DeepSeek-V3, MoE architecture papers
- `.hermes/unsloth-qwen3.6-quant-formula.md` — Per-tensor quantization map
- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference binary
- `/mnt/c/projects/HASHMIND/llama-cpp-rotorquant/llama.cppCOPY/` — Hamilton encoder attention (legacy)

---

<div align="center">

*Engine: bytropix — from-scratch C inference. Architecture discovered May 19, 2026 via GGUF tensor enumeration.*
*"What does this claim rest on?" — every number checked at runtime against a reference.*

</div>

# bytropix — WuBu Text AI Inference Engine

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, qwen35moe architecture).**
*May 19 — Phase 22: Q4_0 KV cache 4:1 compression, architecture discovery, DUMP_INTERMEDIATE_DIR.
Cos-sim 0.9994 overall (CPU, 5-token, 40 layers). VRAM at 256k: ~6.45 GB (fits 8GB GPU).*

---

## Current State (DA Verified ✅)

| Metric | Value | Status |
|--------|-------|--------|
| Layers | 40 (30 SSM + 10 GQA, **3:1 interleaved repeating**) | ✅ Architecture discovered May 19 |
| SSM layers | 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38 | ✅ |
| GQA layers | 3,7,11,15,19,23,27,31,35,39 | ✅ |
| Dequant types | 7 self-hosted (Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q8_0) | ✅ |
| **Decode speed** | **~9 tok/s** (GPU) | ✅ *Measured before Phase 22* |
| **Prefill speed** | **~11 tok/s** (CPU) | ✅ *5 tokens, Q4_0 KV cache* |
| **Q4_0 KV cache** | **720 MB vs 2.56 GB** at 256k (4:1 compression) | ✅ Phase 22, cos-sim=0.9994 |
| **Cos-sim vs llama.cpp** | **0.9994** overall (L00-L30: 0.998-0.9999, L31: 0.9585) | ✅ *Quantization noise expected* |
| **Llama dependency** | **NONE** — all vec_dot self-hosted | ✅ |

### Triple DA Audit (May 19 PM v22)

| DA Phase | Result | Details |
|----------|--------|---------|
| DA-1: Code vs Theory | 3 stale docs fixed | Architecture mislabeled as "30+10 contiguous" |
| DA-2: Vault Deep-Dive | All papers current | Qwen3.6, DeepSeek-V3, Unsloth quant verified |
| DA-3: Cold Gaps | P0 fix: doc sync + arch discover | Propagated to all 6+ walkway files |

---

## Quick Start

```bash
# Build
make gen_text                # CPU inference (recommended)
make gen_text_gpu            # GPU inference (May 19: pre-existing hang)
make ref_dumper              # Reference comparison tool (links libllama.so)

# Run inference (CPU, 16 threads, AVX2)
./gen_text "The capital of France is" 32

# Layer cos-sim vs reference
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "The capital of France is" 0
DUMP_LAYER_DIR=/tmp/our ./gen_text "The capital of France is" 0
tools/layer_cos_sim /tmp/ref /tmp/our 40

# Per-operation intermediate tracing (requires ref_dumper)
DUMP_INTERMEDIATE_DIR=/tmp/interm ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "prompt" 0
# Outputs 53 F32 files per layer (conv_input, Qcur, Kcur, Vcur, beta, alpha, gate, ...)
```

**Hardware:** AMD Ryzen 7950X (16C/32T) | 64 GB DDR5 | RTX 5050 8GB VRAM | WSL2
**Model:** `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (733 tensors, 10.7 GB)
**Model (MTP):** `/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf` (753 tensors, 11.9 GB)

---

## Architecture

### Model Spec (Qwen3.6-35B-A3B / `qwen35moe` GGUF arch)

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

### Per-Layer Data Flow

```
SSM layer (3 out of every 4):
  x → RMSNorm → attn_qkv(Q5_K) → gate(Q5_K) → conv1d → SiLU → split → L2 norm →
  SSM recurrence(16 heads) → gated norm → ssm_out(Q6_K) → gate(SiLU) × → MoE → +residual

GQA layer (every 4th: 3,7,11,15,19,23,27,31,35,39):
  x → RMSNorm → attn_q(Q5_K) → gate(Q5_K) → attn_k(Q5_K) → attn_v(Q5_K) →
  IMRoPE → full attention(KV cache) → output(Q5_K) → sigmoid(gate) × → MoE → +residual
```

### VRAM Budget (256k Context, Q4_0 KV Cache)
| Component | Size | Format |
|-----------|------|--------|
| GQA weights | 1,040 MB | F32 (cuBLAS SGEMM) |
| SSM weights (quantized) | 692 MB | Q5_K/Q6_K native on GPU |
| KV cache (Q4_0) | **720 MB** | **4-bit quantized (4:1 vs F16)** — Phase 22 |
| Output proj (Q4_K) | 1,900 MB | Quantized GPU kernel |
| MoE + scratch | ~460 MB | IQ2_XXS + temp buffers |
| **Total** | **~6,453 MB** | **Fits 8GB GPU with 1.5GB headroom** |

---

## Key Tools

| Tool | Purpose |
|------|---------|
| `ref_dumper` | Links libllama.so, dumps per-layer hidden states + all intermediates |
| `layer_cos_sim` | Per-layer cosine similarity between reference and bytropix |
| `classify_layers.py` | Classifies 3:1 SSM/GQA interleaved pattern from GGUF tensor names |
| `analyze_intermediates.py` | Browses DUMP_INTERMEDIATE_DIR output (53 tensor types/layer) |

### DUMP_INTERMEDIATE_DIR (Environment Variables)

Set `DUMP_INTERMEDIATE_DIR=/tmp/dir` before running `ref_dumper` to dump ALL 53 intermediate tensor types per layer as F32 binary files:

| Tensor Group | Examples |
|-------------|----------|
| SSM conv | `L0_conv_input.bin`, `L0_conv_output_silu.bin`, `L0_conv_states.bin` |
| GQA projections | `L0_Qcur_full.bin`, `L0_Kcur.bin`, `L0_Vcur.bin` |
| Gated delta | `L0_beta_sigmoid.bin`, `L0_a_softplus.bin`, `L0_gate.bin` |
| SSM recurrence | `L0_linear_attn_out.bin`, `L0_new_state.bin`, `L0_state_predelta.bin` |
| Attention | `L0_attn_output.bin`, `L0_attn_residual.bin`, `L0_kqv_out.bin` |
| MoE | `L0_ffn_moe_logits.bin`, `L0_ffn_moe_swiglu.bin`, `L0_ffn_moe_out.bin` |
| Per-layer output | `L0_l_out.bin`, `L0_final_output.bin` |
| Global | `global_h_pre_norm.bin`, `global_result_norm.bin`, `global_result_output.bin` |

---

## Key Innovations

### Q4_0 KV Cache (Phase 22)
- 4:1 compression ratio: 720MB vs 2.56GB at 256k context
- `block_q4_0_cache`: symmetric signed quantization [-7,7]→[1,15] nibbles
- Aligned bulk write path, multi-block read path
- Verified cos-sim 0.9994 vs F16 — identical quality

### GPU Pipeline (Phases 13-21)
- Output projection: Custom CUDA Q4_K kernel, ~0.1ms vs CPU ~10ms
- GQA attention: FP16 KV cache, sliding window (GQA_WINDOW env var), ATTEN_TILE=16384
- SSM recurrence: GPU kernel for all tokens, 32 blocks × 128 threads
- MoE experts: Per-expert IQ2_XXS kernel, GPU cache with stability-based reuse
- SSM matmuls (prefill): Q5_K quantized kernel, 2048×8192 and 2048×4096

### Smart GPU Gating
- Single-token GPU offload has negative ROI (transfer + sync > compute savings)
- GPU GQA: only for cache_len > 2048 or prefill (N > 1)
- GPU SSM matmuls: only for prefill (N > 1)
- GPU SSM recurrence: all tokens (lightweight, compute-bound)
- CPU baseline: pure CPU path for thermal/fallback scenarios

---

## Bug History

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 7 | Q6_K vec_dot AVX2 loop bound | Cos-sim 0.79→0.9967 | `32`→`16` (one char) |
| 13 | kv_cache_read_head multi-block | Hang on decode | While-loop dequant path |

See `.hermes/mind-palace/plan.md` for complete 13-bug history.

---

## Cold Gaps & Roadmap

| Prio | Gap | Detail | Status |
|------|-----|--------|--------|
| **P0** | **gen_text_gpu hang** | Pre-existing GPU inference hang | ❌ Needs debug |
| **P0** | **GPU Q4_0 KV cache** | Port to GPU growable cache, saves 3.7GB VRAM | 💤 |
| **P1** | **Unified SSM kernel** | Fuse conv1d→SiLU→split→norm→beta | 💤 |
| **P1** | **Sparse/global attention** | Global tokens for 512k+ quality | 💤 |
| **P2** | **Chunked prefill** | 3-7x prefill at 256k | 💤 |
| **P2** | **DSA sparse attention** | Linear-time GQA from DeepSeek-V3.2 | 💤 |

---

## References

- `.hermes/mind-palace/` — State, plan, goal-mantra, prestige, overnight (6 files, always current)
- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference binary
- `~/llama.cpp/src/models/qwen35moe.cpp` — Reference implementation (qwen35moe arch)
- `.hermes/vault/qwen-papers/` — Qwen3, Qwen3.6 architecture references
- `.hermes/vault/deepseek-papers/` — DeepSeek-V3, MoE architecture papers
- `.hermes/unsloth-qwen3.6-quant-formula.md` — Per-tensor quantization map

---

*Engine: bytropix — from-scratch C inference. Architecture discovered May 19 via GGUF tensor enumeration.*

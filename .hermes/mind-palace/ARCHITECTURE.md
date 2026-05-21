# WuBuNesting Inference Engine — ARCHITECTURE (May 21, 2026)

**Project**: bytropix — custom C inference engine for Qwen3.6-35B-A3B (qwen35moe)  
**Author**: waefrebeorn / WuBuText AI  
**Model**: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (Unsloth Dynamic 2.0 quantization)  
**Vision**: Moondream3 3D ViT via mmproj, 27 layers, 1152 hidden  
**Reference**: llama.cpp (libllama.so via tools/ref_dumper.cpp)  
**GPU**: CUDA 13.1, sm_120 (RTX 5050 Blackwell)  
**CPU-only cos-sim**: ~0.9968 vs reference (all known bugs fixed)  
**GPU text**: NET-NEGATIVE — CPU-only is 2-5x faster for quantized weights  
**GPU vision**: 122x faster — THE GPU win (pure F32 SGEMM)

---

## 1. ENGINE OVERVIEW

### What It Does
Load GGUF Qwen3.6-35B-A3B model → quantized inference on CPU (optimal path) → produce coherent text. GPU optionally accelerates: vision encoder (122x), SSM/GQA hybrid (net-negative for text). MTP speculative decode working at 8.5 tok/s. Full vision→text pipeline verified.

### Architecture (one forward step, CPU path)

```
Token Embedding (Q5_K matmul) [248320×2048]
    ↓
40× Layer Loop ([SSM,SSM,SSM,GQA] × 10):
    ├── rms_norm (F32, D_MODEL=2048)
    ├── SSM layer (30x):
    │   ├── attn_qkv → Q5_K [2048→8192]
    │   ├── conv1d kernel=4, F32
    │   ├── SSM recurrence (d_state=128)
    │   ├── ssm_out → Q6_K [4096→2048]
    │   ├── attn_gate → sigmoid gate
    │   └── MoE: 8/256 experts × 3 matmuls (gate IQ2_XXS, up IQ2_XXS, down IQ3_XXS)
    ├── GQA layer (10x):
    │   ├── attn_q/gate [2048→8192] + attn_k [2048→512] + attn_v [2048→512] Q5_K
    │   ├── IMRoPE sections [11,11,10,0], θ=10M
    │   ├── KV cache + attention (2 KV heads × 16 Q heads)
    │   ├── attn_output [4096→2048] Q5_K
    │   └── MoE: same as SSM
    └── residual_add
    ↓
rms_norm → output_proj (Q4_K: 2048×248320) → softmax → sample → token
```

### Key Stats
| Metric | Value |
|--------|-------|
| Parameters | 35B total, ~3B active per token (8/256 experts) |
| Weight size | 10.7 GB GGUF (11.5 GB file) |
| Layers | 40 (30 SSM + 10 GQA, 3:1 repeating) |
| Experts | 256 routed + 1 shared, 8 active |
| D_MODEL | 2048 |
| D_FF (expert) | 512 |
| Vocab | 248320 (padded, BOS/EOS=248044) |
| Decode speed (CPU) | 8.9 tok/s (16 threads, OMP) |
| Decode speed (GPU hybrid) | 5.5 tok/s (NET-NEGATIVE) |
| Prefill speed (CPU) | 17.8 tok/s |
| MTP decode | 8.5 tok/s, 4% acceptance |
| Vision→text pipeline | 15.7s (GPU ViT 0.52s + CPU text 6.3s) |
| CPU-only cos-sim vs ref | 0.9968 |

---

## 2. FILE LAYOUT

```
/home/wubu/bytropix/
├── src/                    ← Core engine (C + CUDA)
│   ├── wubu_model.c        ← MAIN: model load, layer forward, gen_text
│   ├── wubu_model_gpu.cu   ← GPU forward: GQA prefill, quant matmul dispatcher
│   ├── wubu_gguf.c         ← GGUF reader
│   ├── wubu_ssm.c/h        ← SSM kernel (Gated DeltaNet recurrence)
│   ├── wubu_gqa.c/h        ← GQA kernel (GQA + IMRoPE)
│   ├── wubu_moe.c/h        ← MoE kernel (router + expert FM)
│   ├── wubu_norm.c/h       ← rms_norm, layer_norm
│   ├── wubu_vision.c/h     ← 3D ViT encoder + MMProj
│   ├── wubu_tokenizer.c/h  ← BPE tokenizer
│   ├── quantized_matmul.c  ← Quantized matmul driver (dispatch by type)
│   ├── quantized_dot_generic.c ← SIMD dot products (Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS)
│   ├── dequant_iq2_xxs.c   ← IQ2_XXS dequant tables
│   ├── cuda_kernels.cu     ← GPU RMSNorm, SiLU, quant matmul kernels
│   ├── gpu_moe_kernel.cu   ← GPU MoE expert compute (v5, Q8_K quantization)
│   ├── gpu_output_proj.cu  ← GPU output projection (cuBLAS SGEMM)
│   ├── gpu_quant_matmul.cu ← GPU batched quant matmul (Q5_K/Q6_K)
│   ├── gpu_ssm_recurrence.cu ← GPU SSM recurrence kernel
│   └── ggml.h              ← GGML type enums, block structs
├── include/                ← Public headers
├── tools/                  ← 200+ tools: gen_text, test tools, debug tools
├── Makefile                ← Build system
└── .hermes/mind-palace/    ← DA audits, vault, plans, state
```

---

## 3. QUANTIZATION TYPES (Verified)

| GGML Type ID | Name | Used For | bpw | SIMD | GPU |
|:---:|------|----------|:---:|:----:|:---:|
| 0 | F32 | Norms, biases, routers, SSM params | 32 | — | ✅ |
| 12 | Q4_K | output.weight | 5.0 | SSE/AVX2 | 🔲 |
| 13 | Q5_K | attn_qkv, gate/up shared, token_embd | 6.5 | SSE/AVX2 | ✅ batched kernel |
| 14 | Q6_K | ssm_out, shared down | 7.5 | SSE/AVX2 | ✅ batched kernel |
| 16 | IQ2_XXS | ffn_gate_exps, ffn_up_exps | 2.2 | C-only (grid lookup) | ✅ (v5 kernel) |
| 18 | IQ3_XXS | ffn_down_exps (37/40 layers) | 3.3 | C-only (grid+ksigns) | ✅ (v5 kernel) |
| 23 | IQ4_XS | ffn_down_exps (3/40 layers) | 4.3 | C-only | 🔲 |

---

## 4. GPU vs CPU: When Each Wins (CRITICAL DESIGN DOC)

### GPU Wins: Pure F32 operations (no quantized weights)
- **Vision encoder**: 27 ViT layers, all F32 SGEMM. GPU: 0.52s, CPU: 63.7s (122x)
- **MMProj**: F32 SGEMM 1152×2048×128. GPU cuBLAS: ~10ms, CPU: ~24s (2400x)
- **Output projection**: Could benefit if F32, but it's Q4_K quantized so ~6ms on CPU already fast

### CPU Wins: Everything with quantized weights
- **MoE**: IQ2_XXS/IQ3_XXS dequant + matmul. CPU has optimized C path with grid lookups. GPU adds H2D/D2H overhead
- **SSM recurrence**: 128-dim state, small enough for CPU. GPU adds kernel launch latency
- **GQA attention**: Small KV cache at 1 token decode. GPU overhead dominates
- **RMSNorm + SiLU**: Trivial on CPU (<1% of total time)

### Why GPU is NET-NEGATIVE for Text
1. **H2D/D2H overhead**: Every token needs ~8KB hidden state transferred per layer. At 40 layers: 320KB per token. On PCIe 4.0 x4 (RTX 5050 mobile): ~40µs per direction → 3.2ms per token overhead
2. **GPU init heating**: First CUDA call warms up the GPU, which shares thermal budget with CPU. CPU throttles from sustained 85°C to 3.5 GHz (from 5.0 GHz turbo)
3. **Quantized weights**: 11GB model on GPU VRAM, but IQ types need CPU-side grid lookup tables that don't port well
4. **Small batch decode**: 1 token at a time. GPU designed for throughput (many parallel operations), not latency

### Recommended Configuration
- `./gen_text` (CPU-only, no GPU=1) — 8.9 tok/s decode
- `./infer_vision_text_gpu` for images — GPU ViT + CPU text
- `FORCE_CPU_MOE=1 GPU=1 ./gen_text_gpu` — hybrid if GPU already warm (5.5 tok/s, worse than CPU)

---

## 5. BUGS FOUND & FIXED (Historical)

### Phase 1-4: CPU Parity Bugs (All Fixed ✅)
1. GQA Q/gate interleave (May 18): cos-sim -0.51→0.9968
2. Output proj transpose: cos-sim -0.457→0.9969
3. SSM state carry: multi-token decode fixed
4. KV cache append-only: full attention per step
5. Tokenizer file open/close per step: fopen/fclose perf fix
6. IQ2_XXS block size: 72→66 bytes, NaN cascade fixed

### Phase 28: GPU Bugs (All Fixed ✅)
7. Q6_K dequant shift (May 18: c07cf14) — wrong block dequant
8. SSM state sync + output proj (08f5f23) — GPU SSM H2D sync
9. GQA interleaved layout (cdccde2) — same as Bug 1 for GPU path
10. Q5_K F16 denormals (bf573b8) — subnormal float16 from quantized weights
11. GPU MoE IQ3_XXS (9093c61) — IQ3_XXS down weight support
12. GPU vision LN residual (3464940) — separate d_residual pointer for in-place norm
13. GPU vision add_kernel symbol clash — unique kernel naming
14. Vision n_patches_total cap — prevent massive heap alloc
15. Vision scores[2304] stack overflow → heap-allocated

### GPU MoE 0.9888 cos-sim: NOT A BUG (DA v13)
- CPU quantize_row_q8_K: negative d, sign-inverted Q8
- GPU Q8_K quantize: positive d, same-sign Q8
- Both mathematically correct, different IEEE rounding
- 0.32% running error compounds through 40 layers → flips token in 240K vocab
- Hybrid path accepted. Not fixable without CPU code port to GPU (3-5 sessions)

---

## 6. PERFORMANCE PROFILE (16-thread CPU)

### Per-Token Decode Time Breakdown
| Component | Time | % | Notes |
|-----------|:----:|:-:|-------|
| MoE (8 experts × 3 matmuls) | ~9ms | 32% | Gate/up IQ2_XXS, down IQ3_XXS |
| GQA (10 layers) | ~7ms | 25% | KV cache read, attention, output proj |
| SSM (30 layers) | ~6ms | 21% | QKV, conv1d, recurrence, out proj |
| Output proj (2048×248320) | ~4ms | 14% | Q4_K matmul, largest single op |
| Norms + router + overhead | ~2ms | 7% | rms_norm × 40, router × 40 |
| **Total** | **~28ms per layer × 40** | | **112ms → 8.9 tok/s** |

### Bottleneck: Memory Bandwidth  
DDR5 ~50GB/s. Model 10.7GB. Minimum: 214ms/forward. Current 112ms → 1.9x above bandwidth limit (thanks to quantized weights and cache reuse).

---

## 7. MTP SPECULATIVE DECODE

### Model: /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9 GB)
- Base model + blk.40 (GQA + MoE) + nextn.* head
- Extra tensors: nextn.hnorm, nextn.enorm, nextn.eh_proj, nextn.shared_head_norm
- 753 tensors (vs 733 base)

### Current Performance
- 8.5 tok/s decode (vs 8.9 tok/s CPU — negligible difference)
- 4% MTP acceptance (very low — quantized IQ2_M head can't draft well)
- MTP=1: single speculative token per step
- Falls back to single-token without MTP=1

### Why Acceptance is Low
- MTP head weights are IQ2_M quantized like main model (no higher precision)
- Draft quality: MTP head uses only blk.40 (1 layer), much weaker than 40-layer main model
- For 83% acceptance (DeepSeek-V3 claim), need higher-precision MTP head or better draft model
- Practical use: MTP is not beneficial at current acceptance rate

---

## 8. VISION PIPELINE

### Architecture
```
ffmpeg screenshot → raw pixels → patch embed (3D: 16×16×2) → 
27× ViT layer (F32 SGEMM, RMSNorm, 1152 hidden, 72 head_dim) → 
spatial merge (spatial_merge_size=2) → MMProj (1152→2048) → 
text model inference → logits
```

### Performance
| Component | CPU | GPU | Speedup |
|-----------|:---:|:---:|:-------:|
| Patch embed | 0.2s | 0.2s | 1x |
| 27× ViT layers | 63.7s | 0.52s | **122x** |
| Spatial merge | 0.3s | 0.3s | 1x |
| MMProj (1152→2048) | 24s | ~10ms | **2400x** |
| Text model (decoding) | 4.77s | 6.3s | 0.76x |
| **Total** | **~93s** | **15.7s** | **5.9x** |

### GPU Vision Pipeline
- `infer_vision_text_gpu`: ffmpeg → GPU ViT (cuda_vision.cu) → GPU MMProj (cuBLAS) → CPU text
- 2 critical bugs fixed: in-place LN residual (separate d_residual param), add_kernel symbol clash
- Verified: 256×256 → 128 patches × 2048 dim
- Logit range [-10.77, 14.09], no NaN/Inf

---

## 9. CUDA sm_120 (Blackwell RTX 5050) — Compiler Bugs

### Bug 1: static __shared__ inside loops
**Symptom**: Kernels with `__shared__` arrays declared inside for-loop body hang on sm_120.  
**Fix**: `extern __shared__ float smem[]` + manual offset calculation.  
**Cost**: ~20 lines of pointer arithmetic per kernel.  
**Applied**: gpu_moe_kernel.cu v5.

### Bug 2: __syncthreads() + between-warps reduction
**Symptom**: Pattern (warp leaders→shared mem→__syncthreads→selected threads reduce) hangs.  
**Fix**: Thread 0 reads all warp leaders, serial reduction.  
**Applied**: gpu_moe_kernel.cu v5.

### Bug 3: extern __shared__ uint8_t + syncthreads
**Symptom**: `extern __shared__ uint8_t smem_u8[]` + `__syncthreads()` in loops causes wrong code gen.  
**Hypothesis**: Compiler aliasing analysis treats uint8_t/float as non-aliasing, mis-optimizes sync ordering.  
**Fix**: Use `extern __shared__ float smem[]` always.  
**Applied**: gpu_moe_kernel.cu v5.

### Bug 4: compute-sanitizer
**Symptom**: `compute-sanitizer` fails on WDDM (Windows driver for RTX 5050). Debugger not initialized.  
**Workaround**: Manual printf + cos-sim comparison with CPU reference.

### Hardware Opportunities (Unused)
| Resource | Available | Status |
|----------|-----------|--------|
| FP8 Tensor Cores | sm_120 native | Not used (FP32 only) |
| CUDA Graphs | Graph capture | Not used |
| Async H2D copies | Overlap with compute | Sequential expert uploads |
| Multi-block parallelism | 32 blocks/SM | 1 block per kernel |
| Shared memory | 48KB/block | ~8KB used |

---

## 10. BUILD & RUN

```bash
# Build
cd /home/wubu/bytropix
make gen_text              # CPU inference (RECOMMENDED)
make gen_text_gpu          # GPU hybrid (net-negative)
make gen_text_mtp          # MTP speculative decode
make test_vision_real      # Vision pipeline (CPU)
make test_vision_real_gpu  # Vision pipeline (GPU ViT + CPU text)

# CPU inference (optimal path)
./gen_text "The capital of France is" 32           # 32 tokens
CHAT=1 ./gen_text "Hello" 128                      # Chat mode

# GPU hybrid (net-negative, only if GPU already warm)
GPU=1 FORCE_CPU_MOE=1 ./gen_text_gpu "Hello" 64

# MTP speculative decode
MTP=1 OMP_NUM_THREADS=16 ./gen_text_mtp "Hello" 30

# Vision pipeline
./test_vision_real <mmproj.gguf> <pixels.bin> [H] [W] [model.gguf]

# Reference comparison
./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "The capital of" 0
./test_full_model

# Per-layer cos-sim
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper model.gguf
DUMP_LAYER_DIR=/tmp/our ./gen_text "prompt" 0
./layer_cos_sim /tmp/ref /tmp/our 40
```

---

## 11. RESEARCH PAPERS CONSULTED

| Paper | Source | Key Insight Applied |
|-------|--------|-------------------|
| DeepSeek-V3 (2412.19437) | vault/deepseek-papers/ | 256 experts/8 active ✅ confirmed; MTP; auxiliary-loss-free load balancing |
| DeepSeek-V3.2 (2512.02556) | vault/deepseek-papers/ | DSA sparse attention O(L log L) for P2 |
| DeepSeekMoE (2401.06066) | vault/deepseek-papers/ | Sigmoid gating, shared experts, fine-grained segmentation |
| DeepSeek-R1 (2501.12948) | vault/deepseek-papers/ | Pure RL for reasoning — post-training reference |
| Qwen2.5-1M (2504.05752) | vault/qwen-papers/ | Chunked prefill 3-7x, RoPE extrapolation 4x |
| Qwen3 (2505.XXXXX) | vault/qwen-papers/ | Thinking/non-thinking mode |
| Gemma 3 | vault/deepmind-2026/ | 30:10 local:global ratio validation |

---

*Updated: May 21, 2026 — Phase 28o*
*"CPU for quantized text, GPU for vision F32. Know your hardware."*

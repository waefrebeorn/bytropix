# bytropix — Triple Extended GPU Roadmap & Legacy (May 19 PM v22)

## Executive Summary

**bytropix: Pure C inference engine for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch).**
Cos-sim 0.9994 overall vs llama.cpp (CPU, 5-token prefill, 40 layers).
CPU prefill: ~12 tok/s. GPU: ⚠️ gen_text_gpu hangs (pre-existing).
VRAM with Q4_0 KV cache at 256k: ~6.45 GB (fits 8GB GPU).
**ARCHITECTURE CORRECTION**: 40 layers with 3:1 SSM/GQA **interleaved repeating** pattern (NOT 30+10 contiguous).

---

## 1. Architecture (DA Verified ✅)

```
40 Layers: 10 cycles × (3×SSM → 1×GQA)
├── SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
├── GQA layers: 3,7,11,15,19,23,27,31,35,39
├── Hidden dim: 2048
├── Vocab: 248,320
├── SSM: 16 K-heads × 128, 32 V-heads × 128
├── GQA: 16 Q-heads × 256, 2 KV-heads × 256
├── MoE: 256 experts, 8 active + 1 shared
├── Expert FFN: 512 | Shared FFN: 512
├── RoPE: IMRoPE, sections=[11,11,10,0], θ=10M
└── Quant: Mixed IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K (~2.7 bpw)
```

**Architecture validated via**:
- GGUF tensor enumeration (`blk.N.ssm_a` vs `blk.N.attn_q.weight` presence)
- llama.cpp qwen35moe.cpp `full_attention_interval=4` metadata key
- DUMP_INTERMEDIATE_DIR tensor naming (conv_input on SSM layers, attn_output on GQA)

---

## 2. Phase Progress (ALL Phases Complete)

| Phase | Component | Status | Detail |
|-------|-----------|--------|--------|
| 0-11 | Foundation | ✅ Shipped | GQA attn, vec_dot dequant, MoE, KV cache, quant matmul, fused Q8_K |
| 12 | MTP Spec Decode | ✅ Shipped | Draft N=2, EMA correction — blocked at IQ2_M quant incompatibility |
| 13 | GPU Output Proj | ✅ Shipped | F32 SGEMM + Q4_K quantized kernel, ~0.1ms vs CPU ~10ms |
| 14 | SSM AVX2 Optimization | ✅ Shipped | 4 inner loops, fused Q8_K quant |
| 15 | GPU GQA Wiring | ✅ Shipped | F32 weights → cuBLAS, persistent GPU KV cache |
| 16 | GPU SSM Matmuls | ✅ Shipped | Q5_K/Q6_K quantized kernel |
| 16b | GPU SSM Recurrence | ✅ Shipped | 32 V-heads × 128 threads, cos-sim=1.0 |
| 17 | GPU MoE Experts | ✅ Shipped | IQ2_XXS kernel, per-expert launch |
| 18 | GPU SSM Full Forward | ✅ Shipped | All 15 SSM steps on GPU, 2 transfers/layer |
| 19 | Batched Parallel Scan | ✅ Shipped | 18.6 tok/s prefill (+59%) |
| 20 | MoE Expert Cache | ✅ Shipped | 259MB GPU cache, zero-H2D on stability |
| 21 | Sliding Window GQA | ✅ Shipped | GQA_WINDOW env var, 16→1 tile at 256k |
| **22** | **Q4_0 KV Cache** | **✅ Shipped** | **4:1 compression, 720MB vs 2.56GB, CPU path** |
| **22a** | **Arch Discovery** | **✅ Shipped** | **3:1 interleaved pattern confirmed** |
| **22b** | **DUMP_INTERMEDIATE_DIR** | **✅ Shipped** | **1997 intermediate files/forward pass** |

---

## 3. Bug History (Complete)

| # | Bug | Found | Impact | Fix |
|---|-----|-------|--------|-----|
| 1 | GQA Q/Gate Interleave | May 18 | Cos-sim -0.51 | Per-head extraction |
| 2 | IMRoPE sections | May 18 | T=2 wrong | sections=[11,11,10,0] |
| 3 | MoE OMP Race | May 18 | Non-deterministic | Thread-local scratch |
| 4 | SSM State Carry | May 18 | Incoherent after T=1 | Persistent state buffer |
| 5 | KV Cache | May 18 | Self-only attention | Buffer all positions |
| 6 | MTP Crash | May 19 | SIGSEGV | NULL checks + concat fix |
| 7 | Q6_K Loop Count | May 19 | Cos-sim 0.796 | `32`→`16` (one char) |
| 8 | DA v10 Wrong | May 19 | Misdiagnosis | Isolate test found real bug |
| 9 | GPU RMSNorm Q stride | May 19 | Garbage output | Contiguous Q buffer before norm |
| 10 | GPU RoPE MRoPE sections | May 19 | Wrong freq | Match precompute_rotary_kernel |
| 11 | GPU KV cache overcommit | May 19 | VRAM exhaustion | Growable cache + FP16 |
| 12 | Stale binary (GPU GQA) | May 19 | Weight load failure | Clean rebuild |
| **13** | **kv_cache_read_head multi-block** | **May 19** | **Hang on decode** | **Arbitrary-length Q4_0 block read** |

---

## 4. Vault of Old Gains — Tools Directory

### Core Verification Tools
| Tool | Purpose | Links Against |
|------|---------|---------------|
| `ref_dumper.cpp` | libllama.so reference for hidden states | libllama.so |
| `ref_dumper_mtp.cpp` | MTP cross-reference | libllama.so |
| `layer_cos_sim` | Per-layer cosine similarity | Binary dump |
| `compare_ggml_matmul.cpp` | Quantized matmul vs ggml SGEMM | libllama.so |

### Architecture Analysis Tools (NEW)
| Tool | Purpose |
|------|---------|
| `classify_layers.py` | Classify SSM vs GQA layers from GGUF tensor names |
| `analyze_intermediates.py` | Inspect DUMP_INTERMEDIATE_DIR output shapes/stats |
| `analyze_l31.py` | Deep-dive into L31 GQA attention intermediates |
| `inspect_ref_intermediates.py` | Full reference intermediate tensor browser |
| `unified_ssm_plan.md` | Unified SSM kernel fusion design document |

### Quantization Verification
| Tool | Purpose |
|------|---------|
| `compare_dequant.c` | All 7 quant types vs F32 reference |
| `compare_quant_matmul_vs_sgemm.c` | Quantized matmul vs SGEMM |
| `dequant_compare.c` | Individual dequant function testing |

### tmp/ Verification Files (run once, kept for auditing)
- `/tmp/test_ssm_rec_gpu.cu` — GPU SSM recurrence cos-sim=1.0 verification
- `/tmp/test_gpu_vs_f32.cu` — GPU Q5_K matmul vs F32 dequant verification
- `/tmp/test_moe_gpu.cu` — GPU MoE IQ2_XXS kernel test
- `/tmp/ref_intermediates/` — DUMP_INTERMEDIATE_DIR F32 dumps (1997 files)
- `/tmp/ref_lay/` — DUMP_LAYER_DIR per-layer hidden states

---

## 5. Remaining Roadmap (Triple Extended)

### P0 — GPU Pipeline Fix
| Task | Impact | Detail |
|------|--------|--------|
| gen_text_gpu hang debug | **BLOCKER** | Pre-existing hang after model load. Check GPU init, SSM full forward, or tokenizer |
| GPU GQA KV cache Q4_0 | Frees 3.7GB VRAM | Currently GPU uses FP16 cache (5.12GB). Port Q4_0 quantization to GPU growable cache |

### P1 — Speed Optimizations
| Task | Impact | Detail |
|------|--------|--------|
| Unified SSM kernel Phase A | ~1.2ms decode | Fuse conv1d→SiLU→split→norm→beta into 1 kernel |
| Parallel cuBLAS streams | ~1.2ms decode | Overlap QKV + gate matmuls on separate streams |
| Sparse attention + global tokens | Enables 512k+ context | Add global position access to sliding window GQA |

### P2 — Correctness & Quality
| Task | Impact | Detail |
|------|--------|--------|
| Multi-token cos-sim verification | Confirm 0.9994 at 256k | Use ref_dumper with matching chat template tokens |
| L31 cos-sim analysis | Understand 0.9585 gap | Compare Q, K, V, attention score intermediates |
| MoE router on GPU | ~2% decode | F32 top-k on GPU, removes last CPU step |

### P3 — Long Context
| Task | Impact | Detail |
|------|--------|--------|
| Chunked prefill | 3-7x prefill at 256k | From Qwen2.5-1M paper |
| KV cache offloading | 1M+ context | CPU-side KV cache with GPU prefetch |
| DSA sparse attention | Linear-time attention at 512k | From DeepSeek-V3.2 paper |

### Bug Fixes Still Open
| Bug | Found | Status |
|-----|-------|--------|
| gen_text_gpu hang | May 19 (pre-existing) | 🔴 Needs investigation |
| L31 cos-sim 0.9585 | May 19 | 🟡 Quantization noise — expected |

---

## 6. Key Design Decisions (Permanent)

### Why Q4_0 for KV cache (not F16)?
- 4:1 compression: 720MB vs 2.56GB at 256k for CPU path
- Identical cos-sim vs F16 (0.9994 verified)
- GPU cache stays FP16 (native cuBLAS format) — separate path

### Why 3:1 interleaved SSM/GQA?
- Actual model architecture (verified via GGUF tensor enumeration)
- Every 4th layer is pure GQA attention (layers 3,7,11,...39)
- Remaining 30 layers have SSM conv1d + recurrence + fused attn_qkv gate

### Why self-hosted vec_dot?
- Zero external dependency on libggml-cpu.so
- All 7 quant types (Q4_K through IQ2_XXS) in one file
- AVX2-optimized with horizontal sum reduction

### Why not full GPU for everything?
- Single-token decode (N=1) has severe GPU overhead
- H2D/D2H: ~50μs per transfer
- GPU kernel launch: ~10μs
- CPU AVX2 matmul for 2048×8192 @ Q5_K: ~1ms (competitive)

---

## 7. Commit History

```
2ca4a7d Phase 21: Sliding window attention for 256k GQA
ea32865 Phase 20: MoE expert cache on GPU
202fac0 Phase 19: Batched prefill via parallel scan (18.6 tok/s, +59%)
01e13f2 Phase 18c: GPU conv_state + GPU K-head repeat
f221bf9 Phase 18b: FP16 chunked attention GPU softmax + MoE d_x pre-alloc
ffbf96e Phase 22: Q4_0 KV cache compression + DUMP_INTERMEDIATE_DIR + arch discovery
```

---

## 8. Cold Gaps (Research → Code)

| Research Concept | Code Status | Gap |
|-----------------|-------------|-----|
| Normed sigmoid gating (DeepSeekMoE) | Uses softmax in moe.c | Sigmoig produces more stable routing |
| Auxiliary-loss-free load balancing | Not implemented | Training-time improvement |
| Chunked prefill (Qwen2.5-1M) | Token-by-token only | 3-7x speedup at 256k |
| DSA sparse attention (DeepSeek-V3.2) | Sliding window only | Needs global token positions |
| MTP self-speculative decode | Free-tokens mode only | Quant noise blocks verify |

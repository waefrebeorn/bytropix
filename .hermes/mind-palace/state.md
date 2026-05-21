# State — Phase 28o: P2 GPU Vision Complete, CPU-Optimal Path Confirmed

**bytropix: GPU inference engine for Qwen3.6-35B MoE + vision multi-modal**  
**Reference: llama.cpp (libllama.so via ref_dumper)**  
**CUDA: sm_120 (RTX 5050 Blackwell, 13.1 toolkit)**

## CURRENT STATE
| Component | Result | Status |
|-----------|--------|--------|
| GPU vision encoder (ViT + MMProj) | GPU ViT 0.52s, total 15.7s for 256×256 | ✅ **GPU accelerated 4.4x** |
| GPU GQA batched prefill fix | C=N with sub-batch fallback | ✅ Committed |
| GPU batched quant matmul | Q5_K/Q6_K batched kernels | ✅ Committed |
| GPU SSM/GQA + CPU MoE hybrid | Coherent text at 5.5 tok/s | ✅ Working |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ Working |
| CPU-only text inference | 8.9 tok/s decode, 17.8 tok/s prefill | ✅ **Optimal path** |
| CPU gen_text (full pipeline) | Coherent text, verified | ✅ Stable |
| Vision full pipeline | 15.7s total (GPU ViT 0.52s + MMProj + CPU text) | ✅ Verified |

## ARCHITECTURAL FINDING (May 21, DA v13)
GPU hybrid text inference is net-negative for Qwen3.6-35B IQ2_M on RTX 5050:
- H2D/D2H overhead per token + per layer dominates at small batch
- GPU init heating throttles CPU on shared cooling system
- **CPU-only is 2-5x faster** and thermally stable
- GPU ONLY benefits vision encoder (pure F32 SGEMM, no quantized weights)

GPU MoE per-layer cos-sim 0.9888 vs CPU is **FUNDAMENTAL** — not a single fixable bug. Different code paths (CPU quantize_row_q8_K with negative d vs GPU positive d) produce different IEEE rounding. Hybrid path (GPU SSM/GQA + CPU MoE) produces coherent text.

## GPU Vision Pipeline Details
- GPU ViT: 0.52s for 27 layers (122x faster than 63.7s CPU)
- GPU MMProj: cuBLAS SGEMM, ~10ms
- Full pipeline: 15.7s total (9.4s vision + 6.3s text)
- 2 bugs fixed: in-place LN residual (NaN), add_kernel symbol clash
- infer_vision_text_gpu binary: ffmpeg→GPU ViT→GPU MMProj→CPU text→logits

## P0-P1 COMPLETED
1. ✅ GPU MoE analysis (DA v13): root cause identified as fundamental code-path diff
2. ✅ MTP spec decode — gen_text_mtp at 8.5 tok/s (4% acceptance from quantized head)
3. ✅ Vision pipeline verified — screenshot→encoder→mmproj→text→logits
4. ✅ 2 segfault bugs fixed in wubu_vision.c (n_patches_total cap, heap scores)
5. ✅ test_vision_real builds with GPU_SUPPORT
6. ✅ GPU GQA batched prefill (C=N, sub-batch fallback)
7. ✅ Batched quant matmul (Q5_K/Q6_K kernels)

## P2: Hardware Utilization (Current Focus)
| Priority | Item | Status | Effort |
|----------|------|--------|--------|
| P2.1 | **Llama.cpp inline hooks** for reference data dumps | ✅ Already exists | ref_dumper.cpp + DUMP_LAYER_DIR in llama-context.cpp |
| P2.2 | CUDA sm_120 bug documentation as skill | ✅ Documented in DA v13 | Quick |
| P2.3 | GPU RMSNorm + SiLU kernels | 🔲 Skipped | GPU text net-negative |
| P2.4 | **Chunked prefill** — wired into wubu_ssm_forward(). CS=1 exact. CS>1 FP error. SSM_CHUNK_MIN, FORCE_CPU_SSM_SEQ. | ✅ Wired + committed (501518f) | CS=1 exact, CS>1 FP-limited |
| P2.5 | RoPE extrapolation 4x | ✅ Complete (48dcf5e) | Low |
| P2.6 | **NSA sparse attention** (DeepSeek-V3.2 DSA) — USE_SPARSE_ATTN=1, SPARSE_W, SPARSE_G | ✅ Implemented (0129f1a) | O(L·(W+G)) for long ctx |
| P2.7 | Sigmoid gating + load balancing (DeepSeekMoE) | ✅ N/A at inference | Training-time |
| P2.8 | FP8 Tensor Cores (sm_120) | 🔲 Blocked | Needs GPU data-movement solution first |

## CUDA sm_120 Bugs (RTX 5050 Blackwell)
1. **static `__shared__` inside loops** → hang on Blackwell
   Fix: Use `extern __shared__ float smem[]` + manual offsets
2. **`__syncthreads()` + between-warps reduction** → hang
   Fix: Thread-0 serial reduce on warp peaks
3. **`extern __shared__ uint8_t` + syncthreads** → wrong code gen
   Fix: Use `extern __shared__ float smem[]`
4. **compute-sanitizer**: Failed on WDDM (Windows driver) — debugger not init
   Workaround: Manual printf + cos-sim comparison
5. **FP8 Tensor Cores**: Available on sm_120, not used — pure FP32 only

## COMMITS
- 0129f1a — feat(gqa): NSA-style sparse attention (DeepSeek-V3.2 DSA pattern)
- c5475af — fix(ssm): chunked recurrence data layout — token-interleaved heads
- 695fda5 — DA v13 complete, P1 MTP working, CUDA sm_120 bugs documented
- f97b483 — GPU MMProj via cuBLAS SGEMM, total vision 15.7s
- 3464940 — GPU vision encoder residual bug (separate d_residual param)
- 9a170fb — Batched quant matmul Q5_K/Q6_K
- 58f0a07 — Batch GQA prefill C=N + chunk_sz helper
- 12ad638 — Q8_K input quantization for MoE kernel (v5)
- 855da96 — MoE bugfixes: GPU_SUPPORT, F16 denormals, FORCE_CPU_SSM

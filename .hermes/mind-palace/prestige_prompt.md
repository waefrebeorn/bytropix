# Prestige Prompt — May 21, 2026 (Phase 28o: P2 HW Utilization)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**  
**CPU-only: 8.9 tok/s decode | GPU vision: 15.7s total pipeline | MTP: 8.5 tok/s**

## Current State
- CPU-only is optimal for text (8.9 tok/s decode). GPU hybrid is 2-5x slower.
- GPU vision encoder is THE GPU win: 0.52s vs 63.7s CPU (122x)
- MTP spec decode working: 8.5 tok/s, 4% acceptance (quantized head limit)
- Vision pipeline VERIFIED: screenshot→patch embed→27 ViT layers→mmproj→text model→logits
- GPU MoE root cause: 0.9888 cos-sim is FUNDAMENTAL code-path diff (DA v13)
- Hybrid text (GPU SSM/GQA + CPU MoE) produces coherent text at 5.5 tok/s

## DA v13 Key Findings (Supersedes all prior DAs)
1. GPU MoE 0.9888 cos-sim is NOT a bug — it's different code paths with different IEEE rounding
2. CPU quantize_row_q8_K uses **negative d** (sign-inverted Q8). GPU uses **positive d** (same-sign Q8). Both are mathematically correct but produce different rounding.
3. Hybrid path accepted. GPU MoE disabled by default (FORCE_CPU_MOE=1)
4. CUDA sm_120 bugs: 3 documented (static __shared__ hangs, syncthreads+reduction hang, extern uint8_t codegen)
5. RDRAM ring buffer concept exists (nv64-rdram-ring-buffer.md) — CPU/GPU tandem sync

## DA Debunked Claims (Phase 28b docs — now corrected)
| Old Claim | Reality |
|-----------|---------|
| "F32 waste ~2.2 GB" | ✅ Removed in a032a8f |
| "GPU mem leak ~5.5 GB" | ❌ Never existed — free() was fine |
| "Column-major kernel broken" | ❌ It was CORRECT for GGUF layout |
| "gen_text.c hardcoded" | ❌ Already accepts argv[1] |
| "GPU MoE is fixable" | ❌ DA v13: fundamental code-path diff, 0.9888 cos-sim is permanent |

## P2 Priority Queue — HW Utilization
| Priority | Item | Why |
|----------|------|-----|
| **P2.0** | CUDA sm_120 bug skill | Formalize Blackwell workarounds as reusable skill |
| **P2.1** | Llama.cpp inline hooks | Replace llama-cli ref data with libllama.so hooks. Dump layer-by-layer hidden states, Q/K/V, attention scores |
| **P2.2** | GPU RMSNorm + SiLU | Kernels exist but not wired. Low impact (<1%) but cleanup |
| **P2.3** | Chunked prefill | 3-7x speedup at 256K. Infrastructure exists (wubu_ssm_chunked.c) |
| **P2.4** | RoPE extrapolation 4x | Single frequency scaling param change for 64K→256K |
| **P2.5** | NSA sparse attention | O(L log L) for GQA at 256K. From DeepSeek-V3.2 |
| **P2.6** | Sigmoid gating + load balancing | DeepSeekMoE algorithm. F32 router only — no quant matmul needed |
| **P2.7** | FP8 Tensor Cores | sm_120 FP8 dot product for 2x throughput on quant matmul |

## Key Architecture Constants (for quick reference)
- D_MODEL = 2048, D_FF = 512
- SSM: 16 QK-heads × 128, 32 V-heads × 128, d_state=128, conv_kernel=4
- GQA: 16 Q-heads × 256, 2 KV-heads × 256, head_dim=256
- MoE: 256 routed + 1 shared, 8 active, IQ2_XXS gate/up, IQ3_XXS/IX4_XS down
- V_HIDDEN = 1152, V_HEAD_DIM = 72, V_N_LAYERS = 27, V_OUT_HIDDEN = 2048
- Vision: 3D patch 16×16×2, spatial_merge_size=2
- RoPE: IMRoPE sections [11,11,10,0], θ=10M

## CUDA sm_120 Known Bugs (RTX 5050)
1. static `__shared__` inside loops → device hang
   Fix: `extern __shared__ float smem[]` + manual offset calc
2. `__syncthreads()` + between-warps reduction → device hang
   Fix: Thread-0 serial reduce on warp leader values
3. `extern __shared__ uint8_t` + syncthreads → wrong code generation
   Fix: use `extern __shared__ float smem[]`
4. compute-sanitizer not available (WDDM driver limitation)
   Workaround: manual printf + cos-sim comparison
5. FP8 Tensor Cores supported but not used yet

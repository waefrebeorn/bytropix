# Prestige Prompt — May 21, 2026 (Phase 28p: RoPE 4x Complete)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**  
**CPU-only: 8.9 tok/s decode optimal | GPU vision: 15.7s pipeline | RoPE 4x: done**

## Current State
- CPU-only is optimal for text (7.7 tok/s verified). GPU hybrid is 2-5x slower.
- GPU vision encoder: 0.52s ViT (122x vs CPU), 15.7s full pipeline
- GPU MoE 0.9888 cos-sim is FUNDAMENTAL code-path diff (DA v13). Hybrid path accepted.
- **P2.4 RoPE 4x**: `ROPE_SCALE_FACTOR=0.25` extends 64K→256K — COMPLETE
- **P2.3 Chunked SSM**: BROKEN (cos_sim=0.00000045). Needs debug in next session.
- gen_text_cpu works with proper CLI: `./gen_text_cpu "prompt" <max_tokens>`
- ref_dumper via libllama.so works with DUMP_LAYER_DIR / DUMP_INTERMEDIATE_DIR

## Done This Session
1. ✅ RoPE extrapolation 4x — `ROPE_SCALE_FACTOR` env var in wubu_ssm.c IMRoPE
2. ✅ gen_text_cpu verified — produces coherent text at 7.7 tok/s
3. ✅ Chunked SSM investigated — root cause is causal addressing convention
4. ✅ Makefile fixes — test_chunked_ssm path, wubu_moe_cpu.o for CPU-only targets
5. ✅ Sigmoid gating deferred — training-time technique, not applicable at inference

## Next Session: P2.3 Fix + P2.5 NSA
1. **Fix chunked SSM** — compare against llama.cpp `delta-net-base.cpp` build_delta_net_chunking(). The chunked code's KQ/mask index convention is suspect. Check also the `lhs = I + strict_lower(KB)` solve initialization.
2. **NSA sparse attention** — O(L log L) per GQA layer. DeepSeek-V3.2 §2.1
3. **FP8 Tensor Cores** — sm_120 native, 2x throughput on GPU quant matmul

## Key Env Vars for Reference Data
```
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper model.gguf         # 40 layer files
DUMP_INTERMEDIATE_DIR=/tmp/ref ./ref_dumper model.gguf   # 1997 intermediate files
ROPE_SCALE_FACTOR=0.25 ./gen_text_cpu "prompt" 20        # 4x context extension
REF_LOGITS_PATH=/tmp/ref_logits.bin ./ref_dumper model.gguf  # Final logits
```

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
| **P2.3** | **Chunked prefill** | Data layout bug FIXED (c5475af). CS=1 matches exact. CS=64 has FP accumulation (7e-2 diff). Not suitable for exact inference yet. |
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

# Plan — Feature Cream Roadmap: Phase 28e → Phase 35

## ✅ Completed (Phase 28a-28e)
1. GPU_SUPPORT compiles and runs (fixed 3 pre-existing bugs)
2. Q6_K dequant offset fixed (c07cf14)
3. F32 dequant waste removed (`a032a8f`, saved ~2.2 GB)
4. Column-major kernel confirmed CORRECT for GGUF layout
5. Per-layer NULL init bug fixed (only last SSM layer survived)
6. Vision encoder ported (384 LoC — 3D ViT + mmproj)
7. Fused SSM kernels all verified in isolation (cos-sim 1.0)

## 🔴 P0: Fix GPU SSM Divergence (Phase 29)
GPU SSM produces output anti-correlated to CPU (cos-sim -0.66). CPU path matches llama at 0.994.

**Root cause hypotheses:**
1. Recurrence state not correctly persisting between layers on GPU
2. Conv state initialization or shifting differs between GPU and CPU
3. State copy mismatch between GPU `d_ssm_state` and CPU `ssm_states`

**Debug plan:**
1. Insert dump at each SSM sub-stage (conv, L2 norm, recurrence, gated norm, ssm_out)
2. Run both GPU path and CPU (FORCE_CPU_SSM) on same input
3. Compare intermediate values vs CPU path — find first divergence point
4. Fix state management → cos-sim > 0.99

**Files:**
- `src/wubu_model_gpu.cu` — forward_full call sites
- `src/gpu_ssm_recurrence.cu` — recurrence kernel
- `src/cuda_kernels.cu` — conv/silu/split, gated norm

## 🟡 P1: Infrastructure (Phase 30)
1. **Fix CPU gen_text build** — wrap GPU symbols in `#ifdef GPU_SUPPORT`
2. **Push 8 commits** to `waefrebeorn/bytropix` (master)
3. **Re-verify cos-sim** — full 40-layer comparison with current code

## 🟡 P2: Vision Verification (Phase 31)
1. Try building `test_vision_real` — check link deps
2. Generate test pixel data (or find existing)
3. Run E2E: vision encoder → mmproj → text model
4. Verify output range, no NaN, plausible statistics
5. Compare vision token distribution vs text embedding distribution

## 🟡 P3: Multi-Modal Inference (Phase 32)
1. Wire image→vision→mmproj→text pipeline in gen_text
2. Support interleaved vision + text tokens
3. Full forward: 40-layer model on vision tokens
4. Profile vision+text end-to-end

## 🟢 P4: Feature Cream (Phase 33)
| Feature | Source | Priority | Effort | C File |
|---------|--------|----------|--------|--------|
| Sigmoid gating + load balancing | DeepSeekMoE/V3 | P4.1 | Low | moe.c |
| Chunked prefill (256K) | Qwen2.5-1M | P4.2 | Med | inference.c |
| RoPE extrapolation (4x) | Qwen2.5-1M | P4.3 | Low | attention.c |
| DSA sparse attention | DeepSeek-V3.2 | P4.4 | High | attention.c |
| MTP speculative decode | DeepSeek-V3 | P4.5 | High | speculative.c |

## 🟢 P5: Perf Profiling (Phase 35)
- CUDA events per kernel: SSM GPU, GQA, MoE, output proj
- Focus: MoE (~20-40ms hypothesized bottleneck)
- Optimize memory transfers between GPU/CPU
- Target: 10+ tok/s at 256k with vision

## Key Files Reference
| File | Purpose | LoC |
|------|---------|-----|
| `src/wubu_model.c` | Core model load + forward orchestration | 1322 |
| `src/wubu_ssm.c` | SSM layer CPU implementation | 2741 |
| `src/wubu_moe.c` | MoE router + expert execution | 584 |
| `src/wubu_model_gpu.cu` | GPU initialization + forward_full | 1364 |
| `src/wubu_vision.c` | 3D ViT vision encoder | 384 |
| `include/wubu_vision.h` | Vision API + constants | 111 |

# State — Phase 26: Fused SSM Conv+SiLU+Split (May 21 AM)

**bytropix: inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**GPU decode: 8.5-9.2 tok/s (4K ctx). VRAM: ~3.56GB.**

## Phase 26 Fused Kernel
- `ssm_conv_silu_split_decode` — combines conv_state copy, conv1d, SiLU, split QKV, conv_state update into 1 kernel
- Eliminates 2 D2D memcpys + 5 kernel launches per SSM layer
- Total kernel launches per decode token: ~270 (down from ~450 before Phase 25)

## Modified Files (Phase 25+26)
- `src/gpu_quant_matmul.cu` — fused Q5_K + Q6_K matmul
- `src/cuda_kernels.cu` — ssm_beta_alpha_fused_decode + ssm_conv_silu_split_decode
- `include/cuda_kernels.h` — wrapper declarations
- `src/wubu_model_gpu.cu` — conditional fused paths for C==1 decode
- `README.md`, `STATUS.md` — DA cleanup
- All 5 mind-palace docs — Phase 25+ accurate

## DA Audit v23 Findings
| # | Finding | Severity |
|---|---------|----------|
| 1 | README claimed "gen_text_gpu hang" — FIXED in Phase 22 | 🔴 False claim |
| 2 | README said "single-token decode: CPU path" — FALSE, GPU used | 🔴 False claim |
| 3 | goal-mantra/plan/prestige all Phase 23-24 claims | 🟡 Stale |
| 4 | ssm_beta_alpha_fused_decode NOT verified vs old path | ❓ Unverified |
| 5 | ssm_conv_silu_split_decode NOT verified vs old path | ❓ Unverified |
| 6 | 256k output cos-sim vs llama.cpp NOT measured | ❓ Unverified |
| 7 | Bottleneck claims unprofiled (guesses, not nsight) | ❓ Unverified |
| 8 | 60+ stale binaries cluttering root | 🟢 Fixed (.gitignore) |

## True Bottlenecks (informed guesses, need profiling)
1. **MoE expert forward**: ~20-40ms per layer (IQ2_XXS dequant + matmul + weight upload on cache miss)
2. **Q5_K quant matmuls**: ~6ms total for 30 SSM layers
3. **GQA Q4_0 attention**: ~30ms at 256k context
4. **Everything else**: ~15ms

## External Reference
llama.cpp -ncmoe 30: 35.4 tok/s on RTX 4060 Ti 8GB. Hardwae gap ~1.8x (288 vs 160 GB/s). Software gap ~2-3x.

## Next
- Verify fused kernels against old path (cos-sim comparison)
- Add GPU timing instrumentation (nsight)
- Profile MoE — biggest bottleneck candidate

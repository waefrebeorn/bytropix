# Prestige Prompt — May 20 PM (Phase 25 — Fused Quant Matmul + SSM Beta/Alpha ✅)

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
**GPU decode: 8.5 tok/s (4K), 4.8 tok/s (256k). VRAM: ~3.56GB.**

## Phase 25: Fused Quant Matmul + SSM Beta/Alpha ✅
### What
- `quant_matmul_q5_k_kernel` / `quant_matmul_q6_k_kernel`: incremental dequant+dot, eliminates bv[256] local array spill (~15 regs vs 256)
- `ssm_beta_alpha_fused_decode`: manual dot product + sigmoid/softplus/gate for N=1, replaces 2 cuBLAS calls + 4 element-wise launches

## Phase 26 Target
- Fuse conv1d + SiLU + split + L2 norm into 1 kernel (currently 5 launches per layer)

## DA Gaps
- ❓ Fused kernel correctness NOT verified vs old path
- ❓ 256k cos-sim NOT measured
- ❓ Bottleneck claims unprofiled (guesses, not nsight)

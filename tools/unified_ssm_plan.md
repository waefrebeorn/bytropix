#!/usr/bin/env python3
"""
Unified SSM Kernel Planner
Analyze the current 15-step SSM forward and design a fused kernel.

Current GPU pipeline (wubu_model_gpu_ssm_forward_full):
15 separate operations per layer, each with own kernel launch:

1. H2D: normed[2048] → GPU
2. QKV matmul: normed → QKV (cuBLAS SGEMM 2048×8192)
3. Gate matmul: normed → gate (cuBLAS SGEMM 2048×4096)
4. Conv1d: QKV → conv_out (conv1d kernel, CONV_KERNEL=4)
5. SiLU: conv_out → silu_out (elementwise)
6. Split: silu_out → q/k/v per-head (scratch copy)
7. L2 norm: per-head → normed_q/k (per-head RMS)
8. Beta matmul: normed_q → beta_signal (dot product)
9. Gamma/Alpha matmul: normed_q → delta (element ops)
10. Recurrence: SSM step (parallel scan or sequential)
11. Gated norm: output → normed_out
12. SSM out matmul: normed_out → h_ssm (cuBLAS SGEMM 128×2048)
13. Gate SiLU/Split: gate → gate_sig (sigmoid)
14. Gated output: h_ssm * gate_sig → attn_out
15. D2H: attn_out[2048] → CPU

Each launch: ~5μs overhead
15 launches × 30 layers × 1 token = 2.25ms wasted on launch overhead
At ~9 tok/s decode, total decode time ~111ms, launch overhead ~2%
So fusion gives ~2% speedup, not 15% as previously estimated.

Bigger bottlenecks:
- cuBLAS SGEMM for QKV (2048×8192 = 16M MACs → ~80μs on RTX 5050)
- cuBLAS SGEMM for ssm_out (128×2048 = 262K MACs → ~10μs)
- SSM recurrence (128×128 = 16K per head × 16 heads × 30 layers → ~7.7M MACs → ~40μs)

Total: ~130μs per layer × 30 layers = 3.9ms per decode step
At 111ms/9tok = 12.3ms per token, ~3.9ms is SSM, leaving ~8.4ms for GQA, MoE, output proj

MVP Fusion strategy:
Phase A: Fuse Steps 4-8 (conv1d→SiLU→split→norm→beta) into single kernel
  - Eliminates ~4 kernel launches (~20μs) per layer
  - Saves intermediate scratch buffer writes (~16KB per step)
  
Phase B: Fuse Steps 1-3 (QKV+gate matmuls) by launching cuBLAS in parallel
  - QKV: 2048×8192
  - Gate: 2048×4096
  - Can use cublasGemmEx with different streams

Phase C: Fuse Steps 10-14 (recurrence→norm→out→gate) into single kernel
  - Shared memory for intermediate values
  - 16 V-heads × 128 state dim = 16KB shared memory per block
"""

# This is a planning file, print summary for terminal
print("Unified SSM Kernel Plan")
print("=" * 60)
print()
print("Current bottleneck breakdown per decode step:")
print("  QKV SGEMM (cuBLAS):         ~80μs")
print("  Gate SGEMM (cuBLAS):        ~40μs")
print("  Conv1d+SiLU+Split+Norm:     ~30μs (5 kernel launches)")
print("  SSM recurrence (GPU):       ~40μs")
print("  Gated norm + ssm_out:       ~20μs (3 kernel launches)")
print("  Gate SiLU + gated output:   ~10μs (2 kernel launches)")
print("  H2D/D2H transfer:           ~10μs")
print("  ─────────────────────────────────")
print("  Total per SSM layer:        ~230μs")
print("  30 SSM layers:              ~6.9ms")
print("  Launch overhead (15×5μs×30): ~2.25ms")
print("  Total SSM path:             ~9.15ms")
print()
print("PHASE A: Fuse conv1d→SiLU→split→norm→beta (steps 4-8)")
print("  Combine 5 kernels → 1 kernel")
print("  Save: 4×5μs launch + 4×scratch writes = ~40μs/layer")
print("  Total: 40μs × 30 = 1.2ms")
print("  Speedup: ~1% at 9 tok/s")
print()
print("PHASE B: Parallel cuBLAS streams for QKV+Gate (steps 1-3)")
print("  Save: ~40μs/layer (overlap compute)")
print("  Total: 40μs × 30 = 1.2ms")
print("  Speedup: ~1% at 9 tok/s")
print()
print("PHASE C: Fuse recurrence→norm→out→gate (steps 10-14)")
print("  Save: 4×5μs launch = 20μs/layer")
print("  Total: 20μs × 30 = 0.6ms")
print("  Speedup: ~0.5% at 9 tok/s")
print()
print("TOTAL FUSION SAVINGS: ~3ms per decode step")
print("Projected decode speed: 9.0 → 10.0 tok/s (+11%)")
print()
print("However, biggest wins come from:")
print("1) MoE on GPU (already done - expert cache)")
print("2) Output proj on GPU (already done)")
print("3) GPU SSM full forward (already done - Phase 18)")
print("4) Sliding window attention (already done - Phase 21)")
print()
print("Next big bottleneck: attention memory bandwidth at 256k context")
print("Currently: 5.12 GB FP16 KV cache × 2 (K+V) = 10.24 GB reads per GQA layer")
print("At 16384-token window: 655 MB reads per GQA layer")
print("10 GQA layers × 655 MB = 6.55 GB total reads at 256k")
print("GPU memory bandwidth (RTX 5050): ~112 GB/s")
print("Attention read time: 6.55 GB / 112 GB/s = ~58ms")

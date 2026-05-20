# Overnight Map — Phase 19: Batched Prefill Achieved

## State: Prefill 18.6 tok/s, Decode 8.7 tok/s on 8GB Laptop GPU

Phase 19 complete: **Batched prefill via wubu_cuda_ssm_parallel_scan**. For C>1 (prefill batches), the SSM recurrence uses a single parallel associative scan call instead of token-by-token recurrence. Prefill throughput jumped from 11.7 to 18.6 tok/s (+59%).

## GPU SSM Pipeline (fully optimized)
1. H2D: upload h_norm (8KB) — fundamental
2-14. GPU: quantized matmuls → cuBLAS → element-wise → conv1d → SiLU → split → L2 norm → parallel scan recurrence → SiLU(z) → gated norm → ssm_out (all on device)
15. D2H: download attn_out (8KB) — fundamental

Only 2 transfers per layer. Everything else on device.

## Remaining Targets (priority order)
1. **MoE expert cache on GPU** — cache last-8 active experts per layer, skip H2D upload when routing stable
2. **Sparse/streaming attention** — O(n·k) attention for >256k GQA
3. **MoE router on GPU** — removes CPU from forward

# Overnight Map — May 19, 2026 Late PM (Phase 18 — GPU SSM Full Forward Complete)

## State: 256k Context Running at 8.7 tok/s on 8GB Laptop GPU

Phase 18 complete: **GPU SSM conv+norm pipeline**. All 15 steps of the SSM forward now run on GPU:
1. Quantized matmuls (Q5_K QKV/Gate) → 2. cuBLAS Beta/Alpha → 3. Sigmoid/Softplus → 4. Conv input build
5. Conv1d+SiLU → 6. Conv state update → 7. Split QKV → 8. L2 norm Q/K → 9. Recurrence
10. z=SiLU(z) → 11. Gated norm → 12. SSM out proj (Q6_K) → 13. Single D2H download

Previously CPU steps (conv1d, SiLU, split, L2 norm, gated norm, ssm_out) now GPU.
From 4 H2D/D2H transfers per SSM layer down to 2 (H2D input + D2H output).

VRAM budget tighter: ~8GB with 256k context vs previous ~3.9GB (SSM scratch + F32 weights added).

## Key Architecture
- All SSM small F32 weights uploaded to GPU: beta, alpha, dt_bias, a, conv1d, norm
- 49MB SSM scratch buffer for intermediate tensors
- Full GPU path for BOTH decode (N=1) and prefill (N>1)
- Fallback to hybrid (GPU recurrence only) if full path fails

## Remaining Targets (in priority order)
1. ✅ **GPU SSM conv+norm kernel** — DONE (Phase 18)
2. **Batched prefill** — GPU for all SSM layers in one pass, use parallel scan for C>1
3. **SSM conv state on GPU** — persistent GPU conv_state, eliminate conv_state H2D/D2H
4. **Sparse/streaming attention** — O(n·k) for >256k, NSA-style or sliding window
5. **MoE router on GPU** — removes CPU from forward, ~2% improvement
6. **Unified SSM forward kernel** — fuse ALL SSM steps into single GPU kernel

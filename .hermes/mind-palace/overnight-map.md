# Overnight Map — May 19, 2026 PM (256k Context Milestone Reached)

## State: 256k Context Running at 7.8 tok/s on 8GB Laptop GPU

The core goal is achieved: 256k context decode at 7.8 tok/s on RTX 5050 (8151 MB VRAM, ~3.8 GB used).

## Key Architecture
- **FP16 KV cache** — 5GB at 256k (cublasGemmEx CUDA_R_16F → CUBLAS_COMPUTE_32F)
- **Growable cache** — starts 4096, doubles on demand
- **ATTEN_TILE=16384** — 16 tiles per 256k layer vs 64
- **GPU SSM recurrence** — all N, cos-sim=1.0
- **Smart gating** — GPU only when benefit > overhead
- **Hybrid decode** — GPU for attention + recurrence + MoE + output; CPU for SSM conv+norm

## Remaining Targets (in priority order)
1. **GPU SSM conv+norm kernel** — last CPU step per SSM layer
2. **Batched prefill** — GPU for all SSM layers in one pass
3. **SSM conv state on GPU** — persistent GPU conv_state
4. **Sparse/streaming attention** — O(n·k) for >256k
5. **MoE router on GPU** — removes CPU from forward

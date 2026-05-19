# Plan — May 19, 2026 — PHASES 0-7 DONE. NV64 DESIGN DOC WRITTEN.

## Phase 0-6: CORE + MTP ✓
## Phase 7: Hardware Saturation ✓
- GQA attn stack buf + AVX2 FMA
- AVX2 vec_dot Q4_K/Q5_K/Q6_K (256-bit)
- Prefetch next column in quantized_matmul
- Llama deps killed (no libggml-cpu.so, self-contained vec_dot)

## Phase 8: MoE Optimization [NEXT]
- AVX2 IQ2_XXS vec_dot (256-bit grid lookup)
- AVX2 IQ3_XXS vec_dot (256-bit with sign handling)
- OpenMP task-based expert scheduling (8 experts per layer)
- Expert prefetch: predict next layer's experts, preload to L2

## Phase 9: NV64 RDRAM Ring Buffer
- Implement ring_slot_t[64] with atomic head/tail
- Prefetch agent thread (graduated T2→T1→T0)
- Arbiter/scheduler with token-tick barriers
- CPU/GPU tandem: split layers 0-19/20-39
- CUDA kernels for layers 20-39

## Phase 10: Distributed Inference
- Ring slot = machine[i % N]
- Distributed arbiter via token passing
- Multi-node weight prefetch
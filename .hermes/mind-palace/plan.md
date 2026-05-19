# Plan — May 19, 2026 (02:45) — TRIPLE DA UPDATED

## Phase 0-7: DONE (Core → MTP → HW Saturation)
- GQA attn stack buf + AVX2 FMA
- AVX2 vec_dot Q4_K/Q5_K/Q6_K (256-bit)
- Prefetch next column
- Llama deps killed (self-contained vec_dot)
- NV64 RDRAM design doc written

## Phase 8: MoE Optimization [NEXT — P0]
- AVX2 IQ2_XXS vec_dot (256-bit grid lookup from llama.cpp ggml-quants.c)
- AVX2 IQ3_XXS vec_dot (256-bit with sign handling)
- OpenMP task-based expert scheduling (8 experts across threads)
- Expert prefetch: predict next layer's experts, preload to L2

## Phase 9: MoE Router Upgrade [P1]
- Normalized sigmoid gating (replace softmax over 256 experts)
- Load-balancing bias support (for training)

## Phase 10: NV64 RDRAM Ring Buffer [P1]
- ring_slot_t[64] with atomic head/tail
- Prefetch agent thread (graduated T2→T1→T0)
- Arbiter/scheduler with token-tick barriers
- CPU/GPU tandem: split layers 0-19/20-39

## Phase 11: Distributed Inference [P2]
- Ring slot = machine[i % N], token-passing arbiter
# WuBuText AI — Plan (May 15)

## Purpose
Post-S7 priorities. All 7 original streams complete. Now: math games, manifold, optimizations.

## Priority Queue

P0 — **Fix pre-existing NaN in model logits** (~0.5%). Blocks reliable training eval.

P0 — **Fix CPU GQA RMSNorm dim** (d=4096 with weight[256]). Wrong for i>=256.

P1 — **Wire GPU vision** (cuda_vision.cu 217ms vs CPU 74s) into `infer_vision_text`.

P1 — **Implement RSGD optimizer** for Poincaré params (step in tangent, project back).

P2 — **Poincaré GQA**: Replace softmax with hyperbolic distance attention.

P2 — **Data pipeline**: Tokenize corpus → .bin for C training loop.

P3 — **Nested SSM**: Product of K Poincaré balls with K curvatures.

P3 — **Nested MoE**: Poincaré distance routing + 2-level hierarchy.

P3 — **TST**: Token Superposition Training (bag s=8, MCE loss).

P4 — **CUDA kernels**: SSM scan (parallel associative), MoE dispatch (grouped GEMM).

P4 — **Moondream3 port**: Weight dump + C ViT + hyperbolic graft.

# WuBuText AI — Project Overview

## What We're Building

**WuBuText AI** is a pure C + CUDA implementation of a Qwen3.6-35B-A3B-compatible language model, replacing standard Euclidean computation with WuBu nested hyperbolic geometry. The project implements the full model stack from scratch — GGUF weight extraction, SSM recurrence, GQA attention, Möbius gyration operations, and CUDA kernel acceleration — on consumer hardware (RTX 5050, 8 GB VRAM).

The model spec: 40 layers (30 SSM + 10 GQA, 3:1 repeating), 2048 hidden dimension, 248,320-token vocabulary, and 262K native context. MoE dispatch (256 experts, 8 active + 1 shared) is planned for Phase 4.

## Why Geometry Matters

Standard transformer attention scales **O(N²)** with sequence length — a fundamental bottleneck for long-context reasoning. WuBu nested hyperbolic geometry aims to compress this to **O(N)** by operating in Poincaré ball space, where hierarchical structure is captured geometrically rather than through pairwise quadratic attention:

- **Euclidean embeddings** are mapped to the **Poincaré ball** via exponential map (radius R = 0.956)
- **Möbius gyration** replaces dot-product attention with hyperbolic operations in tangent space
- **Nested hierarchy levels** (planned for Phase 4+) model token relationships at multiple scales without explicit pairwise comparisons

The central thesis: *geometry can replace computation*. If relationships between tokens are encoded in their positions on a hyperbolic manifold, the model doesn't need to compute N² attention scores to find them.

## Current State: Phase 3 — Training Loop

The project follows a 6-phase roadmap. As of May 2026:

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | GGUF Tensor Layout | ✅ Complete |
| 1 | Embedding Graft (GGUF → Poincaré) | ✅ Complete |
| 2 | Attention Port (SSM + GQA in C/CUDA) | ✅ Complete |
| 2.5 | GPU Verification | ✅ Complete |
| **3** | **Training Loop** | **🔄 In Progress** |
| 4 | MoE Port (256 experts) | ⏳ Future |
| 5 | Vision Port (27-layer 3D ViT) | ⏳ Future |
| 6 | CUDA Optimization | ⏳ Ongoing |

### Key Results (Preliminary — Phase 2.5 Benchmarks)

- **GPU forward pass**: 9.53 tokens/second on RTX 5050 (all 40 layers)
- **Speedup vs CPU baseline**: 47.83× (419.85 ms per forward pass)
- **Embedding quality**: 95% nearest-neighbor preservation after Euclidean → Poincaré mapping (preliminary)
- **Embedding extraction**: 73 zero-norm special tokens correctly positioned at origin

*These are early results on an in-progress implementation. Performance is expected to improve with Phase 6 CUDA kernel optimization.*

### Phase 3: Token-Superposition Training (TST)

The training loop uses **Token-Superposition Training** (TST), a drop-in pre-training acceleration method from Peng, Gigant, and Quesnelle (Nous Research, arXiv:2605.06546). TST works in two phases:

1. **Superposition Phase** (~25% of steps): Bag `s` contiguous tokens, average their embeddings into one "s-token", run the forward pass on `1/s` the sequence length, and train with multi-hot cross-entropy loss across all targets in the bag
2. **Recovery Phase** (~75% of steps): Remove superposition, train with standard next-token CE loss, carrying over weights from Phase A

The method was validated by its authors on models up to 10B A1B MoE, achieving up to **2.5× speedup at equal loss**. Our target for this implementation is 2×+ on the 3B-active-parameter model.

**Current blocker**: BBPE tokenizer merge lookup is O(N²) — a hash-table rewrite is in progress before training can begin.

## Hardware

All development and benchmarking runs on a single **NVIDIA RTX 5050** (Ada Lovelace, compute 8.9, 8 GB VRAM, ~72 TFLOPS f16). This constraint drives architectural decisions: model weights must be quantized (Q5_K/Q8_K for storage, f16 for compute), optimizer states are CPU-offloaded, and the training micro-batch is limited to batch size 2 with gradient accumulation.

## Team & Research Context

This work builds on research from **Nous Research** and collaborators:

- **Token-Superposition Training** (Peng, Gigant, Quesnelle, May 2026) — the training methodology for Phase 3
- **WuBu Nested Hyperbolic Geometry** — original theoretical framework developed across six research phases, with four formal proofs verified in Lean 4
- **Hamilton Encoder** — earlier research phase exploring hyperbolic encoders for geometric deep learning

The project is a research prototype, not a production system. All results should be understood as **preliminary, in-progress findings** from a single-developer effort on consumer hardware.

## Quick Links

| Resource | Location |
|----------|----------|
| Full Architecture | `3-architecture.md` |
| Implementation Status | `4-implementation-status.md` |
| Research Vault Tour | `2-research-vault.md` |
| Future Roadmap | `7-future-roadmap.md` |
| TST Paper Reference | `../references/TST_TOKEN_SUPERPOSITION.md` |

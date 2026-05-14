# WuBuText AI — Project Overview

## What We're Building

**WuBuText AI** is a pure C + CUDA implementation of a Qwen3.6-35B-A3B-compatible language model. The project implements the full model stack from scratch — GGUF weight extraction, SSM recurrence, GQA attention, MoE routing, and CUDA kernel acceleration — on consumer hardware (RTX 5050, 6.4 GB VRAM).

The model spec: 40 layers (30 SSM + 10 GQA, 3:1 repeating), 2048 hidden dimension, 248,320-token vocabulary, and 262K native context. MoE dispatch (256 experts, 8 active + 1 shared) is implemented and verified on CPU.

## Why Geometry Matters

Standard transformer attention scales **O(N²)** with sequence length. WuBu nested hyperbolic geometry aims to compress this to **O(N)** by operating in Poincaré ball space, where hierarchical structure is captured geometrically rather than through pairwise quadratic attention:

- **Euclidean embeddings** are mapped to the **Poincaré ball** via exponential map (radius R = 0.956)
- **Möbius gyration** replaces dot-product attention with hyperbolic operations in tangent space
- The central thesis: *geometry can replace computation*

## Current State: Phase 3 — Training Loop

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | GGUF Tensor Layout | ✅ Complete |
| 1 | Embedding Graft (GGUF → Poincaré) | ✅ Complete |
| 2 | Attention Port (SSM + GQA in C) | ✅ CPU complete — GPU broken |
| 2.5 | GPU Benchmarking | ⚠️ Verified stalled — GPU weight loading bug |
| **3** | **Training Loop** | **🔄 In Progress** |
| 4 | MoE Port (256 experts) | ✅ CPU forward done — training pending |

**Key finding (May 13 DA Audit):** CPU path works correctly (CE loss 12.66). GPU weight loading in bench.c produces zeros — undiscovered until this audit. root_cause = Q4_K dequant fix was applied to wubu_model.c but NOT fully propagated to bench.c's GPU loading path.

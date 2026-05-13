# Bytropix Future Roadmap

> Plans and explorations, not promises. Timelines are aspirational; scope
> adjusts based on research outcomes, compute availability, and real-world
> deployment feedback.

---

## Phase 4 — MoE Port (256-Expert Mixture)

| Component | Direction |
|-----------|-----------|
| Expert count | 256 experts total |
| Active per token | 8 active + 1 shared expert |
| Quantization | IQ2_XS (active) / IQ1_S (shared) — aggressive compression |
| Router | Hyperbolic-distance-based routing in Poincaré ball |
| Load balancing | Auxiliary-loss-free — reliance on hyperbolic geometry to spread load naturally |

**Planned milestones**:
- Replace dense FFN with MoE feedforward block, validate loss match
- Implement hyperbolic router; compare against cosine-similarity baseline
- Integrate IQ2_XS / IQ1_S quantizers; measure perplexity impact
- Profile expert utilization across training; tune router temperature

---

## Phase 5 — Vision Port (3D ViT + Temporal Fusion)

| Component | Direction |
|-----------|-----------|
| Encoder | 27-layer 3D Vision Transformer |
| Patch strategy | Temporal patch embedding for video / multi-frame input |
| Spatial compression | Spatial merge layer to reduce sequence length after early layers |
| Position encoding | Multi-resolution Rotary Position Embedding (MRoPE) |
| Modality alignment | Contrastive alignment head between vision encoder and hyperbolic text backbone |

**Planned milestones**:
- Build 3D ViT in isolation; verify on ImageNet / Video-MME
- Add temporal patch + spatial merge; measure throughput vs. accuracy trade-off
- Train vision-language contrastive objective; evaluate zero-shot retrieval
- Merge checkpoint with Phase 4 MoE backbone; test joint forward pass

---

## Phase 6 — CUDA Kernel Optimization

| Kernel | Target speedup | Approach |
|--------|----------------|----------|
| MatMul (FFN / MoE) | 1.5–2× | cuBLAS kernel fusion + expert-grouped GEMM |
| SSM scan (S4 / Mamba-style) | 2–3× | Custom CUDA associative scan, register-tiled |
| MoE dispatch | 3–5× | Expert-aware scatter/gather with warp-level routing |
| FlashAttention | 1.5–2× | Fwd+bwd attention kernels for long-sequence hyperbolic attention |

**Planned milestones**:
- Profile current PyTorch implementation bottlenecks (nsys / ncu)
- Implement and unit-test each CUDA kernel individually
- Integrate into training loop; benchmark tokens/second and memory bandwidth
- Release as standalone `bytropix-kernels` package with Python bindings

---

## Long-term Research Directions

| Direction | Why | How (current thinking) |
|-----------|-----|------------------------|
| **Nested hyperbolic hierarchy** | 4-level Russian-doll geometry — token → phrase → sentence → document | Multiple Poincaré balls at increasing radii; hierarchical contrastive loss |
| **1M context window** | Long-document / multi-hour video reasoning | Ring attention + hyperbolic positional bias + memory offloading |
| **100+ tok/s inference** | Real-time interactive use | MoE sparsity + INT4 quantization + speculative decoding + kernel fusion |
| **Cross-modal hyperbolic space** | Unify text, image, audio in a single geometric representation | Train projection heads into a shared hyperboloid; evaluate on retrieval + QA |
| **Self-supervised hyperbolic pretraining** | Remove dependence on Euclidean foundation models | Manifold-aware masked modeling; test on mathematical / scientific text |

---

## Design Principles (ongoing)

1. **Measure first** — every architectural change gets a perplexity / throughput / memory benchmark before shipping.
2. **Incremental merge** — each phase lands on `main` independently; no multi-phase mega-PRs.
3. **Reproducibility** — every experiment hash-pinned (config, data split, random seed) and logged.
4. **Quantization-aware design** — train with simulated quantization from day one of each new component.

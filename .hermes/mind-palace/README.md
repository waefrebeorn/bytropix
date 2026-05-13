# WuBuText AI — Mind Palace

Central planning and knowledge structure for the WuBuText AI project.
Building end-to-end C training/fine-tuning/inference using WuBu nested hyperbolic math + DeepSeek/Qwen research.

**Key documents (read first after tier1-core):**
- [Math Optimization Roadmap](core/math/wubu_math_optimization_roadmap.md) — Euclidean→Poincaré→Nested optimization path for every component
- [Master Implementation Plan v2](plans/master_impl_plan_v2.md) — All 6 phases, step-by-step, with dependency graph
- [Devil's Advocate v2](plans/devils_advocate_v2.md) — 7 risks with mitigations, updated after proofread

## Structure

```
.hermes/
├── mind-palace/              ← YOU ARE HERE — central navigation
│   ├── README.md              ← This file
│   ├── tier1-core/            ← Foundational knowledge (read first)
│   ├── tier2-research/        ← Research references
│   ├── tier3-impl/            ← Implementation plans
│   ├── tier4-validation/      ← Testing and benchmarks
│   └── timeline/              ← Scheduling and progress
├── plans/                     ← Generated plan documents
├── research/                  ← Paper downloads and references
├── skills/                    ← Reusable Hermes Agent skills
└── vault/                     ← Bytropix vault index
```

## How To Use This

**Navigation rule:** Each directory has a `README.md` that links to sub-items.
Reading order: Tier 1 → Tier 2 → Tier 3 → Tier 4.

**Tier 1 — Core Context** (what we're building and why)
1. `1-wubu-theory/` — WuBu nesting physics, hyperbolic geometry, K-theory
2. `2-arch-reference/` — Reference architectures (DeepSeek V2/V3 MLA, Qwen3.5+ Gated DeltaNet)
3. `3-baseline-c/` — Current C training state, existing code, what works

**Tier 2 — Research** (what we're stealing from)
4. `4-deepseek/` — DeepSeek papers: MLA, MoE, DSA sparse attention, R1 RL
5. `5-qwen/` — Qwen papers: Gated DeltaNet, hybrid arch, embedding extraction
6. `6-fast-attention/` — Fast attention: FlashAttention, NSA, Gated Sparse, MISA, Mamba
7. `7-hyperbolic/` — Hyperbolic neural nets: Poincaré, Möbius, Fully Hyperbolic

**Tier 3 — Implementation** (how we build it)
8. `8-embedding-graft/` — Extract Qwen embeddings, map to Poincaré, test grafting
9. `9-attention-port/` — Port Gated DeltaNet to C, add hyperbolic gyration
10. `10-training-loop/` — Training loop, data pipeline, loss functions, optimizers
11. `11-moe-port/` — MoE routing with wubu nested geometry
12. `12-vision/` — Vision encoder port for WuBuVision

**Tier 4 — Validation** (proving it works)
13. `13-benchmarks/` — minGPT functional comparison, perplexity, convergence
14. `14-debug/` — Known issues, debugging workflows, FAIL_LOG

## Current Status

- Phase: **Research + Architecture Design**
- Target model: **Qwen3.6-35B-A3B** (35B total, 3B active, hidden=2048)
- Platform: RTX 5050 6.4GB VRAM, pure C + CUDA
- Embeddings: Ready to extract from GGUF

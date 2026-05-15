# WuBuText AI — Presentation Layer

**Purpose:** Curated navigation panel for presenting the WuBuText AI project — its research history, architecture, implementation progress, and future roadmap.

This folder is the **public-facing presentation layer** of the `.hermes/` vault. It organizes all project resources for quick browsing by visitors, reviewers, and future sessions.

---

## Navigation

```
presentation/
├── README.md               ← This file (presentation overview + upkeep plan)
├── 1-project-overview.md     → One-page summary: what, why, how
├── 2-research-vault.md       → Guided tour of bytropix research (all 6 phases + variants)
├── 3-architecture.md         → System architecture: GGUF → C → CUDA → WuBu math
├── 4-implementation-status.md → Current implementation state per phase
├── 5-diagrams.md             → Index of all SVG diagrams with descriptions
├── 6-references.md           → Paper references with annotations
├── 7-future-roadmap.md       → Phase 4-6 plans + long-term vision
└── UPKEEP.md                 → Maintenance instructions for this presentation layer
```

---

## Quick Links to All Resources

### Diagrams (updated May 13, 2026)

| Diagram | Location | What It Shows |
|---------|----------|---------------|
| Phase Roadmap | `../../DIAGRAMS/phase-roadmap.svg` | Full project timeline + key metrics |
| GGUF Pipeline | `../../DIAGRAMS/gguf-rip-pipeline.svg` | How Qwen3.6 weights become WuBu embeddings |
| llama.cpp Clone | `../../DIAGRAMS/llamacpp-clone-infrastructure.svg` | Fork study + extraction + benchmark workflow |
| WuBu Math | `../../DIAGRAMS/wubu-math-pipeline.svg` | Euclidean→Poincaré→Möbius→Nested pipeline |
| Nesting Arch | `../../DIAGRAMS/wubu-nesting-architecture.svg` | Original 4-level hyperbolic architecture |
| Hamilton Encoder | `../../DIAGRAMS/hamilton-encoder-pipeline.svg` | Hamilton encoder pipeline |
| Research Timeline | `../../DIAGRAMS/research-timeline.svg` | Complete discovery timeline (Aug 2025–May 2026) |

### Mind Palace (Project Planning)

| Document | Location | Content |
|----------|----------|---------|
| Master Plan v2 | `../mind-palace/plans/master_impl_plan_v2.md` | All 6 phases, step-by-step, dependency graph |
| Training Loop | `../mind-palace/tier3-impl/10-training-loop/README.md` | Phase 3 detail with TST method |
| Attention Port | `../mind-palace/tier3-impl/9-attention-port/README.md` | SSM recurrence + GQA tensor layout analysis |
| Embedding Graft | `../mind-palace/tier3-impl/8-embedding-graft/plan.md` | GGUF extraction + Poincaré mapping |
| Devil's Advocate | `../plans/2026-05-12-devil-advocate-roadmap.md` | 7 risks with mitigations |
| Fresh Start | `../mind-palace/fresh_start_prompt.md` | Session initialization prompt |

### Research References

| Reference | Location | Description |
|-----------|----------|-------------|
| TST Paper | `../references/TST_TOKEN_SUPERPOSITION.md` | Token-Superposition Training (2605.06546) |
| TST PDF | `../references/2605.06546_token_superposition.pdf` | Full paper (1.8MB) |
| Qwen3.6 Arch | `../research/papers/Qwen/Qwen3.6-35B_Arch_Reference.md` | Model architecture reference |
| Qwen3.5 Arch | `../research/papers/Qwen/Qwen3.5-9B_Arch_Reference.md` | Smaller sibling architecture |
| DeepSeek V3 | `../research/papers/` | MLA, MoE, DSA references |

### Source Code Map

|| Component | Files | Status |
||-----------|-------|--------|
|| GGUF Reader | `include/gguf_reader.h`, `src/gguf_reader.c` | ✅ Done |
|| SSM Forward | `include/wubu_ssm.h`, `src/wubu_ssm.c` | ✅ Done |
|| GQA Forward | `include/wubu_ssm.h`, `src/wubu_ssm.c` | ✅ Done |
|| Möbius Ops | `include/wubu_mobius.h`, `src/wubu_mobius.c` | ✅ Done |
|| Model Assembly | `include/wubu_model.h`, `src/wubu_model.c` | ✅ Done |
|| CUDA Kernels | `include/cuda_kernels.h`, `src/cuda_kernels.cu` | ✅ Done |
|| Bench GPU | `include/bench.h`, `src/bench.c`, `tools/bench_e2e.c` | ✅ Fixed |
|| GPU Test | `tools/test_gpu.c` | ✅ Match verified |
|| Tokenizer | `include/wubu_tokenizer.h`, `src/wubu_tokenizer.c` | ✅ Working |
|| RSGD Optimizer | `include/rsgd.h`, `src/rsgd.c` | ✅ Done |
|| Poincaré GQA | `include/wubu_poincare_gqa.h`, `src/wubu_poincare_gqa.c` | ✅ Done |
|| Nested SSM | `include/wubu_nested_ssm.h`, `src/wubu_nested_ssm.c` | ✅ Done |
|| TST Training | `include/wubu_tst.h`, `src/wubu_tst.c` | ✅ Done |
|| Nested MoE | `include/wubu_moe_hyperbolic.h`, `src/wubu_moe_hyperbolic.c` | ✅ Done |
|| Vision GPU | `include/cuda_vision.h`, `src/cuda_vision.cu` | ✅ Done |
|| Moondream3 | `include/wubu_vision_moondream.h`, `src/wubu_vision_moondream.c` | ✅ Stub |
|| Training GPU | `tools/train_gpu.c` | ✅ CE=12.42 |
|| Data Pipeline | `tools/td.c` + scripts | ✅ 1.07M tokens |

### Bytropix Research Vault (Earlier Work)

| Vault | Location | Description |
|-------|----------|-------------|
| Theory | `../vault/theory/README.md` | Physics, philosophy, papers |
| Encoders | `../vault/encoders/README.md` | 6 research phases, hash-mind, Hamilton |
| Attention | `../vault/attention/README.md` | 4 attention variant analyses |
| Audio | `../vault/audio/README.md` | WubuSynth galactic core |
| Diffusion | `../vault/diffusion/README.md` | HGA UNet, funnel diffusion |
| Optimizers | `../vault/optimizers/README.md` | Q-Controller, PID |
| Phase 3 | `../vault/phase3/README.md` | Generative pipeline analysis |
| Hamilton | `../vault/hamilton/README.md` | Hamilton encoder documentation |
| Hash-Mind | `../vault/hash-mind/README.md` | WuBuMind V1-V7.1 JAX analysis |
| C-Training | `../vault/c-training/README.md` | C port experiments |
| DraftPY | `../vault/draftPY/README.md` | Experimental Python scripts |
| Lean Proofs | `../vault/lean-proofs/README.md` | Formal verification |

---

## Upkeep Plan

This presentation layer must stay in sync with the project. See `UPKEEP.md` for maintenance instructions.

### Update Triggers

The presentation layer should be updated when:
1. A new phase completes (status changes in main README)
2. A new diagram is added
3. A major paper is integrated
4. Architecture decisions change
5. Performance metrics are updated

### Responsibility

Keep-it-fresh (KIF) during regular session work. If you notice a doc is stale, fix it immediately rather than deferring.

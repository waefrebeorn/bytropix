# WuBuText AI — Presentation Layer

**Purpose:** Curated navigation panel for presenting the WuBuText AI project — its research history, architecture, implementation progress, and future roadmap.

This folder is the **public-facing presentation layer** of the `.hermes/` vault. It organizes all project resources for quick browsing by visitors, reviewers, and future sessions.

---

## Navigation

```
presentation/
├── README.md               ← This file (presentation overview)
├── 1-project-overview.md     → One-page summary: what, why, how
├── 2-research-vault.md       → Guided tour of bytropix research (all phases + variants)
├── 3-architecture.md         → System architecture: GGUF → C → CUDA → WuBu math
├── 4-implementation-status.md → Current implementation state per phase
├── 5-diagrams.md             → Index of all SVG diagrams (10 total)
├── 6-references.md           → Paper references with annotations
└── 7-future-roadmap.md       → P0-P3 plans + vault porting + tailslayer
```

---

## Quick Links to All Resources

### Diagrams (May 15 PM v6 — 10 SVGs, 3 new)

| Diagram | Location | What It Shows |
|---------|----------|---------------|
| Phase Roadmap | `../../DIAGRAMS/phase-roadmap.svg` | Full project timeline + key metrics |
| **Training Pipeline** | `../../DIAGRAMS/training-pipeline.svg` | **NEW** — 11s/step training flow: GGUF→Dequant→GPU→MoE→Proj→Loss→Flags |
| **Tailslayer Pattern** | `../../DIAGRAMS/tailslayer-pattern.svg` | **NEW** — Hedged-read → spec-decode analogy |
| Paper Audit | `DIAGRAMS/paper-audit.svg` | **NEW** — 14 Qwen3.6 params vs C implementation. Updated May 15: CONV_DIM resolved (not a bug), RoPE+rotary_dim confirmed added. |
| GGUF Pipeline | `../../DIAGRAMS/gguf-rip-pipeline.svg` | How Qwen3.6 weights become WuBu embeddings |
| llama.cpp Clone | `../../DIAGRAMS/llamacpp-clone-infrastructure.svg` | Fork study + extraction + benchmark workflow |
| WuBu Math | `../../DIAGRAMS/wubu-math-pipeline.svg` | Euclidean→Poincaré→Möbius→Nested pipeline |
| Nesting Arch | `../../DIAGRAMS/wubu-nesting-architecture.svg` | Original 4-level hyperbolic architecture |
| Hamilton Encoder | `../../DIAGRAMS/hamilton-encoder-pipeline.svg` | Hamilton encoder pipeline |
| Research Timeline | `../../DIAGRAMS/research-timeline.svg` | Complete discovery timeline (Aug 2025–May 2026) |

### Mind Palace (Project Planning)

| Document | Location | Content |
|----------|----------|---------|
| Goal Mantra | `../mind-palace/goal-mantra.md` | Prestige paste — full state, commands, priorities (v6) |
| Plan | `../mind-palace/plan.md` | Full priority queue P0-P3 + vault + tailslayer (v6) |
| State | `../mind-palace/state.md` | Binary dashboard + paper audit + tailslayer (v6) |
| Fresh Start | `../mind-palace/fresh_start_prompt.md` | Session initialization prompt (v6) |

### Research References

| Reference | Location | Description |
|-----------|----------|-------------|
| TST Paper | `../references/TST_TOKEN_SUPERPOSITION.md` | Token-Superposition Training (2605.06546) |
| Qwen3.6 Arch | `../research/papers/Qwen/Qwen3.6-35B_Arch_Reference.md` | Model architecture reference |
| Tailslayer Notes | `../../THEORY/papers/tailslayer-notes.md` | Hedged-read spec-decode analysis |

### Vault Entries (14 total, May 15: +tailslayer)

| Vault | Location | Priority |
|-------|----------|----------|
| Tailslayer | `../vault/tailslayer/README.md` | **P2 — NEW** |
| Attention | `../vault/attention/README.md` | P2 — sparse attention port |
| Optimizers | `../vault/optimizers/README.md` | P2 — Q-Controller + PID |
| Hamilton | `../vault/hamilton/README.md` | P2 — geodesic encoder CUDA |
| Theory | `../vault/theory/README.md` | Reference |
| Encoders | `../vault/encoders/README.md` | Research |
| Audio | `../vault/audio/README.md` | Standalone |
| Diffusion | `../vault/diffusion/README.md` | Low priority |
| Phase 3 | `../vault/phase3/README.md` | Low priority |
| Hash-Mind | `../vault/hash-mind/README.md` | Reference |
| C-Training | `../vault/c-training/README.md` | Reference |
| DraftPY | `../vault/draftPY/README.md` | Idea source |
| Lean Proofs | `../vault/lean-proofs/README.md` | Low priority |
| Math | `../vault/math/README.md` | Empty (reference) |

---

## Upkeep

All docs updated to v6 (May 15 2026). Next update trigger: new phase completion, new diagram, or significant metric change.

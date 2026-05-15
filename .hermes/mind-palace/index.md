# WuBuText AI — Mind Palace Index (May 15 PM v6 — COMPREHENSIVE)

## Walkway (Read in Order)

| # | File | Purpose | Last Updated |
|---|------|---------|-------------|
| 1 | `goal-mantra.md` | 🏆 Prestige paste — full state, commands, priorities | **v6** |
| 2 | `state.md` | 📊 Binary dashboard, metrics, known issues, paper audit, tailslayer | **v6** |
| 3 | `plan.md` | 🗺️ Priority queue P0-P3, vault porting, paper findings, tailslayer | **v6** |
| 4 | `entry.md` | 🔧 Build commands, hardware spec, external repos | **v6** |
| 5 | `testing.md` | 🧪 Test protocol, binary list, known issues | **v6** |
| 6 | `project.md` | 🎯 Mission, phases, achievements, remaining | **v6** |
| 7 | `overnight-map.md` | 🌙 Autonomous session navigation | **v6** |
| 8 | `fresh_start_prompt.md` | 🚀 Boot prompt for new sessions | **v6** |

## References

| File | Purpose |
|------|---------|
| `plans/master_impl_plan_v2.md` | Original 6-phase implementation plan (historical) |
| `plans/devils_advocate_v4.md` | Devil's advocate audit v4 |
| `tier1-core/` | WuBu theory, architecture |
| `tier3-impl/10-training-loop/` | Training loop details |
| `tier3-impl/11-moe-port/` | MoE port details |
| `tier3-impl/12-vision/` | Vision port plans |

## Vaults (14 entries, May 15: +tailslayer)

| Vault | Description | Code Status | Port Priority |
|-------|-------------|-------------|--------------|
| `vault/attention/` | Sparse attention (O(n·k) linear) | PyTorch | **P2 — highest ROI** |
| `vault/tailslayer/` | Hedged reads → spec-decode CUDA kernel | C++ template + tREFI probe | **P2 — new** |
| `vault/optimizers/` | Q-Controller + PID meta-learning | JAX | P2 — low effort |
| `vault/hamilton/` | Geodesic encoder | ✅ CUDA in llama.cpp fork | P2 — medium effort |
| `vault/c-training/` | Pure C transformer training | ✅ Running, 4000 steps/s | Reference for patterns |
| `vault/hash-mind/` | WuBuMind JAX V1-V7.1 | JAX + C port | Study for ideas |
| `vault/theory/` | WuBu nesting physics & philosophy | Reference only | — |
| `vault/encoders/` | Symmetric AE → QAE → generative | Python | Research |
| `vault/diffusion/` | HGA-UNet, funnel diffusion | Python | Low priority |
| `vault/phase3/` | Text-to-image pipeline | Python, 66K lines | Low priority |
| `vault/audio/` | WubuSynth audio synthesis | Python | Standalone |
| `vault/draftPY/` | 40+ experimental scripts | Python | Idea source |
| `vault/lean-proofs/` | Lean 4 formal proofs | 4 incomplete | Low priority |
| `vault/math/` | Math directory (empty, reference) | — | — |
| `research/papers/` | Qwen architecture references (32 files) | — | Audit done May 15 |

## Diagrams

| Diagram | Description |
|---------|-------------|
| `DIAGRAMS/gguf-rip-pipeline.svg` | GGUF → C/CUDA pipeline |
| `DIAGRAMS/phase-roadmap.svg` | Phase roadmap (100% complete, v6) |
| `DIAGRAMS/wubu-math-pipeline.svg` | Euclidean → Poincaré → Möbius pipeline |
| `DIAGRAMS/research-timeline.svg` | August 2025 → May 2026 research arc |
| `DIAGRAMS/llamacpp-clone-infrastructure.svg` | llama.cpp fork infrastructure |
| `DIAGRAMS/hamilton-encoder-pipeline.svg` | Hamilton encoder pipeline |
| `DIAGRAMS/wubu-nesting-architecture.svg` | WuBu nesting architecture |

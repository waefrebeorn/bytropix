# WuBuText AI — Mind Palace Index (May 16 v7 — HONEST)

## Walkway (Read in Order)

| # | File | Purpose | Last Updated |
|---|------|---------|-------------|
| 1 | `plans/devils_advocate_v5.md` | 🔍 Meta audit — all goals, math, models | **v5 NEW** |
| 2 | `state.md` | 📊 Binary dashboard — HONEST real status | **v12** |
| 3 | `goal-mantra.md` | 🏆 Prestige paste — commands, priorities | **v12 HONEST** |
| 4 | `plan.md` | 🗺️ Priority queue P0-P3 | **v11** |
| 5 | `entry.md` | 🔧 Build commands, hardware spec, API server | **v7 HONEST** |
| 6 | `testing.md` | 🧪 Test protocol, known limitations | **v7** |
| 7 | `project.md` | 🎯 Mission, phases, remaining work | **v7 HONEST** |
| 8 | `overnight-map.md` | 🌙 Autonomous session navigation | **v7 HONEST** |
| 9 | `fresh_start_prompt.md` | 🚀 Boot prompt for new sessions | **v7 HONEST** |

## HARD TRUTH
- Inference is BROKEN. Only 2/8 binaries verified correct.
- 6/15 math components are forward-only (no gradient flow).
- Reference: `~/llama.cpp/build/bin/llama-cli`

## References

| File | Purpose |
|------|---------|
| `plans/devils_advocate_v4.md` | Devil's advocate audit v4 |
| `tier1-core/` | WuBu theory, architecture |
| `tier3-impl/` | Implementation details |

## Vaults (14 entries)

| Vault | Description | Code Status | Priority |
|-------|-------------|-------------|----------|
| `vault/attention/` | Sparse attention (O(n·k) linear) | PyTorch | P2 |
| `vault/tailslayer/` | Hedged reads → spec-decode CUDA kernel | C++ template | P2 |
| `vault/optimizers/` | Q-Controller + PID meta-learning | JAX | P2 |
| `vault/hamilton/` | Geodesic encoder | ✅ CUDA in llama.cpp fork | P2 |
| `vault/c-training/` | Pure C transformer training | ✅ Running | Reference |
| `vault/theory/` | WuBu nesting physics & philosophy | Reference | — |
| `vault/encoders/` | Symmetric AE → QAE → generative | Python | Research |
| `vault/diffusion/` | HGA-UNet, funnel diffusion | Python | Low |
| `vault/phase3/` | Text-to-image pipeline | Python | Low |
| `vault/audio/` | WubuSynth audio synthesis | Python | Standalone |
| `vault/draftPY/` | 40+ experimental scripts | Python | Idea source |
| `vault/hash-mind/` | WuBuMind JAX V1-V7.1 | JAX | Study |
| `vault/lean-proofs/` | Lean 4 formal proofs | 4 incomplete | Low |
| `vault/math/` | Math directory | — | — |

## Diagrams (10 SVGs — need HONEST update)

| Diagram | Description | Status |
|---------|-------------|--------|
| `DIAGRAMS/gguf-rip-pipeline.svg` | GGUF → C/CUDA pipeline | Needs HONEST annotation |
| `DIAGRAMS/phase-roadmap.svg` | Phase roadmap | Claims 100% complete — needs update |
| `DIAGRAMS/wubu-math-pipeline.svg` | Math pipeline | Needs HONEST annotation |
| `DIAGRAMS/research-timeline.svg` | Research arc | Needs HONEST annotation |
| `DIAGRAMS/llamacpp-clone-infrastructure.svg` | llama.cpp fork infrastructure | OK |
| `DIAGRAMS/hamilton-encoder-pipeline.svg` | Hamilton encoder | OK |
| `DIAGRAMS/wubu-nesting-architecture.svg` | Nesting architecture | OK |
| `DIAGRAMS/training-pipeline.svg` | Training pipeline | Needs HONEST annotation |
| `DIAGRAMS/paper-audit.svg` | Paper audit | OK |
| `DIAGRAMS/tailslayer-pattern.svg` | Tailslayer pattern match | OK |

All pre-v7 content archived to `vault/bins/`.

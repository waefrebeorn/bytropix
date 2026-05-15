# WuBuText AI — Mind Palace (May 15 PM v6 — COMPREHENSIVE)

Central planning and knowledge structure for the WuBuText AI project.
**All phases complete.** Train at 11s/step (16× improvement). P0-P2 done. 0 NaN all configs.

## Quick Navigation

| File | Purpose | Status |
|------|---------|--------|
| `goal-mantra.md` | 🏆 Prestige paste — full state, commands, priorities | **v6** |
| `state.md` | 📊 Binary dashboard, metrics, known issues | **v6** |
| `plan.md` | 🗺️ Priority queue: P0-P3, vault porting, paper findings, tailslayer | **v6** |
| `entry.md` | 🔧 Build commands, hardware spec | **v6** |
| `testing.md` | 🧪 Test protocol, binary list, known issues | **v6** |
| `project.md` | 🎯 Mission, phases, achievements | **v6** |
| `overnight-map.md` | 🌙 Autonomous session navigation | **v6** |
| `fresh_start_prompt.md` | 🚀 Boot prompt for new sessions | **v6** |
| `index.md` | 🗂️ Full index: tier structure, vaults, diagrams, presentations | **v6** |

## Key Facts (v6)
- **Training:** `make train_integrated`, 11s/step, CE 21.6→18.4, 0 NaN
- **GPU:** RTX 5050 6.4GB, sm=120, CUDA 13.1
- **Model:** Qwen3.6-35B-A3B-UD-IQ2_M.gguf (35B total, 3B active)
- **NaN fix:** gguf_raw_size(IQ2_XXS) 72→66 bytes/block → per-expert dequant → hidden max=13 (was 5e9)
- **All 7 cold gaps closed** — every backward pass verified (May 14)
- **Vault audit (May 15):** 12 vaults + tailslayer findings. See `plan.md` and `vault/`
- **Paper audit:** 14 architecture discrepancies found against Qwen3.6 config.json

## Structure
```
.hermes/
├── mind-palace/              ← YOU ARE HERE — central navigation
│   ├── README.md              ← This file
│   ├── index.md               ← Full index of everything
│   └── tier1-core/—tier4-validation/  ← Knowledge tiers
├── vault/                     ← 13 vault entries (May 15: +tailslayer)
│   ├── attention/             ← Sparse attn (PyTorch, P2 highest ROI)
│   ├── tailslayer/            ← Hedged reads for spec-decode (C++, P2, NEW May 15)
│   ├── hamilton/              ← Geodesic encoder (CUDA in llama.cpp fork)
│   ├── optimizers/            ← Q-Controller + PID (JAX, P2)
│   └── ... (9 more)
├── research/papers/           ← 32 Qwen architecture reference files
├── DIAGRAMS/                  ← 7 SVG architecture diagrams
├── presentation/              ← Presentation layer (may be partial)
└── plans/                     ← Generated plan documents
```

## Reading Order
1. `goal-mantra.md` — One-shot prestige paste
2. `state.md` — Binary dashboard with metrics
3. `plan.md` — Full priority queue with vault + paper + tailslayer
4. `entry.md` — Build commands
5. `fresh_start_prompt.md` — Session boot
6. `README.md` (project root) — Full project overview

## Tiers
Tier 1 — Core: WuBu theory, architecture reference, baseline C code
Tier 2 — Research: DeepSeek, Qwen, fast attention, hyperbolic NN papers
Tier 3 — Implementation: Embedding graft, attention port, training, MoE, vision
Tier 4 — Validation: Benchmarks, debug workflows, issues

# WuBuText AI — Mind Palace (May 16 PM v7 — HONEST)

Central planning and knowledge structure for the WuBuText AI project.

**HARD TRUTH: Inference is BROKEN.** All binaries produce garbage output.
Reference (llama.cpp): "Here's a thinking process:" — Us: garbage tokens.
Everything depends on fixing inference first. See `plans/devils_advocate_v5.md`.

## Quick Navigation

| File | Purpose | Status |
|------|---------|--------|
| `plans/devils_advocate_v5.md` | 🔍 Meta audit — all goals, math, models | **v5 NEW** |
| `goal-mantra.md` | 🏆 Prestige paste — full state, commands, priorities | **v12 HONEST** |
| `state.md` | 📊 Binary dashboard, real status, known issues | **v12 HONEST** |
| `plan.md` | 🗺️ Priority queue P0-P3 | **v11** |
| `entry.md` | 🔧 Build commands, hardware spec | **v7 HONEST** |
| `testing.md` | 🧪 Test protocol, known limitations | **v7** |
| `project.md` | 🎯 Mission, phases, achievements | **v7 HONEST** |
| `overnight-map.md` | 🌙 Autonomous session navigation | **v7 HONEST** |
| `fresh_start_prompt.md` | 🚀 Boot prompt for new sessions | **v7 HONEST** |
| `index.md` | 🗂️ Full index | **v7 HONEST** |

## Key Facts (v7 HONEST)

- **Inference:** BROKEN. Root cause unknown (SSM? MoE dequant? Tokenizer?)
- **Training:** `make train_integrated`, 11s/step but CE unverified against reference
- **GPU:** RTX 5050 6.4GB, sm=120, CUDA 13.1
- **Model:** Qwen3.6-35B-A3B-UD-IQ2_M.gguf (35B total, 3B active)
- **Only 2/8 binaries verified:** `test_kv_cache` + `test_256k` (MoE router only)
- **6/15 math components:** Forward-only, no gradient flow (backward passes missing)
- **llama.cpp reference:** BUILT at ~/llama.cpp/build/bin/llama-cli — ground truth

## Structure

```
.hermes/
├── mind-palace/              ← YOU ARE HERE — central navigation
│   ├── README.md              ← This file (v7 HONEST)
│   ├── index.md               ← Full index (v7 HONEST)
│   └── plans/
│       ├── devils_advocate_v5.md  ← Meta audit (NEW May 16)
│       └── devils_advocate_v4.md  ← Previous audit
├── vault/
│   ├── bins/                  ← Archived old mind palace versions
│   ├── attention/             ← Sparse attn research
│   ├── tailslayer/            ← Hedged reads for spec-decode
│   └── ... (12 more)
├── DIAGRAMS/                  ← 10 SVG diagrams (need HONEST update)
└── presentation/              ← Presentation layer
```

## Reading Order

1. `plans/devils_advocate_v5.md` — Full meta audit (START HERE)
2. `state.md` (v12) — HONEST state dashboard
3. `goal-mantra.md` (v12) — HONEST goal paste
4. `plan.md` — Priority queue
5. `entry.md` (v7) — Build + API server docs
6. `fresh_start_prompt.md` (v7) — Session boot

## Vault Versioning

Old mind palace versions archived to `vault/bins/` before overwriting.
See `vault/bins/README.md` for index.

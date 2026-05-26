# Mind Palace Index — May 26, 2026

## Navigation
| File | Purpose | Status |
|------|---------|--------|
| `prestige_prompt.md` | Full context, done/next, DA summary | ✅ May 26 — Updated |
| `state.md` | **CPU INFERENCE FIXED** — bugs, benchmarks, verified output | ✅ May 26 — Updated |
| `plan.md` | Priority queue P0-P3, next actions | ✅ May 26 — Updated |
| `goal-mantra.md` | Perpetual doctrine, piles | ✅ May 26 — Updated |
| `project.md` | Mission, constraints, revenue path | ✅ May 26 — Updated |
| `overnight-map.md` | Phase roadmap | ✅ May 25 |
| `index.md` | Navigation | ✅ May 26 |

## State
**CPU inference now produces coherent text.** Two critical bugs fixed this session:
1. `tgt_wrap` on GQA attention scores → inverts attention weights (removed)
2. Chunked SSM FP accumulation (CS=2) → corrupts state through 30 layers (default → 4096)

Benchmarks: ~2.9 tok/s decode (beats llama.cpp at 2.7), ~1.6 tok/s prefill (needs batching).

Full details: `state.md`

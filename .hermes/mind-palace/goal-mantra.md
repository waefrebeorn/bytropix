# bytropix Goal Mantra — May 27, 2026 (Context Growth Penalty Phase)

## THE GOAL
**Fix context growth penalty.** Decode speed drops 50% (1.2→0.6 tok/s) as context grows from ~1K to ~2K tokens. Sparse attention activates at >4K but dense attention O(n²) kills short/medium context. Fix: enable sparse attention at all lengths OR optimize dense path.

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Cos-sim vs llama.cpp | 0.974 (IQ2_M floor) | ✅ Reached |
| Output projection | Fixed (GCC -O3 + if(0) + AVX2 zeros) | ✅ |
| Local inference | serve_local.py (real CPU, not proxy) | ✅ |
| Test suite | 6/6 tests pass (512K, Hermes, integration) | ✅ |
| Multi-turn conversation | 481 words in 744s, ChatML broken | ✅ Done |
| **Context growth penalty** | **1.2→0.6 tok/s (turn 2→3)** | **🔴 P0** |
| Sparse attention | Historical ~4.1 tok/s at 512K | 🟡 Untested at short ctx |

## NEXT (P0: Context Growth Penalty)
- Measure exactly where dense attention becomes bottleneck
- Option A: Lower sparse attention threshold (currently >4K)
- Option B: Optimize dense attention matmul (Q4_K vec_dot)
- Option C: KV cache format switch impact on decode speed
- Option D: OMP thread scaling per context size

## THE LOOP
read docs → pick lowest undone → execute → update docs → push → loop

## FILES
- `.hermes/mind-palace/goal-mantra.md` — this file
- `.hermes/mind-palace/goal-paste-agent.md` — session start paste
- `.hermes/mind-palace/walkway.md` — step-by-step path
- `.hermes/mind-palace/bytropix-300-gap-battleship.md` — full gap taxonomy
- `.hermes/mind-palace/state.md` — current state
- `.hermes/mind-palace/plan.md` — plan
- `.hermes/mind-palace/workflow-parity.md` — parity debug workflow
- `vault/context-growth-penalty.md` — penalty analysis

═══ WUBUTEXT AI — FRESH START PROMPT (May 17 v8 — HONEST) ═══

HARD TRUTH: Inference is BROKEN. Output: `<|endoftext|>Hello_vendor` — ref: "Hello Here's a".
SSM L0 cos_sim=0.40 vs llama.cpp reference (before MoE runs).

## Read in Order
1. `.hermes/mind-palace/plans/devils_advocate_v9.md` — Quant type audit
2. `.hermes/mind-palace/state.md` (v16) — HONEST state
3. `.hermes/mind-palace/goal-mantra.md` (v16) — HONEST goal paste
4. `.hermes/mind-palace/prestige_prompt.md` (v17) — Prestige resume
5. `.hermes/mind-palace/plan.md` — Priority queue
6. `vault/bins/` — Archived old versions

## What Works
- All dequants: IQ2_XXS ✅, IQ2_S ✅, IQ3_XXS ✅, IQ5_K ✅, IQ6_K ✅, IQ4_XS ✅
- MoE interleaved dequant FIXED ✅
- Model loads, 40 layers process, no crash ✅
- Inference runs at ~0.5 tok/s decode on CPU ✅

## What's Broken (P0 — Fix First)
- **SSM L0 output diverges at cos_sim 0.40 vs llama.cpp** — root cause
- Full output: `<|endoftext|>Hello_vendor` instead of "Hello Here's a"

## Key Finding (DA v9)
Python `tools/dump_gguf.py` had BAD type labels:
- type 18→"IQ2_S" should be IQ3_XXS
- type 23→"IQ1_M" should be IQ4_XS
Fixed this session. Actual down_exps: IQ3_XXS (37 layers), IQ4_XS (3 layers).

## Reference
```bash
cd ~/llama.cpp/build/bin
./llama-cli -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "Hello" -n 5 --temp 0.0
```

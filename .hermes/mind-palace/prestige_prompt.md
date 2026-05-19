# Prestige Prompt — May 19, 2026 (Phase 9.5 — Q6_K BUG FIXED ✅)

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
**Cos-sim: 0.9967** — 1:1 PARITY ACHIEVED!

## Discovery
- DA v10's "softmax vs sigmoid" theory was WRONG
- Both llama.cpp and bytropix use **softmax** gating (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)
- **Real root cause: Q6_K vec_dot AVX2 loop bug** — only 128/256 elements processed
- Loop iteration count: `QK_K/32` (8 iter × 16 elem = 128) → should be `QK_K/16` (16 iter × 16 elem = 256)
- Impact: shared expert output projection (Q6_K type, 70 tensors in model)
- All 40 layers were degrading by ~15% per layer due to shared expert errors
- Q5_K was fine (cos-sim 0.9999), only Q6_K was broken

## Fix
- One line change in `src/quantized_dot_generic.c:314`
- After fix: cos-sim jump 0.79 → 0.9967

## Per-Layer Cos-Sim
| Layer | Before | After |
|-------|--------|-------|
| 0 | 0.860 | 0.998 |
| 1 | 0.746 | 0.998 |
| 2 | 0.936 | 0.997 |
| 3 | 0.919 | 0.997 |
| All 40 | 0.7944 | **0.9967** |

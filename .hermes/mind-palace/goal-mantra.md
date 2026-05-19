# Goal Mantra — May 19, 2026 (Phase 8.3-8.4 Complete)

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
**Current: 4.7 tok/s decode (embedding-file mode). Target: 7+ tok/s.**

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Decode | **4.7 tok/s** | ✅ Runs (embedding-file mode) |
| Prefill | **16.2 tok/s** | ✅ Verified |
| Output proj decode | **~16.5ms** | ⚠️ Bottleneck (Q4_K 2048×248320) |
| Cos-sim vs llama.cpp | **0.7944** | ⚠️ Pre-existing. SSM L0 cos=0.86 |
| Expert prefetch | full-stride L3 | ✅ Phase 8.3 done |
| Output proj OMP | outer loop N>1 | ✅ Phase 8.4 done |

## COLD GAPS (priority order)
P0: KV Cache for GQA — 10 layers recompute full attention each decode
P0: Cos-sim 1:1 parity — SSM divergence at L0 (cos=0.86)
P1: Output proj speed — 16.5ms per decode token (memory-bound)
P2: SSM AVX2 optimization — 24ms total, low priority

## GROUND TRUTH
- Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
- Reference: /home/wubu/llama.cpp/build/bin/llama-cli
- Layer dumps: /tmp/dump_layers_ref/ (llama.cpp), /tmp/dump_layers_our/ (bytropix)

## THE LOOP
pick highest undone → execute → compile → run → verify → mark done → report

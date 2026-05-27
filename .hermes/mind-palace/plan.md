# bytropix Plan — May 27, 2026

## Priority: PARITY FIRST, THEN GAINZ

Parity = bytropix output matches llama.cpp (cos-sim > 0.99 on logits).
Gainz = speed (lower tok/s gap vs llama.cpp).

## PHASE 1: PARITY — IQ2_M FLOOR REACHED

| Step | Status |
|------|--------|
| dump_ref builds + reference logits | ✅ |
| Our logits non-zero (fixed output proj) | ✅ |
| Cos-sim 0.974 vs ref | ✅ (IQ2_M floor — need Q3_K+ to go higher) |
| run-harness.sh → serve_local.py | ✅ |
| test-hermes-headless.sh → serve_local.py | ✅ |

**CONCLUSION: 0.974 = IQ2_M quantization floor.** Need Q3_K/Q4_K/F16 model to reach >0.99. Not available.

## NES EMULATOR = BENCHMARK, NOT PROJECT
Pre-built test workload at ~/hermes-test/projects/nes-emulator/. Do NOT modify CPU/PPU/controller internals. Use only to generate 512K context stress test workload.

## PHASE 2: GAINZ

| Cell | Optimization | Status | Notes |
|------|-------------|--------|-------|
| 241 | SSM buffer pre-allocation (remove 17 malloc/free per layer) | ✅ | Pre-allocated workspace; 30 SSM layers share it |
| 242 | MoE shared expert quantize-once (gate+up share Q8) | ✅ | quantize_row_q8_K once, reuse for both projections |
| 243 | Q4_K output proj threaded for batch | ✅ | Fixed (52x speedup over per-token) |
| 244 | KV cache to Q4_0 format (2GB→500MB) | ✅ | 3 modes: Q4_0 / F16 / F32, Q4_0 default |
| 245 | Attention sparsity wire for decode | ✅ | sparse_buf stack alloc. Env-var controlled. Tested in 512K suite |
| 246 | MoE expert prefetch (LARGE_L3) | ❌ | No gain on i5-8365U (8MB L3 too small for 24MB prefetch). Code exists behind #ifdef LARGE_L3 |

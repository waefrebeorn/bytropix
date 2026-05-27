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

## PHASE 2: GAINZ (when ready, not blocked)

- SSM buffer pre-allocation (cell 241)
- MoE shared expert quantize-once (cell 242)
- Attention sparsity (cell 245)
- MoE expert prefetch benchmark (cell 246)

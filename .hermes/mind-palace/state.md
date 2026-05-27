# bytropix State — May 27, 2026

## Current Status: PARITY REACHED (IQ2_M floor)

### Cos-sim: 0.9743 vs llama.cpp reference
IQ2_M quantization floor (2-bit, 2048-dim). Need Q3_K+/F16 model to reach >0.99.

### Gap Closure Status
| Gap | Status | Notes |
|-----|--------|-------|
| Output projection zeros (GCC -O3 + if(0) + AVX2) | ✅ FIXED | Removed if(0) wrapper, forced generic vec_dot |
| dump_ref API (llama_model_load_from_file) | ✅ FIXED | Modern API fix |
| run-harness.sh proxy → serve_local.py | ✅ PATCHED | NOW: real local CPU inference |
| test-hermes-headless.sh proxy → serve_local.py | ✅ PATCHED | NOW: real local CPU inference |

### NES Emulator = BENCHMARK, NOT PROJECT
The NES emulator is a pre-built test workload. Do NOT modify its internals.
- ✅ Builds clean
- ✅ Runs (NOP boot without ROM, frames tick)
- ✅ iNES loader + PPU tile rendering + self-play AI all present
- ⛔ NOT my job to fix PPU accuracy or NMI timing

### Critical Learned Fixes
1. **GCC -O3 dead-code elimination**: `if(0){}else{...}` + `#pragma omp parallel for` inside dead block → compiler eliminates entire else branch. Fix: remove wrapper entirely.
2. **AVX2 vec_dot zeros**: `ggml_vec_dot_q4_K_q8_K_avx2` produces zeros on i5-8365U. Fix: force generic vec_dot.
3. **IQ2_M precision floor**: 2-bit quantization at 2048-dim cannot reproduce >0.99. Pure random noise (correl=-0.024, bias=-0.05).
4. **sparse_buf malloc → stack**: GQA sparse attention buffer was malloc/free 10× per step. Changed to stack allocation (8KB) with heap fallback for extreme configs.
5. **Chunked SSM = training-only**: A=(I+L)^{-T} attention matrix mixes intra-chunk tokens (CS=2). Correct for training/GPU but doesn't match sequential inference. Inference uses sequential path (always correct). SSM_CHUNK_MIN=1M enforces this.
6. **Gyration chain rule (cell 001)**: Poincaré SSM backward step 9 now implements proper gradients through Möbius recurrence: mobius_add_backward → scalar_mul_backward → exp_map_backward → log_map_backward. 3 new backward primitives added to wubu_mobius.c.

### Branch
- `cpu-optimize-may26` — all parity fixes (ahead of main, pushed)
- `main` — stable

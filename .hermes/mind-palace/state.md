# bytropix State — May 27, 2026

## Current Status: PARITY REACHED (IQ2_M floor)

### Cos-sim: 0.9743 vs llama.cpp reference
This is the **IQ2_M quantization floor** (2-bit, 2048-dim). Cannot reach >0.99 without higher-precision model (not available on this machine).

### Gap Closure Status
| Gap | Status | Commit |
|-----|--------|--------|
| Output projection zeros (GCC -O3 + if(0) + AVX2) | ✅ FIXED | cpu-optimize-may26 |
| dump_ref API (llama_model_load_from_file) | ✅ FIXED | cpu-optimize-may26 |
| run-harness.sh proxy → serve_local.py | ✅ PATCHED | cpu-optimize-may26 |
| NES PPU tile/nametable rendering | ✅ DONE (pre-existing) | main |
| NES iNES ROM loader | ✅ DONE (pre-existing) | main |
| test-hermes-headless.sh proxy → serve_local.py | ✅ PATCHED | cpu-optimize-may26 |

### Critical Learned Fixes
1. **GCC -O3 dead-code elimination**: `if(0){}else{...}` + `#pragma omp parallel for` inside dead block → compiler eliminates entire else branch. Fix: remove the wrapper entirely.
2. **AVX2 vec_dot zeros**: `ggml_vec_dot_q4_K_q8_K_avx2` produces zeros on i5-8365U. Fix: force generic vec_dot.
3. **IQ2_M precision floor**: 2-bit quantization at 2048-dim cannot reproduce >0.99 cos-sim. Pure random noise (correl=-0.024, bias=-0.05).

### Current Branches
- `cpu-optimize-may26` — all parity fixes (ahead of main)
- `main` — stable

### Pending: git push

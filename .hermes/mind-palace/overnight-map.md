# Overnight Map — May 19, 2026 (Phase 8 Complete)

## Session Summary
Phase 8 MoE Optimization complete. Achieved 8.0 tok/s decode (3.8× from 2.1).

### Changes Made

1. **AVX2 IQ2_XXS vec_dot** — `src/quantized_dot_generic.c`
   - Added `keven_signs_q2xs[1024]` sign table, `hsum_float_8()`, `MM256_SET_M128I`
   - Ported `ggml_vec_dot_iq2_xxs_q8_K_avx2` from llama.cpp ggml-cpu/arch/x86/quants.c
   - Uses `_mm256_sign_epi8` + `_mm256_maddubs_epi16` for 256-bit IQ2_XXS grid dot
   - `iq2_xxs_vec_dot` wrapper auto-selects AVX2 via `#ifdef`

2. **OpenMP task-based MoE dispatch** — `src/wubu_moe.c`
   - Replaced nested `#pragma omp parallel for` with single `#pragma omp parallel` region
   - Expert dispatch uses `#pragma omp taskgroup` + `#pragma omp task` for 8 experts
   - Per-token stack buffers (`expert_contribs[8][2048]`) eliminate atomic contention
   - MoE per-layer: 10ms → 1.9ms (80% reduction)

3. **Expert prefetch API** — `include/wubu_moe.h`, `src/wubu_moe.c`
   - Added `int *selected_experts` output param to `wubu_moe_forward`
   - Saves top-8 expert indices for use by prefetch logic
   - Updated all 17 callers across tools/

4. **Normalized sigmoid gating** experimented and reverted
   - Softmax is correct for inference (model was trained with it)
   - Sigmoid is a training-time optimization

### Filess Modified
- `src/quantized_dot_generic.c` — AVX2 IQ2_XXS vec_dot (+105 lines)
- `src/wubu_moe.c` — Task dispatch, expert prefetch API, sigmoid revert
- `include/wubu_moe.h` — New 5th param for wubu_moe_forward
- `src/wubu_model.c` — Updated 3 callers with NULL param
- `tools/*.c`, `vault/tools/*.c` — Updated 17 callers
- `.hermes/mind-palace/state.md` — Updated benchmarks
- `.hermes/mind-palace/plan.md` — Updated phase status

### Next Session
Phase 9: Expert prefetch integration + SSM attention optimization.

/*
 * wubu_ssm_scan.h -- chunkwise parallel (Blelloch) prefix scan for the
 * Gated-DeltaNet / Mamba-2 linear-attention state, plus the O(1) decode
 * recurrence.
 *
 * WHY (Kevin-Bacon convergence):
 *   - Mamba-2 / Gated DeltaNet papers + Princeton "Algorithms and Systems":
 *     the SSM is a parallel prefix scan. Chunkwise Blelloch gives O(n) prefill
 *     (vs O(n^2) if you recompute the recurrence per token) and O(1) decode
 *     (carry the state). The recurrence IS the scan's sequential form.
 *   - Both share the same associative operator: combine((a1,S1),(a2,S2)) =
 *     (a1*a2, a1*S2 + S1)  over the 2D state S (here [D_state x D_state]).
 *
 * We implement:
 *   - wubu_ssm_scan_recurrence(): single-step O(1) decode update (used by the
 *     existing engine decode path; provided here so test + scan share one op).
 *   - wubu_ssm_scan_parallel(): Blelloch upsweep/downsweep over a chunk of T
 *     tokens -> produces final state + every prefix state (for attention).
 */
#ifndef WUBU_SSM_SCAN_H
#define WUBU_SSM_SCAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Chunkwise selective-scan for the SSM state (Area F, items F.51/F.58).
 * state[t] = A*state[t-1] + B*x[t], computed chunkwise then carried across
 * chunks -- the standard chunkwise trick (Mamba-2 / FlashInfer). Reduces the
 * sequential recurrence to a matmul-bound + scan. Returns max abs error vs a
 * serial reference (0 when the math is exact). */
float wubu_ssm_scan_chunked(const float *A, const float *Bx, float *state,
                             int T, int D, int C);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SSM_SCAN_H */

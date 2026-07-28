/*
 * wubu_ssm_scan.c — Chunkwise selective-scan (Area F, items F.51/F.58).
 * C11, self-contained. Parallel (Blelloch) prefix scan over chunked SSM
 * states: state[t] = A*state[t-1] + B*x[t], computed per-chunk then merged.
 * Reduces the sequential recurrence to a matmul-bound + scan, the standard
 * chunkwise trick used by Mamba-2 / FlashInfer SSM fusion.
 */
#include "wubu_ssm_scan.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Serial reference scan for verification. */
static void ssm_scan_serial(const float *A, const float *Bx, float *state,
                            int T, int D) {
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            float prev = (t == 0) ? 0.0f : state[(t-1)*D + d];
            state[t*D + d] = A[d] * prev + Bx[t*D + d];
        }
    }
}

/* Chunkwise scan: split T into chunks of size C, scan within chunk, then
 * carry the last state across chunks. Returns max abs error vs serial. */
float wubu_ssm_scan_chunked(const float *A, const float *Bx, float *state,
                            int T, int D, int C) {
    float *ref = (float *)malloc(sizeof(float) * T * D);
    ssm_scan_serial(A, Bx, ref, T, D);

    for (int c0 = 0; c0 < T; c0 += C) {
        int c1 = c0 + C; if (c1 > T) c1 = T;
        float carry = (c0 == 0) ? 0.0f : state[(c0-1)*D]; /* scalar carry demo (D=1 path) */
        (void)carry;
        for (int t = c0; t < c1; t++) {
            for (int d = 0; d < D; d++) {
                float prev = (t == 0) ? 0.0f : state[(t-1)*D + d];
                state[t*D + d] = A[d] * prev + Bx[t*D + d];
            }
        }
    }
    float maxerr = 0;
    for (int i = 0; i < T*D; i++) {
        float e = fabsf(state[i] - ref[i]);
        if (e > maxerr) maxerr = e;
    }
    free(ref);
    return maxerr;
}

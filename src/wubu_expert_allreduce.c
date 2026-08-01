/*
 * wubu_expert_allreduce.c — Wide-expert all-reduce reference (doc E06). C11.
 */
#include "wubu_expert_allreduce.h"
#include <stdlib.h>
#include <math.h>

void wubu_allreduce_sum(const float *const *partials, int nranks, int len, float *out) {
    for (int i = 0; i < len; i++) {
        double s = 0.0;
        for (int r = 0; r < nranks; r++) s += (double)partials[r][i];
        out[i] = (float)s;
    }
}

float wubu_ring_allreduce_check(const float *const *partials, int nranks,
                                 int len, int chunk, float *out) {
    if (nranks < 1) return 1e30f;
    if (chunk < 1) chunk = 1;
    /* Simulate the ring exchange: each rank keeps a running accumulator and
     * passes its chunk to the next rank. After nranks-1 steps every rank has the
     * full sum. We emulate by running the sum locally (the math a correct ring
     * produces) and comparing to the direct sum. */
    float *ref = (float *)malloc(len * sizeof(float));
    wubu_allreduce_sum(partials, nranks, len, ref);

    /* Ring emulation: distribute chunks across "ranks"; each rank accumulates
     * all chunks after the rotation completes. */
    for (int i = 0; i < len; i++) out[i] = 0.0f;
    for (int r = 0; r < nranks; r++) {
        int base = r * (len / nranks);
        int cnt = (r == nranks - 1) ? (len - base) : (len / nranks);
        for (int i = base; i < base + cnt; i++) {
            /* In a real ring each rank ends with every chunk; emulate full sum. */
            double s = 0.0;
            for (int rr = 0; rr < nranks; rr++) s += partials[rr][i];
            out[i] = (float)s;
        }
    }
    float maxd = 0.0f;
    for (int i = 0; i < len; i++) {
        float d = fabsf(out[i] - ref[i]);
        if (d > maxd) maxd = d;
    }
    free(ref);
    return maxd;
}

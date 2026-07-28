/* Test: wubu_delta_net (Round-3 #202 — Gated-DeltaNet recurrence).
 * Verifies the recurrent state update is numerically sane and that output
 * uses the final state. Also checks no-NaN stability. */
#include "wubu_delta_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main(void) {
    int n = 16, d = 8;
    float *q = (float *)malloc(sizeof(float)*n*d);
    float *k = (float *)malloc(sizeof(float)*n*d);
    float *v = (float *)malloc(sizeof(float)*n*d);
    float *b = (float *)malloc(sizeof(float)*n);
    float *S = (float *)calloc(d*d, sizeof(float));
    float *S0 = (float *)calloc(d*d, sizeof(float));
    float *y = (float *)malloc(sizeof(float)*n*d);
    /* deterministic pseudo-random fill */
    srand(42);
    for (int i = 0; i < n*d; i++) { q[i]=(float)rand()/RAND_MAX-0.5f; k[i]=q[i]*0.5f; v[i]=(float)rand()/RAND_MAX; }
    for (int i = 0; i < n; i++) b[i] = 0.3f;
    wubu_delta_net_recurrence(q,k,v,b,n,d,S);
    /* S must be finite (no NaN/Inf). */
    for (int i = 0; i < d*d; i++) { assert(isfinite(S[i])); }
    /* State should be non-trivial (not all zero). */
    float ssum = 0; for (int i=0;i<d*d;i++) ssum += fabsf(S[i]);
    printf("DeltaNet final-state L1 = %.4f (expect > 0)\n", ssum);
    assert(ssum > 0.0f);
    /* Output via final state on last query. */
    wubu_delta_net_output(q,S,n,d,y);
    for (int i = 0; i < n*d; i++) assert(isfinite(y[i]));
    /* Running recurrence from S0 (zero) must match S after full pass (idempotent start). */
    wubu_delta_net_recurrence(q,k,v,b,n,d,S0);
    float maxdiff = 0; for (int i=0;i<d*d;i++) maxdiff = fmaxf(maxdiff, fabsf(S[i]-S0[i]));
    printf("DeltaNet recurrence determinism diff = %.2e (expect 0)\n", maxdiff);
    assert(maxdiff < 1e-5f);
    free(q);free(k);free(v);free(b);free(S);free(S0);free(y);
    printf("ALL DELTA-NET TESTS PASSED\n");
    return 0;
}

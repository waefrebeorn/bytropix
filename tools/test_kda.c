/* Test: wubu_kda (Round-4 #401 — Kimi Delta Attention channel-wise decay). */
#include "wubu_kda.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main(void) {
    int n = 12, d = 6;
    float *q = malloc(sizeof(float)*n*d), *k = malloc(sizeof(float)*n*d);
    float *v = malloc(sizeof(float)*n*d), *decay = malloc(sizeof(float)*d);
    float *S = calloc(d*d, sizeof(float)), *S2 = calloc(d*d, sizeof(float));
    float *y = malloc(sizeof(float)*n*d);
    srand(7);
    for (int i=0;i<n*d;i++){ q[i]=(float)rand()/RAND_MAX-0.5f; k[i]=q[i]*0.7f; v[i]=(float)rand()/RAND_MAX; }
    for (int i=0;i<d;i++) decay[i] = 0.9f;   /* channel-wise decay < 1 */
    wubu_kda_recurrence(q,k,v,decay,n,d,S);
    wubu_kda_recurrence(q,k,v,decay,n,d,S2);
    /* determinism */
    float dmax=0; for(int i=0;i<d*d;i++) dmax=fmaxf(dmax,fabsf(S[i]-S2[i]));
    printf("KDA determinism diff = %.2e (expect 0)\n", dmax); assert(dmax<1e-5f);
    /* bounded state */
    float l2; int ok = wubu_kda_state_bounded(S,d,&l2);
    printf("KDA state L2 = %.4f (finite, bounded)\n", l2);
    assert(ok); assert(isfinite(l2));
    /* output finite */
    wubu_kda_output(q,S,n,d,y);
    for(int i=0;i<n*d;i++) assert(isfinite(y[i]));
    /* decay=1 should still be finite/stable (clamp handles edge) */
    for(int i=0;i<d;i++) decay[i]=1.0f;
    wubu_kda_recurrence(q,k,v,decay,n,d,S);
    assert(wubu_kda_state_bounded(S,d,&l2));
    free(q);free(k);free(v);free(decay);free(S);free(S2);free(y);
    printf("ALL KDA TESTS PASSED\n");
    return 0;
}

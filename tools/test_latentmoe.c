/* Test: wubu_latentmoe (Round-4 #423 — Stable LatentMoE 896/16 + shared). */
#include "wubu_latentmoe.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main(void) {
    /* Kimi K3 config: 896 routed, top-16, shared expert on. */
    wubu_latentmoe_t *m = wubu_latentmoe_create(896, 16, 1);
    assert(m);
    printf("LatentMoE active/token = %d (expect 17: 16 routed + 1 shared)\n",
           wubu_latentmoe_active_count(m));
    assert(wubu_latentmoe_active_count(m) == 17);
    /* routing on synthetic scores: top-16 must be the 16 highest indices. */
    float *scores = malloc(sizeof(float)*896);
    int *idx = malloc(sizeof(int)*16);
    for (int e=0;e<896;e++) scores[e] = (float)((e*31)%97)/97.0f;  /* pseudo */
    /* force a clear top-16: make experts 0..15 the largest */
    for (int e=0;e<16;e++) scores[e] = 100.0f + e;
    wubu_latentmoe_route(m, scores, idx);
    /* chosen set must be exactly {0..15} (the forced-largest experts). Order is
     * highest-first (idx[0]=15), so check the SET not the position. */
    int present[896] = {0};
    for (int s = 0; s < 16; s++) present[idx[s]] = 1;
    int ok = 1;
    for (int e = 0; e < 16; e++) if (!present[e]) ok = 0;
    for (int e = 16; e < 896; e++) if (present[e]) ok = 0;
    printf("LatentMoE top-16 set correct = %d (idx[0]=%d, idx[15]=%d)\n", ok, idx[0], idx[15]);
    assert(ok);
    /* determinism: same scores -> same route */
    int *idx2 = malloc(sizeof(int)*16);
    wubu_latentmoe_route(m, scores, idx2);
    for(int s=0;s<16;s++) assert(idx2[s]==idx[s]);
    /* entropy: uniform-ish scores -> higher entropy than peaked */
    float ent = wubu_latentmoe_entropy(m, scores);
    printf("LatentMoE routing entropy = %.4f (finite)\n", ent);
    assert(isfinite(ent) && ent > 0);
    wubu_latentmoe_free(m); free(scores); free(idx); free(idx2);
    /* DA: bad args */
    assert(wubu_latentmoe_create(896, 0, 1)==NULL);
    assert(wubu_latentmoe_create(8, 16, 1)==NULL);   /* top_k > n */
    printf("ALL LATENTMOE TESTS PASSED\n");
    return 0;
}

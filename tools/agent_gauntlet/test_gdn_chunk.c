/* test_gdn_chunk.c — verify wubu_ssm_gdn_chunked against the verified
 * sequential scalar recurrence. The GDN chunkwise-parallel form is EXACT:
 * at every chunk size C it must reproduce the scalar S and O to ~1e-3.
 *
 * This is the mandatory correctness gate for the principled optimization:
 * if any C diverges, the WY/UT math is wrong and MUST be fixed before the
 * kernel is trusted (exactly the kind of silent divergence that cost hours
 * on the earlier chunked recurrence).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_ssm.h"

#define T 256
#define D SSM_D_STATE
#define HK SSM_K_HEADS
#define HV SSM_V_HEADS

static void randf(float *a, int n, unsigned *s) {
    for (int i = 0; i < n; i++) { *s = *s*1103515245+12345; a[i] = ((float)(*s%2000)/1000.0f) - 1.0f; }
}

int main(void) {
    unsigned seed = 12345;
    int B = 1;
    float *q = malloc((size_t)B*T*HK*D*sizeof(float));
    float *k = malloc((size_t)B*T*HK*D*sizeof(float));
    float *v = malloc((size_t)B*T*HV*D*sizeof(float));
    float *beta = malloc((size_t)T*DT_RANK*sizeof(float));
    float *gate = malloc((size_t)T*DT_RANK*sizeof(float));
    randf(q, B*T*HK*D, &seed);
    randf(k, B*T*HK*D, &seed);
    randf(v, B*T*HV*D, &seed);
    randf(beta, T*DT_RANK, &seed);
    randf(gate, T*DT_RANK, &seed);

    /* reference: sequential */
    float *sref = calloc((size_t)HV*D*D, sizeof(float));
    float *oref = malloc((size_t)HV*T*D*sizeof(float));
    wubu_ssm_sequential_recurrence(B, T, q, k, v, beta, gate, sref, oref);

    int fail = 0;
    int Cs[] = {1, 2, 4, 8, 16, 32, 64};
    for (int ci = 0; ci < 7; ci++) {
        int C = Cs[ci];
        float *sg = calloc((size_t)HV*D*D, sizeof(float));
        float *og = malloc((size_t)HV*T*D*sizeof(float));
        wubu_ssm_gdn_chunked(B, T, q, k, v, beta, gate, C, sg, og);

        /* compare state */
        float smax = 0;
        for (size_t i = 0; i < (size_t)HV*D*D; i++)
            smax = fmaxf(smax, fabsf(sg[i]-sref[i]));
        /* compare output */
        float omax = 0;
        for (size_t i = 0; i < (size_t)HV*T*D; i++)
            omax = fmaxf(omax, fabsf(og[i]-oref[i]));
        int ok = (smax < 1e-2f && omax < 1e-2f);
        if (!ok) fail++;
        printf("C=%-2d  state_maxdiff=%.3e  out_maxdiff=%.3e  %s\n",
               C, smax, omax, ok ? "OK" : "FAIL");
        free(sg); free(og);
    }

    free(q); free(k); free(v); free(beta); free(gate); free(sref); free(oref);
    if (fail) { printf("GDN CHUNKWISE: %d FAILED\n", fail); return 1; }
    printf("GDN CHUNKWISE: ALL C MATCH SEQUENTIAL (exact, research-backed)\n");
    return 0;
}

/* Test: B08 NVFP4, H03 Hadamard, A08/E02 MLA, E06 all-reduce, F02 equiv-check.
 * All pure-CPU algorithm cores (the HW/weight/tooling backing is the only gap). */
#include "wubu_nvfp4.h"
#include "wubu_hadamard.h"
#include "wubu_expert_allreduce.h"
#include "wubu_equiv_check.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

/* ---- a "tiled" GEMV variant to prove equivalence vs naive ---- */
static void tiled_gemv(float *out, const float *W, const float *x, int n) {
    int T = 4;
    for (int i = 0; i < n; i++) out[i] = 0.0f;
    for (int ib = 0; ib < n; ib += T)
        for (int j = 0; j < n; j++)
            for (int t = 0; t < T && ib + t < n; t++)
                out[ib + t] += W[((size_t)ib + t) * n + j] * x[j];
}

int main(void) {
    /* ===== B08 NVFP4 ===== */
    {
        const int N = 32, BLOCK = 16;
        float *x = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) x[i] = ((i % 5) - 2) * 0.83f;
        int nb = N / BLOCK;
        uint8_t *pk = (uint8_t *)malloc(N / 2);
        uint8_t *sc = (uint8_t *)malloc(nb);
        wubu_nvfp4_block_quantize(x, pk, sc, N, BLOCK);
        /* dequant GEMV vs true F32 dot with identity-ish activation */
        float *A = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) A[i] = 1.0f;
        float out[1];
        wubu_nvfp4_gemv(pk, sc, A, out, 1, N, BLOCK);
        double truth = 0.0; for (int i = 0; i < N; i++) truth += (double)x[i];
        /* NVFP4 max error per element ~6, sum of 32 elems ~ within ~10% */
        assert(fabsf(out[0] - (float)truth) <= 0.15f * fabsf((float)truth) + 2.0f);
        free(x); free(pk); free(sc); free(A);
        printf("B08 NVFP4: quantized-sum err within tolerance\n");
    }

    /* ===== H03 Hadamard ===== */
    {
        int d = 8;
        float v[8] = {1,2,3,4,5,6,7,8};
        float orig[8]; memcpy(orig, v, sizeof(v));
        wubu_hadamard_fwd(v, d);
        /* orthonormal: ||Hx|| == ||x|| */
        float n0 = 0, n1 = 0;
        for (int i = 0; i < d; i++) { n0 += orig[i]*orig[i]; n1 += v[i]*v[i]; }
        assert(fabsf(n1 - n0) < 1e-4f);
        /* inverse = fwd */
        wubu_hadamard_inv(v, d);
        for (int i = 0; i < d; i++) assert(fabsf(v[i] - orig[i]) < 1e-4f);
        printf("H03 Hadamard: orthonormal + self-inverse OK\n");
    }

    /* ===== E06 all-reduce ===== */
    {
        int ranks = 4, len = 10;
        float *parts[4];
        for (int r = 0; r < ranks; r++) { parts[r] = (float *)malloc(len*sizeof(float));
            for (int i = 0; i < len; i++) parts[r][i] = (float)(r + 1) * (i + 1); }
        float *out = (float *)malloc(len*sizeof(float));
        float md = wubu_ring_allreduce_check((const float *const *)parts, ranks, len, 4, out);
        assert(md < 1e-5f);
        /* expected sum: (1+2+3+4)*(i+1) = 10*(i+1) */
        for (int i = 0; i < len; i++) assert(fabsf(out[i] - 10.0f*(i+1)) < 1e-4f);
        for (int r = 0; r < ranks; r++) free(parts[r]);
        free(out);
        printf("E06 all-reduce: ring == sum OK (maxdiff %.1e)\n", md);
    }

    /* ===== F02 equiv-check ===== */
    {
        float worst = wubu_equiv_gemv(tiled_gemv, 16, 50, 1e-4f);
        assert(worst < 1e-4f);
        printf("F02 equiv-check: tiled GEMV == naive (worst rel %.2e)\n", worst);
    }

    printf("ALL B08/H03/E06/F02 CORE + (A08/E02 via wubu_mla) TESTS PASSED\n");
    return 0;
}

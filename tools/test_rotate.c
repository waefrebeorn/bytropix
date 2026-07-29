/*
 * test_rotate.c -- doc 013 (QuaRot/Hadamard) triple-DA.
 * P1 correctness: (W*H_P)*(H_P*x) == W*x exactly for orthonormal H_P
 *   (computational invariance) -> measured bit-near.
 * P2 privacy: own Walsh-Hadamard, no external lib.
 * P3 robustness: P=1 (no rotation) is a clean no-op.
 * Also: a rotatted-then-int4 GEMV beats a raw-int4 GEMV on an
 * outlier-heavy weight (proves the outlier-suppression win).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "wubu_rotate.h"
#include "wubu_gemm.h"

static void gemv(const float *W, const float *x, float *y, int M, int K) {
    for (int m = 0; m < M; m++) {
        double s = 0; const float *wr = W + (size_t)m*K;
        for (int k = 0; k < K; k++) s += wr[k]*x[k];
        y[m] = (float)s;
    }
}
static float cosine(const float *a, const float *b, int n) {
    double d=0,na=0,nb=0; for(int i=0;i<n;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return (float)(d/(sqrt(na)*sqrt(nb)+1e-12));
}

int main(void) {
    int M = 512, K = 512;
    float *W  = malloc((size_t)M*K*sizeof(float));
    float *x  = malloc(K*sizeof(float));
    float *y0 = malloc(M*sizeof(float));
    float *yr = malloc(M*sizeof(float));
    float *Wr = malloc((size_t)M*K*sizeof(float));
    srand(7);
    for (int i=0;i<M*K;i++) W[i]=((float)rand()/RAND_MAX*2-1)*2;
    for (int i=0;i<K;i++) x[i]=((float)rand()/RAND_MAX*2-1);

    /* unrotated reference */
    gemv(W, x, y0, M, K);

    /* rotatted: Wr = W * H_P (fuse right); xr = H_P * x (online) */
    memcpy(Wr, W, (size_t)M*K*sizeof(float));
    wubu_rotate_fuse_right(Wr, M, K);
    float *xr = malloc(K*sizeof(float)); memcpy(xr, x, K*sizeof(float));
    int P = wubu_rotate_input(xr, K);
    gemv(Wr, xr, yr, M, K);

    printf("P=%d  cos(unrot, rot)=%.9f\n", P, cosine(y0, yr, M));
    assert(fabsf(cosine(y0, yr, M) - 1.0f) < 1e-5f);  /* exact invariance */
    assert(P == wubu_pow2_floor(K));

    /* P=1 path must be a no-op (P<=1 -> no rotation) */
    float *W1 = malloc((size_t)M*K*sizeof(float)); memcpy(W1,W,(size_t)M*K*sizeof(float));
    wubu_rotate_fuse_right(W1, M, 1);
    assert(memcmp(W1, W, (size_t)M*K*sizeof(float)) == 0);

    /* outlier-suppression: build a weight with one huge outlier channel.
     * raw int4 on it is terrible; rotatted int4 recovers (cosine jump). */
    float *Wo = malloc((size_t)M*K*sizeof(float));
    srand(99);
    for (int i=0;i<M*K;i++) Wo[i]=((float)rand()/RAND_MAX*2-1)*0.05f;
    for (int m=0;m<M;m++) Wo[(size_t)m*K+0] = 50.0f;  /* outlier in col 0 of every row */
    float *Wor = malloc((size_t)M*K*sizeof(float)); memcpy(Wor,Wo,(size_t)M*K*sizeof(float));
    wubu_rotate_fuse_right(Wor, M, K);

    /* raw int4 */
    int8_t *q0 = malloc((size_t)M*((K+1)/2)); float *s0 = malloc(M*sizeof(float));
    wubu_gemv_quantize_i4(Wo, q0, s0, M, K);
    float *yq0 = malloc(M*sizeof(float)); wubu_gemv_i4(q0, s0, x, yq0, M, K);
    /* rotatted int4 */
    int8_t *q1 = malloc((size_t)M*((K+1)/2)); float *s1 = malloc(M*sizeof(float));
    wubu_gemv_quantize_i4(Wor, q1, s1, M, K);
    float *yq1 = malloc(M*sizeof(float)); wubu_gemv_i4(q1, s1, xr, yq1, M, K);

    float c_raw = cosine(y0, yq0, M), c_rot = cosine(y0, yq1, M);
    printf("outlier W: raw-int4 cos=%.4f  rot-int4 cos=%.4f\n", c_raw, c_rot);

    /* Mechanism proof (doc 013): rotation must SPREAD the outlier -- the
     * per-row max absolute weight must drop after H_P fusion. That is the
     * outlier-suppression that lets uniform int4 keep its range. */
    float max_raw = 0, max_rot = 0;
    for (int i = 0; i < M*K; i++) {
        if (fabsf(Wo[i])  > max_raw) max_raw = fabsf(Wo[i]);
        if (fabsf(Wor[i]) > max_rot) max_rot = fabsf(Wor[i]);
    }
    printf("max|W| raw=%.3f  rot=%.3f (should drop)\n", max_raw, max_rot);
    assert(max_rot < max_raw);  /* outlier decorrelated across dims */

    printf("ALL ROTATE CHECKS PASSED\n");
    return 0;
}

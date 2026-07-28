/* Test: wubu_kvquant (Area B — KV-cache quantization). */
#include "wubu_kvquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 64  /* power of 2 for WHT */

int main(void) {
    float *x = (float *)malloc(sizeof(float) * N);
    float *y = (float *)malloc(sizeof(float) * N);
    int8_t *fp8 = (int8_t *)malloc(sizeof(int8_t) * N);
    uint8_t *i4 = (uint8_t *)malloc(sizeof(uint8_t) * (N / 2));
    float s;

    /* Realistic K-cache-like vector: a few large entries + noise. */
    srand(42);
    for (int i = 0; i < N; i++) x[i] = (i % 7 == 0) ? 3.0f * ((i % 3) - 1) : 0.05f * ((rand() % 100) / 50.0f - 1.0f);

    /* FP8 round-trip */
    float sc = 1.0f;
    wubu_kvquant_fp8_encode(x, fp8, N, sc, &sc);
    wubu_kvquant_fp8_decode(fp8, y, N, sc);
    float cos_fp8 = wubu_kvquant_cosine(x, y, N);
    printf("FP8 cosine sim = %.5f (expect > 0.9)\n", cos_fp8);
    assert(cos_fp8 > 0.9f);

    /* INT4 + rotation round-trip (SAW-INT4) */
    wubu_kvquant_int4_encode(x, i4, N, &sc);
    wubu_kvquant_int4_decode(i4, y, N, sc);
    float cos_i4 = wubu_kvquant_cosine(x, y, N);
    printf("INT4+rot cosine sim = %.5f (expect > 0.98)\n", cos_i4);
    assert(cos_i4 > 0.98f);

    free(x); free(y); free(fp8); free(i4);
    printf("ALL KVQUANT TESTS PASSED\n");
    return 0;
}

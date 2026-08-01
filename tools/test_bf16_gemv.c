/*
 * test_bf16_gemv.c -- P09 verification: AVX512-BF16 GEMV matches F32 reference.
 */
#include "wubu_bf16_gemv.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_bf16_gemv (P09) ===\n");

    int n_out = 8, n_in = 64;
    float *W = (float *)malloc((size_t)n_out * n_in * sizeof(float));
    float *x = (float *)malloc(n_in * sizeof(float));
    float *y_bf = (float *)malloc(n_out * sizeof(float));
    float *y_ref = (float *)malloc(n_out * sizeof(float));
    srand(42);
    for (int i = 0; i < n_out * n_in; i++) W[i] = ((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < n_in; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f);

    /* F32 reference */
    for (int i = 0; i < n_out; i++) {
        float acc = 0.0f;
        for (int j = 0; j < n_in; j++) acc += W[i*n_in+j] * x[j];
        y_ref[i] = acc;
    }
    int used = -1;
    wubu_bf16_gemv(W, x, y_bf, n_out, n_in, &used);
    CHECK(used == 0 || used == 1, "used_bf16 is 0 or 1");

    float maxerr = 0.0f;
    for (int i = 0; i < n_out; i++) {
        float e = fabsf(y_bf[i] - y_ref[i]);
        if (e > maxerr) maxerr = e;
    }
    /* BF16 round-trip introduces ~1e-2 relative error max; absolute here
     * bounded because inputs in [-0.5,0.5]. Allow generous tolerance. */
    CHECK(maxerr < 0.05f, "BF16 GEMV matches F32 reference within tolerance");
    printf("  (max abs err vs F32 reference: %.5f, path=%s)\n", maxerr,
           used ? "AVX512-BF16" : "F32 fallback");

    /* edge cases */
    CHECK(wubu_bf16_gemv(NULL, x, y_bf, n_out, n_in, NULL) == 0, "null W -> 0");
    CHECK(wubu_bf16_gemv(W, x, y_bf, 0, n_in, NULL) == 0, "n_out<=0 -> 0");

    free(W); free(x); free(y_bf); free(y_ref);
    if (failures == 0) { printf("ALL BF16-GEMV TESTS PASSED\n"); return 0; }
    printf("%d BF16-GEMV TEST(S) FAILED\n", failures);
    return 1;
}

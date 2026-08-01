/* Test: FP8 E4M3/E5M2 emulation (doc B07). Round-trip + dot accuracy. */
#include "wubu_fp8.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

static int approx(float a, float b, float tol) {
    return fabsf(a - b) <= tol * (1.0f + fabsf(b));
}

int main(void) {
    /* 1. Round-trip a spread of values (E4M3 range ~448). */
    float vals[16] = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 10.0f, 100.0f,
                      448.0f, -448.0f, 0.01f, -0.01f, 123.25f, -123.25f,
                      3.5f, -3.5f };
    for (int i = 0; i < 16; i++) {
        uint8_t q = wubu_fp8_e4m3_from_f32(vals[i]);
        float r = wubu_fp8_e4m3_to_f32(q);
        /* E4M3 has 3 mantissa bits -> ~6% relative error near 1.0 */
        assert(approx(r, vals[i], 0.07f) || fabsf(vals[i]) >= 448.0f);
    }

    /* 2. E5M2 round-trip (wider range, has Inf). */
    for (int i = 0; i < 16; i++) {
        uint8_t q = wubu_fp8_e5m2_from_f32(vals[i]);
        float r = wubu_fp8_e5m2_to_f32(q);
        assert(approx(r, vals[i], 0.13f) || fabsf(vals[i]) >= 448.0f);
    }

    /* 3. Quantize/dequantize a vector. */
    const int N = 64;
    float *x = (float *)malloc(N * sizeof(float));
    uint8_t *q = (uint8_t *)malloc(N);
    float *y = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) x[i] = ((i % 7) - 3) * 1.37f;
    assert(wubu_fp8_quantize(x, q, N, 0) == N);
    wubu_fp8_dequantize(q, y, N, 0);
    for (int i = 0; i < N; i++) assert(approx(y[i], x[i], 0.07f));

    /* 4. Dot product: FP8 weights vs F32 activation vs true F32 dot. */
    float *act = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) act[i] = sinf(i * 0.3f);
    uint8_t *w = (uint8_t *)malloc(N);
    wubu_fp8_quantize(x, w, N, 0);
    float d_fp8 = wubu_fp8_dot(w, act, N, 0);
    double d_true = 0.0;
    for (int i = 0; i < N; i++) d_true += (double)x[i] * (double)act[i];
    assert(fabsf(d_fp8 - (float)d_true) <= 0.08f * fabsf((float)d_true) + 1e-3f);

    /* 5. GEMV: matches per-row dot. */
    const int R = 4;
    uint8_t *W = (uint8_t *)malloc((size_t)R * N);
    float *out = (float *)malloc(R * sizeof(float));
    for (int r = 0; r < R; r++)
        for (int i = 0; i < N; i++) W[(size_t)r * N + i] = wubu_fp8_e4m3_from_f32(x[i] * (1.0f + 0.1f * r));
    wubu_fp8_gemv(W, act, out, R, N, 0);
    for (int r = 0; r < R; r++) {
        float expect = wubu_fp8_dot(W + (size_t)r * N, act, N, 0);
        assert(fabsf(out[r] - expect) < 1e-5f);
    }

    free(x); free(q); free(y); free(act); free(w); free(W); free(out);
    printf("ALL FP8 (E4M3/E5M2) TESTS PASSED\n");
    return 0;
}

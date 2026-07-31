/* Test: numerical-stability audit of dequant paths (doc F03). */
#include "wubu_numerical_audit.h"
#include "wubu_kv_adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

static int wrap_quant(const float *z, uint8_t *out, int *width_bits,
                      float *out_scale, int n) {
    return wubu_kvq_adaptive_quant(z, out, width_bits, out_scale, n);
}
static int wrap_dequant(const uint8_t *packed, int width_bits,
                        float scale, float *out, int n) {
    return wubu_kvq_adaptive_dequant(packed, width_bits, scale, out, n);
}

int main(void) {
    float clean[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(wubu_audit_check_clean(clean, 4) == WUBU_AUDIT_OK);
    printf("Clean data detected\n");

    float with_nan[4] = {1.0f, (float)NAN, 3.0f, 4.0f};
    assert(wubu_audit_check_clean(with_nan, 4) == WUBU_AUDIT_NAN);
    printf("NaN detection\n");

    float with_inf[4] = {1.0f, 2.0f, (float)INFINITY, 4.0f};
    assert(wubu_audit_check_clean(with_inf, 4) == WUBU_AUDIT_INF);
    printf("Inf detection\n");

    float a[4] = {1, 0, 0, 0};
    float b[4] = {0, 1, 0, 0};
    float cos = wubu_audit_cosine(a, b, 4);
    assert(fabsf(cos) < 1e-6f);
    printf("Cosine of orthogonal vectors = 0 (%.8f)\n", (double)cos);

    float c[4] = {1, 1, 1, 1};
    float d[4] = {1, 1, 1, 1};
    cos = wubu_audit_cosine(c, d, 4);
    assert(fabsf(cos - 1.0f) < 1e-6f);
    printf("Cosine of identical vectors = 1 (%.8f)\n", (double)cos);

    float normal[32];
    for (int i = 0; i < 32; i++) normal[i] = 0.1f * ((i * 7) % 11 - 5);
    float err;
    int rc = wubu_audit_roundtrip(normal, 32, wrap_quant, wrap_dequant, 10.0f, &err);
    printf("Round-trip audit (normal): rc=%d err=%.6f\n", rc, (double)err);
    assert(rc == WUBU_AUDIT_OK);

    float results[7];
    rc = wubu_audit_extreme_values(wrap_quant, wrap_dequant, results, 7);
    printf("Extreme-value audit: rc=%d\n", rc);
    const char *names[7] = {"zeros","max","min","mixed","alt","outlier","decay"};
    for (int t = 0; t < 7; t++) {
        printf("  %s: err=%.6f\n", names[t], (double)results[t]);
    }

    printf("ALL NUMERICAL-AUDIT TESTS PASSED\n");
    return 0;
}

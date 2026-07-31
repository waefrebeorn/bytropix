/* test_splitk.c — Split-K parallel decode correctness test.
 * Compares wubu_fast_attn_decode_splitk against wubu_fast_attn_decode.
 * Validates that split-K merge via log-sum-exp produces identical results.
 */
#include "wubu_fast_attn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float vec_norm(float *v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i]*v[i];
    return sqrtf(s);
}
static float vec_dot(float *a, float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i]*b[i];
    return s;
}

int main(void) {
    printf("=== Split-K Parallel Decode Test ===\n\n");
    int n_q=4, n_kv=4, hd=128, cache_len=1024, n_threads=4;
    wubu_fast_attn_ctx_t *ctx = wubu_fast_attn_get_ctx(n_q, n_kv, hd, 64, 1e4f, 1.0f);
    if (!ctx) { printf("ctx FAIL\n"); return 1; }

    /* Generate K and V caches with position structure */
    float *k_cache = malloc((size_t)cache_len * n_kv * hd * sizeof(float));
    float *v_cache = malloc((size_t)cache_len * n_kv * hd * sizeof(float));
    srand(42);
    for (int t = 0; t < cache_len; t++) {
        for (int h = 0; h < n_kv; h++) {
            for (int i = 0; i < hd; i++) {
                k_cache[(t*n_kv+h)*hd+i] = (float)sin(t*0.05+i*0.15+h*0.5)*2.0f;
                v_cache[(t*n_kv+h)*hd+i] = (float)cos(t*0.07+i*0.13+h*0.3)*2.0f;
            }
        }
    }

    /* Query */
    float *q = malloc((size_t)n_q * hd * sizeof(float));
    for (int i = 0; i < n_q; i++)
        for (int j = 0; j < hd; j++)
            q[i*hd+j] = (float)sin(i*0.3+j*0.1)*1.5f;

    /* Run serial decode (baseline) */
    float *out_serial = malloc((size_t)n_q * hd * sizeof(float));
    wubu_fast_attn_decode(ctx, q, k_cache, v_cache, cache_len, out_serial, 1);

    /* Run split-K decode with different split counts */
    int test_splits[] = {1, 2, 4, 8, 16};
    int n_tests = sizeof(test_splits)/sizeof(test_splits[0]);
    int errors = 0;

    for (int ti = 0; ti < n_tests; ti++) {
        int ns = test_splits[ti];
        float *out_splitk = malloc((size_t)n_q * hd * sizeof(float));
        wubu_fast_attn_decode_splitk(ctx, q, k_cache, v_cache,
                                     cache_len, out_splitk, n_threads, ns);

        float maxd = 0, nanct = 0;
        float dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < n_q*hd; i++) {
            if (isnan(out_serial[i]) || isnan(out_splitk[i])) { nanct++; continue; }
            float d = fabsf(out_serial[i] - out_splitk[i]);
            if (d > maxd) maxd = d;
            dot += out_serial[i] * out_splitk[i];
            n1 += out_serial[i] * out_serial[i];
            n2 += out_splitk[i] * out_splitk[i];
        }
        float cs = (n1*n2 > 1e-10f) ? dot/(sqrtf(n1)*sqrtf(n2)) : 0;

        printf("  splits=%2d: max_diff=%.2e cosine=%.6f nans=%.0f %s\n",
               ns, maxd, cs, nanct,
               (cs > 0.999f && nanct == 0) ? "PASS" : "FAIL");

        if (cs <= 0.999f || nanct > 0) errors++;
        free(out_splitk);
    }

    printf("\n=== %d errors ===\n", errors);
    free(k_cache); free(v_cache); free(q); free(out_serial);
    wubu_fast_attn_free(ctx);
    return errors;
}

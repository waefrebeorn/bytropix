/*
 * test_fast_attn.c — Correctness + speed benchmark for wubu_fast_attn.
 *
 * Verifies numerical correctness against a naive reference implementation
 * and benchmarks the speedup vs the old per-token-malloc approach at
 * progressively larger context sizes (4K, 16K, 64K, 256K, 512K).
 *
 * WSL2: ~13GB RAM, no GPU. All-CPU AVX2-FMA + OpenMP.
 */
#include "wubu_fast_attn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Naive reference: correct but slow (per-position malloc) */
static void ref_attn(
        const float *q, const float *k_cache, const float *v_cache,
        int n_q, int n_kv, int hd, int cache_len,
        float *out)
{
    int group_sz = n_q / n_kv;
    float scale = 1.0f / sqrtf((float)hd);

    float *scores = (float *)malloc((size_t)cache_len * sizeof(float));

    for (int qh = 0; qh < n_q; qh++) {
        int g = qh / group_sz;
        const float *q_h = q + (size_t)qh * hd;

        /* Compute scores */
        float max_s = -INFINITY;
        for (int t = 0; t < cache_len; t++) {
            const float *k_t = k_cache + (size_t)t * n_kv * hd + (size_t)g * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }

        /* Softmax */
        float sum = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum += scores[t];
        }
        float inv = 1.0f / sum;

        /* Weighted V */
        float *out_h = out + (size_t)qh * hd;
        memset(out_h, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t < cache_len; t++) {
            float w = scores[t] * inv;
            const float *v_t = v_cache + (size_t)t * n_kv * hd + (size_t)g * hd;
            for (int d = 0; d < hd; d++) out_h[d] += w * v_t[d];
        }
    }
    free(scores);
}

int main(void) {
    int n_q = 16, n_kv = 2, hd = 128, n_rot = 64;
    float freq_base = 10000000.0f, scale_factor = 0.25f;

    /* Init fast attention context */
    wubu_fast_attn_ctx_t *ctx = wubu_fast_attn_init(
            n_q, n_kv, hd, 512*1024, n_rot, freq_base, scale_factor);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 1; }

    /* Test at multiple context sizes */
    int ctx_sizes[] = {4096, 16384, 65536, 262144};
    int n_sizes = sizeof(ctx_sizes) / sizeof(ctx_sizes[0]);

    int errors = 0;

    for (int si = 0; si < n_sizes; si++) {
        int cache_len = ctx_sizes[si];
        printf("\n=== Context: %d tokens ===\n", cache_len);

        /* Allocate Q, K, V caches, output */
        float *q = (float *)malloc((size_t)n_q * hd * sizeof(float));
        float *k_cache = (float *)malloc((size_t)cache_len * n_kv * hd * sizeof(float));
        float *v_cache = (float *)malloc((size_t)cache_len * n_kv * hd * sizeof(float));
        float *out_fast = (float *)malloc((size_t)n_q * hd * sizeof(float));
        float *out_ref  = (float *)malloc((size_t)n_q * hd * sizeof(float));

        if (!q || !k_cache || !v_cache || !out_fast || !out_ref) {
            fprintf(stderr, "OOM at ctx %d\n", cache_len);
            break;
        }

        /* Fill with deterministic pseudo-random data */
        for (int i = 0; i < n_q * hd; i++) q[i] = (float)((i * 7 + 13) % 17 - 8) * 0.01f;
        for (int i = 0; i < cache_len * n_kv * hd; i++) {
            k_cache[i] = (float)((i * 3 + 1) % 19 - 9) * 0.01f;
            v_cache[i] = (float)((i * 5 + 7) % 23 - 11) * 0.01f;
        }

        /* Apply RoPE to Q via fast context */
        float *k_new = (float *)malloc((size_t)n_kv * hd * sizeof(float));
        memcpy(k_new, k_cache + (size_t)(cache_len-1) * n_kv * hd, (size_t)n_kv * hd * sizeof(float));
        wubu_fast_attn_rope(ctx, q, k_new, cache_len - 1);
        free(k_new);

        /* === Fast decode === */
        double t0 = now_ms();
        wubu_fast_attn_decode(ctx, q, k_cache, v_cache, cache_len, out_fast, 6);
        double t_fast = now_ms() - t0;

        /* === Reference decode (only for smaller contexts) === */
        double t_ref = -1;
        if (cache_len <= 65536) {  /* ref is too slow above 64K */
            double t1 = now_ms();
            ref_attn(q, k_cache, v_cache, n_q, n_kv, hd, cache_len, out_ref);
            t_ref = now_ms() - t1;

            /* Verify correctness */
            float max_diff = 0.0f;
            for (int i = 0; i < n_q * hd; i++) {
                float d = fabsf(out_fast[i] - out_ref[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("[correctness] max_diff = %.8e %s\n",
                   (double)max_diff, max_diff < 1e-3f ? "PASS" : "FAIL");
            if (max_diff > 1e-3f) errors++;
        } else {
            printf("[correctness] skipped (ref too slow)\n");
        }

        printf("[timing] fast = %.2f ms", t_fast);
        if (t_ref > 0) printf(", ref = %.2f ms, speedup = %.2fx", t_ref, t_ref / t_fast);
        printf("\n");

        /* Bandwidth analysis */
        size_t kv_bytes = (size_t)cache_len * n_kv * hd * 2 * sizeof(float);
        double bw_gbs = (double)kv_bytes / (t_fast / 1000.0) / 1e9;
        printf("[bandwidth] KV read = %.1f MB, achieved = %.1f GB/s\n",
               (double)kv_bytes / 1e6, bw_gbs);

        free(q); free(k_cache); free(v_cache);
        free(out_fast); free(out_ref);
    }

    printf("\n=== Summary: %d errors ===\n", errors);
    wubu_fast_attn_free(ctx);
    return errors;
}
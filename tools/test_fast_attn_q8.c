/* test_fast_attn_q8.c — Q8 KV cache fast decode benchmark + correctness */
#include "wubu_fast_attn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { float d; int8_t qs[32]; } __attribute__((packed)) q8_block;

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void quantize_row_q8(const float *src, q8_block *dst, int n) {
    int n_blocks = (n + 31) / 32;
    for (int b = 0; b < n_blocks; b++) {
        int off = b * 32;
        int cnt = (off + 32 <= n) ? 32 : (n - off);
        float amax = 0.0f;
        for (int i = 0; i < cnt; i++) { float a = fabsf(src[off+i]); if (a > amax) amax = a; }
        float scale = (amax > 1e-8f) ? amax / 127.0f : 1e-8f;
        dst[b].d = scale;
        for (int i = 0; i < cnt; i++) {
            int v = (int)roundf(src[off+i] / scale);
            if (v > 127) v = 127; if (v < -128) v = -128;
            dst[b].qs[i] = (int8_t)v;
        }
        for (int i = cnt; i < 32; i++) dst[b].qs[i] = 0;
    }
}

static void dequant_row_q8(const q8_block *src, float *dst, int n) {
    int n_blocks = (n + 31) / 32;
    for (int b = 0; b < n_blocks; b++) {
        for (int i = 0; i < 32 && b*32+i < n; i++)
            dst[b*32+i] = src[b].d * (float)src[b].qs[i];
    }
}

int main(void) {
    int n_q = 16, n_kv = 2, hd = 128, n_rot = 64;
    wubu_fast_attn_ctx_t *ctx = wubu_fast_attn_init(
            n_q, n_kv, hd, 512*1024, n_rot, 10000000.0f, 0.25f);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 1; }

    int blocks_per_head = (hd + 31) / 32;
    int kv_head_bytes = blocks_per_head * (int)sizeof(q8_block);

    int ctx_sizes[] = {4096, 16384, 65536, 262144};
    int n_sizes = 4;
    int errors = 0;

    for (int si = 0; si < n_sizes; si++) {
        int cache_len = ctx_sizes[si];
        printf("\n=== Q8 Context: %d tokens ===\n", cache_len);

        float *q = malloc((size_t)n_q * hd * sizeof(float));
        float *k_f32 = malloc((size_t)cache_len * n_kv * hd * sizeof(float));
        float *v_f32 = malloc((size_t)cache_len * n_kv * hd * sizeof(float));
        float *out_q8 = malloc((size_t)n_q * hd * sizeof(float));
        float *out_f32 = malloc((size_t)n_q * hd * sizeof(float));

        /* Q8 caches */
        q8_block *k_q8 = malloc((size_t)cache_len * n_kv * kv_head_bytes);
        q8_block *v_q8 = malloc((size_t)cache_len * n_kv * kv_head_bytes);

        if (!q || !k_f32 || !v_f32 || !out_q8 || !out_f32 || !k_q8 || !v_q8) {
            fprintf(stderr, "OOM at ctx %d\n", cache_len); break;
        }

        /* Fill with deterministic data */
        for (int i = 0; i < n_q * hd; i++) q[i] = (float)((i*7+13)%17-8)*0.01f;
        for (int i = 0; i < cache_len*n_kv*hd; i++) {
            k_f32[i] = (float)((i*3+1)%19-9)*0.01f;
            v_f32[i] = (float)((i*5+7)%23-11)*0.01f;
        }

        /* Quantize to Q8 */
        for (int t = 0; t < cache_len; t++) {
            for (int g = 0; g < n_kv; g++) {
                quantize_row_q8(k_f32 + (size_t)t*n_kv*hd + g*hd,
                                k_q8 + (size_t)t*n_kv*blocks_per_head + g*blocks_per_head, hd);
                quantize_row_q8(v_f32 + (size_t)t*n_kv*hd + g*hd,
                                v_q8 + (size_t)t*n_kv*blocks_per_head + g*blocks_per_head, hd);
            }
        }

        /* Apply RoPE to Q */
        float *k_new = malloc((size_t)n_kv * hd * sizeof(float));
        memcpy(k_new, k_f32 + (size_t)(cache_len-1)*n_kv*hd, (size_t)n_kv*hd*sizeof(float));
        wubu_fast_attn_rope(ctx, q, k_new, cache_len - 1);
        free(k_new);

        /* F32 baseline */
        double t0 = now_ms();
        wubu_fast_attn_decode(ctx, q, k_f32, v_f32, cache_len, out_f32, 6);
        double t_f32 = now_ms() - t0;

        /* Q8 decode */
        double t1 = now_ms();
        wubu_fast_attn_decode_q8(ctx, q, k_q8, v_q8, cache_len, out_q8, 6);
        double t_q8 = now_ms() - t1;

        /* Correctness: Q8 should be close to F32 (quantization noise) */
        float max_diff = 0.0f;
        for (int i = 0; i < n_q*hd; i++) {
            float d = fabsf(out_q8[i] - out_f32[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("[correctness] F32 vs Q8 max_diff = %.6e %s\n",
               (double)max_diff, max_diff < 0.05f ? "PASS" : "FAIL");
        if (max_diff > 0.05f) errors++;

        /* Bandwidth comparison */
        size_t f32_bytes = (size_t)cache_len * n_kv * hd * 4 * 2;
        size_t q8_bytes = (size_t)cache_len * n_kv * kv_head_bytes * 2;
        printf("[timing] F32=%.2fms (%.1f GB/s), Q8=%.2fms (%.1f GB/s), speedup=%.2fx\n",
               t_f32, (double)f32_bytes/(t_f32/1000.0)/1e9,
               t_q8,  (double)q8_bytes/(t_q8/1000.0)/1e9,
               t_f32 / t_q8);
        printf("[bandwidth] F32 reads %.1f MB, Q8 reads %.1f MB (%.1f%% less)\n",
               (double)f32_bytes/1e6, (double)q8_bytes/1e6,
               100.0*(1.0 - (double)q8_bytes/f32_bytes));

        free(q); free(k_f32); free(v_f32);
        free(out_q8); free(out_f32);
        free(k_q8); free(v_q8);
    }

    printf("\n=== Summary: %d errors ===\n", errors);
    wubu_fast_attn_free(ctx);
    return errors;
}
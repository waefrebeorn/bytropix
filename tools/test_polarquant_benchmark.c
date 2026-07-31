/* test_polarquant_nobreakmark.c — Zero-malloc decode benchmark
 * Measures cycles per token for malloc-free vs the theoretical malloc path.
 * Tests at 256K context scale.
 */
#include "wubu_polarquant.h"
#include "wubu_polar_pso.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <x86intrin.h>

int main(void) {
    int d = 128, bits = 8;
    wubu_polarquant_t pq;
    wubu_polarquant_init(&pq, d, 1, 1.0f, (float)bits);
    int storage = wubu_polarquant_storage_bytes(&pq, d);

    printf("=== Zero-Malloc Decode Benchmark ===\n");
    printf("d=%d, bits=%d, storage=%d bytes/token\n\n", d, bits, storage);

    /* Prepare test data: encode 10000 tokens */
    int n_tokens = 10000;
    int buf_sz = storage + 64;
    float *packed = malloc((size_t)n_tokens * buf_sz * sizeof(float));
    float *orig_all = malloc((size_t)n_tokens * d * sizeof(float));
    int *packed_bytes = malloc((size_t)n_tokens * sizeof(int));

    srand(42);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < d; j++) {
            orig_all[i*d + j] = (float)((rand() % 200) - 100) * 0.01f;
        }
        int ob = buf_sz;
        float *dst = &packed[i * buf_sz];
        wubu_polarquant_quantize_kv(&pq, &orig_all[i*d], dst, &ob);
        packed_bytes[i] = ob;
    }
    printf("Encoded %d tokens (256K context = %.0f tokens)\n\n",
           n_tokens, (float)n_tokens / 256.0f * 256.0f);

    /* Benchmark decode + fused dot */
    float q[128];
    for (int i = 0; i < d; i++) q[i] = (float)((rand() % 200) - 100) * 0.01f;

    /* Warm up */
    for (int i = 0; i < 100; i++) {
        float k[128];
        wubu_polarquant_dequantize_kv(&pq, &packed[i * buf_sz],
                                      packed_bytes[i], k, d);
    }

    /* Measure decode cycles */
    int n_iters = 1000;
    uint64_t start, end;

    /* Decode-only benchmark */
    start = __rdtsc();
    volatile float sink = 0;
    for (int iter = 0; iter < n_iters; iter++) {
        int idx = iter % n_tokens;
        float k[128];
        wubu_polarquant_dequantize_kv(&pq, &packed[idx * buf_sz],
                                      packed_bytes[idx], k, d);
        sink += k[0]; /* prevent dead code elimination */
    }
    end = __rdtsc();
    float decode_cycles = (float)(end - start) / n_iters;
    printf("Decode-only: %.0f cycles/token\n", decode_cycles);
    printf("  At 256K tokens: %.2f ms\n", decode_cycles * 262144.0f / 3.5e9f * 1000);
    printf("  At 512K tokens: %.2f ms\n", decode_cycles * 524288.0f / 3.5e9f * 1000);

    /* Fused dot benchmark */
    start = __rdtsc();
    for (int iter = 0; iter < n_iters; iter++) {
        int idx = iter % n_tokens;
        volatile float dot = wubu_polarquant_fused_dot(
            &pq, q, &packed[idx * buf_sz], packed_bytes[idx]);
        sink += dot;
    }
    end = __rdtsc();
    float dot_cycles = (float)(end - start) / n_iters;
    printf("\nFused decode+dot: %.0f cycles/token\n", dot_cycles);
    printf("  At 256K tokens: %.2f ms\n", dot_cycles * 262144.0f / 3.5e9f * 1000);
    printf("  At 512K tokens: %.2f ms\n", dot_cycles * 524288.0f / 3.5e9f * 1000);

    /* Compare against raw F32 dot product */
    float *k_f32 = malloc((size_t)n_tokens * d * sizeof(float));
    /* Copy decoded K for F32 baseline */
    for (int i = 0; i < 100; i++) {
        wubu_polarquant_dequantize_kv(&pq, &packed[i * buf_sz],
                                      packed_bytes[i], &k_f32[i*d], d);
    }

    start = __rdtsc();
    for (int iter = 0; iter < n_iters; iter++) {
        int idx = iter % 100;
        const float *k = &k_f32[idx * d];
        float dot = 0;
        for (int j = 0; j < d; j++) dot += q[j] * k[j];
        sink += dot;
    }
    end = __rdtsc();
    float f32_dot_cycles = (float)(end - start) / n_iters;
    printf("\nF32 baseline dot: %.0f cycles/vector\n", f32_dot_cycles);
    printf("PolarQuant overhead: %.1fx (decode+normalize vs raw dot)\n",
           dot_cycles / f32_dot_cycles);

    extern void wubu_pso_set_context(const wubu_polar_pso_t *pso);
    wubu_polar_pso_t pso;
    wubu_polar_pso_init(&pso, &pq, bits, d);

    /* Warm up PSO */
    wubu_pso_set_context(&pso);
    for (int i = 0; i < 100; i++) {
        float k[128];
        wubu_pso_decode((const uint8_t *)&packed[i * buf_sz],
                        packed_bytes[i], k, d);
    }

    start = __rdtsc();
    for (int iter = 0; iter < n_iters; iter++) {
        int idx = iter % n_tokens;
        float k[128];
        wubu_pso_decode((const uint8_t *)&packed[idx * buf_sz],
                        packed_bytes[idx], k, d);
        sink += k[0];
    }
    end = __rdtsc();
    float pso_cycles = (float)(end - start) / n_iters;
    printf("\nPSO decode (trig tables + scratch): %.0f cycles/token\n", pso_cycles);
    printf("  Speedup vs base decode: %.1fx\n", decode_cycles / pso_cycles);
    printf("  At 512K: %.2f ms\n", pso_cycles * 524288.0f / 3.5e9f * 1000);

    wubu_pso_set_context(NULL);
    wubu_polar_pso_free(&pso);

    /* Bandwidth calculation at 512K */
    double f32_mb = (double)d * 4 * 2 * 524288 / 1e6;
    double pq_mb = (double)storage * 2 * 524288 / 1e6;
    printf("\n--- Bandwidth at 512K context ---\n");
    printf("F32 KV: %.1f MB (2 streams × %d dim × 4B × 512K)\n", f32_mb, d);
    printf("PQ KV:  %.1f MB (%d bytes/token × 2 streams × 512K)\n", pq_mb, storage);
    printf("Compression: %.1fx\n", f32_mb / pq_mb);

    /* Memory bandwidth required */
    double decode_ms = decode_cycles * 524288.0f / 3.5e9f * 1000;
    double bw_gbs = pq_mb / (decode_ms / 1000.0f);
    printf("Decode bandwidth: %.1f GB/s (PQ stream)\n", bw_gbs);
    printf("F32 bandwidth:    %.1f GB/s\n", f32_mb / (decode_ms / 1000.0f));

    (void)sink;
    free(packed); free(orig_all); free(packed_bytes); free(k_f32);
    wubu_polarquant_free(&pq);

    printf("\n=== Benchmark done ===\n");
    return 0;
}

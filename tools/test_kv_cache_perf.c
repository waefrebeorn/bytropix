/*
 * test_kv_cache_perf.c
 *
 * Performance regression harness for KV cache operations.
 * Measures:
 *   - raw alloc + zero time
 *   - F16 write head throughput
 *   - KIVI quant/dequant throughput
 *   - round-trip collision latency
 *
 * Build: make test_kv_cache_perf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "wubu_kvcache_quant.h"
#include "wubu_kv_styx.h"

#define N_TRIALS 1000
#define TOKENS 128
#define HEAD_DIM 256
#define N_HEADS 16
#define LAYERS 4

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float rand_f(void) {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static void bench_alloc(size_t trials) {
    double t0 = now_sec();
    for (size_t i = 0; i < trials; i++) {
        void *p = malloc(TOKENS * HEAD_DIM * sizeof(float) * 2);
        memset(p, 0, TOKENS * HEAD_DIM * sizeof(float) * 2);
        free(p);
    }
    double dt = now_sec() - t0;
    double mb = (double)(TOKENS * HEAD_DIM * 2 * sizeof(float)) / (1024 * 1024);
    printf("alloc        : %6.2f MB/call, %8.1f calls/s\n", mb, trials / dt);
}

static void bench_kiviquant(size_t trials) {
    int tokens = TOKENS, hd = HEAD_DIM;
    size_t n = (size_t)tokens * hd;
    float *f = malloc(n * sizeof(float));
    uint8_t *q = malloc(n * sizeof(uint8_t) + tokens * sizeof(float));
    for (size_t i = 0; i < n; i++) f[i] = rand_f();

    double t0 = now_sec();
    for (size_t i = 0; i < trials; i++) {
        float scale = 0.0f;
        wubu_kvq_kivi_quant_V(f, q, &scale, tokens, hd);
        wubu_kvq_kivi_dequant_V(q, &scale, f, tokens, hd);
    }
    double dt = now_sec() - t0;
    printf("KIVI r/t     : %8.1f tok/s (av=%.1f us/tok)\n",
           (trials * tokens) / dt, (dt * 1e6) / (double)(trials * tokens));

    free(f);
    free(q);
}

static void bench_styxreg(size_t trials) {
    double t0 = now_sec();
    for (size_t i = 0; i < trials; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/n/kv/perf_%04zu", i);
        void dummy = 0;
        wubu_kv_styx_register(path, &dummy, 1024);
    }
    double dt = now_sec() - t0;
    printf("KV-styx reg  : %8.1f reg/s\n", trials / dt);
}

static void bench_rantokens(size_t trials) {
    int tokens = TOKENS, hd = HEAD_DIM;
    size_t n = (size_t)tokens * hd;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) a[i] = rand_f();

    double t0 = now_sec();
    for (size_t i = 0; i < trials; i++) {
        memcpy(b, a, n * sizeof(float));
        float s = 0.0f;
        for (size_t j = 0; j < n; j++) s += b[j];
        (void)s;
    }
    double dt = now_sec() - t0;
    size_t bytes = n * sizeof(float) * 2;
    double gib = (bytes * trials) / (1024.0 * 1024.0 * 1024.0);
    printf("memcpy+reduce: %8.1f gib/s\n", gib / dt);
    free(a);
    free(b);
}

static float cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return na > 0 && nb > 0 ? (float)(dot / (sqrt(na) * sqrt(nb))) : 0.0f;
}

static int roundtrip_oracle(void) {
    int tokens = 8, hd = HEAD_DIM;
    size_t n = (size_t)tokens * hd;
    float *f = malloc(n * sizeof(float));
    uint8_t *q = malloc(n * sizeof(uint8_t) + tokens * sizeof(float));
    for (size_t i = 0; i < n; i++) f[i] = rand_f();

    float scale = 0.0f;
    wubu_kvq_kivi_quant_V(f, q, &scale, tokens, hd);
    float *deq = malloc(n * sizeof(float));
    wubu_kvq_kivi_dequant_V(q, &scale, deq, tokens, hd);

    float sim = cosine(f, deq, n);
    printf("round-trip   : cosine %.6f ", sim);
    free(f); free(q); free(deq);
    return sim >= 0.997f;
}

int main(void) {
    srand(42);
    printf("KV cache perf regression (%d trials each)\n", N_TRIALS);
    printf("HEAD_DIM=%d  TOKENS=%d  N_HEADS=%d\n\n", HEAD_DIM, TOKENS, N_HEADS);

    bench_alloc(N_TRIALS);
    bench_rantokens(N_TRIALS);
    bench_kiviquant(N_TRIALS * 50);
    bench_styxreg(1000);

    printf("\n");
    return roundtrip_oracle() ? 0 : 1;
}

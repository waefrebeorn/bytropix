/*
 * test_kv_bw.c -- KV-cache bandwidth/stress harness.
 *
 * Build:   make test_kv_bw
 * Run:     ./test_kv_bw
 *
 * Measures three bounded, in-bounds workloads:
 *   1. F16 write-through (memcpy within allocated buffer)
 *   2. KIVI one-head quant/dequant loop
 *   3. 16 GiB mmap seq-read stress
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>
#include "wubu_kvcache_quant.h"

#define HEAD_DIM 256
#define N_HEADS  16
#define LAYERS   4
#define TOKENS   128
#define TRIALS   1024

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float rand_f(void) { return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }

static void bench_f16(size_t trials) {
    size_t bytes = (size_t)LAYERS * TOKENS * N_HEADS * HEAD_DIM * sizeof(float);
    float *cache = malloc(bytes);
    float *buf   = malloc((size_t)N_HEADS * HEAD_DIM * sizeof(float));
    if (!cache || !buf) { printf("F16 alloc failed\n"); free(cache); free(buf); return; }
    for (size_t i = 0; i < bytes / sizeof(float); i++) cache[i] = rand_f();

    double t0 = now_sec();
    for (size_t t = 0; t < trials; t++) {
        /* write into first half, read from second half -- bounded */
        memcpy(cache, cache + bytes / sizeof(float) / 2, bytes / 2);
    }
    double dt = now_sec() - t0;
    double gib = (bytes * trials) / (1024.0 * 1024.0 * 1024.0);
    printf("F16 write-through: %6.2f GiB/s  (%d trials, %.3fs)\n",
           gib / dt, (int)trials, dt);
    free(cache); free(buf);
}

static void bench_kivi_block(size_t trials) {
    size_t block = (size_t)N_HEADS * HEAD_DIM;
    float *a    = malloc(block * sizeof(float));
    uint8_t *q  = malloc(block + N_HEADS * sizeof(float));
    if (!a || !q) { printf("KIVI alloc failed\n"); free(a); free(q); return; }
    for (size_t i = 0; i < block; i++) a[i] = rand_f();
    float s = 0.0f;

    double t0 = now_sec();
    for (size_t t = 0; t < trials; t++) {
        wubu_kvq_kivi_quant_V(a, q, &s, 1, HEAD_DIM);
        wubu_kvq_kivi_dequant_V(q, &s, a, 1, HEAD_DIM);
    }
    double dt = now_sec() - t0;

    int n_tok = (int)(trials * N_HEADS * LAYERS * TOKENS);
    double gib_s = (n_tok * block * sizeof(float)) / (1024.0 * 1024.0 * 1024.0 * dt);
    printf("KIVI one-head q/d: %6.2f GiB/s  (%d trials, %.3fs)\n",
           gib_s, (int)trials, dt);
    free(a); free(q);
}

static void bench_mmap_stress(void) {
    size_t sz = 16ULL * 1024 * 1024 * 1024;
    FILE *f = fopen("/dev/zero", "rb");
    if (!f) { printf("mmap stress: SKIP\n"); return; }
    void *p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fileno(f), 0);
    fclose(f);
    if (p == MAP_FAILED) { printf("mmap stress: FAILED\n"); return; }

    volatile uint64_t sum = 0;
    uint64_t *u = (uint64_t *)p;
    double t0 = now_sec();
    for (size_t i = 0; i < sz / sizeof(uint64_t); i += 4096 / sizeof(uint64_t))
        sum += u[i];
    double dt = now_sec() - t0;
    munmap(p, sz);
    printf("mmap 16GiB read:   %5.1f GiB/s  checksum=%llu\n",
           (sz / (1024.0*1024.0*1024.0)) / dt, (unsigned long long)sum);
    (void)sum;
}

int main(void) {
    srand(42);
    printf("KV cache BW harness\nHEAD_DIM=%d N_HEADS=%d LAYERS=%d TOKENS=%d\n\n",
           HEAD_DIM, N_HEADS, LAYERS, TOKENS);
    bench_f16(TRIALS);
    bench_kivi_block(TRIALS * 20);
    bench_mmap_stress();
    printf("\nDone\n");
    return 0;
}

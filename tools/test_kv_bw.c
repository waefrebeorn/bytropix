#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_model.h"
#include "wubu_kvcache_quant.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float rand_f(void) { return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }

static int64_t kv_alloc_bytes(int scheme, int64_t n_elems) {
    if (scheme == WUBU_KV_KIVI)
        return ((n_elems * 1) / 1) + (n_elems / 512) * 4 + 64;
    return n_elems * sizeof(float);
}

static void bench_scheme(const char *name, int scheme, int layers, int heads, int hd, int tok) {
    int64_t per_layer = (int64_t)2 * heads * tok * hd; /* K+V */
    int64_t bytes = kv_alloc_bytes(scheme, per_layer);
    size_t total = (size_t)bytes * layers;
    void *cache = malloc(total);
    if (!cache) { printf("%s alloc failed\n", name); return; }
    memset(cache, 0xAA, total);
    float *buf = malloc((size_t)heads * hd * sizeof(float));
    for (int i = 0; i < heads * hd; i++) buf[i] = rand_f();

    double t0 = now();
    for (int l = 0; l < layers; l++) {
        for (int t = 0; t < tok; t++) {
            for (int h = 0; h < heads * 2; h++) {
                int64_t off = (int64_t)l * per_layer + (int64_t)t * heads * hd + h * hd;
                kv_cache_write_head(cache, off, buf + h * hd, hd, scheme);
            }
        }
    }
    double dt = now() - t0;
    size_t written = (size_t)((int64_t)layers * tok * heads * 2 * hd * sizeof(float));
    double gib = written / (1024.0 * 1024.0 * 1024.0);
    printf("scheme=%-6s layers=%d heads=%d hd=%d tok=%d bw=%.2f GiB/s time=%.3fs\n",
           name, layers, heads, hd, tok, gib / dt, dt);
    free(cache); free(buf);
}

int main(void) {
    srand(42);
    printf("KV cache bandwidth harness\n");
    printf("ref: alloc+write over %d sim layers (K+V)\n\n", 4);
    bench_scheme("F16",    WUBU_KV_F16,  4, 32, 256, 512);
    bench_scheme("KIVI",   WUBU_KV_KIVI, 4, 32, 256, 512);
    bench_scheme("KIVI",   WUBU_KV_KIVI, 4, 16, 256, 1024);
    return 0;
}

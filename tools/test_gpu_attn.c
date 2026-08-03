/* test_gpu_attn.c -- the GPU attention tile vs the CPU reference (the
 * bp's exact hybrid GQA math): the causal + local-window mask, the
 * 1/sqrt(64) scale, the softmax, the @v. The FD oracle doctrine: the
 * GPU must match the CPU loop to 1e-3. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "gpu_barun.h"

#define HEADS 7
#define DIM 64

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void cpu_attn(float *acc, const float *q, const float *k,
                     const float *v, int seq, int local_win, int is_full)
{
    const int D = HEADS * DIM;
    for (int s = 0; s < seq; s++) {
        float *acc_s = acc + (size_t)s * D;
        memset(acc_s, 0, D * sizeof(float));
        for (int h = 0; h < HEADS; h++) {
            const float *qrow = q + (size_t)s * D + (size_t)h * DIM;
            float maxv = -1e30f;
            int lo = is_full ? 0 : (s > local_win ? s - local_win + 1 : 0);
            int kv_n = 0;
            float probs[512];
            for (int t = lo; t <= s; t++) {
                const float *krow = k + (size_t)t * DIM;
                float dot = 0;
                for (int i = 0; i < DIM; i++) dot += qrow[i] * krow[i];
                dot *= 1.0f / sqrtf((float)DIM);
                if (dot > maxv) maxv = dot;
                probs[kv_n++] = dot;
            }
            float sum = 0;
            for (int i = 0; i < kv_n; i++) {
                probs[i] = expf(probs[i] - maxv);
                sum += probs[i];
            }
            for (int i = 0; i < kv_n; i++) probs[i] /= sum;
            for (int i = 0; i < kv_n; i++) {
                const float *vrow = v + (size_t)(lo + i) * DIM;
                for (int d = 0; d < DIM; d++)
                    acc_s[h * DIM + d] += probs[i] * vrow[d];
            }
        }
    }
}

int main(void)
{
    if (!gpu_barun_init()) { printf("SKIP (no CUDA device)\n"); return 0; }
    int cases[][2] = { {64, 0}, {64, 1}, {256, 0}, {256, 1}, {512, 0}, {512, 1} };
    srand(11);
    for (int ci = 0; ci < 6; ci++) {
        int seq = cases[ci][0], is_full = cases[ci][1];
        const int D = HEADS * DIM;
        float *q = malloc((size_t)seq * D * 4);
        float *k = malloc((size_t)seq * DIM * 4);
        float *v = malloc((size_t)seq * DIM * 4);
        float *ref = malloc((size_t)seq * D * 4);
        float *gpu = malloc((size_t)seq * D * 4);
        for (int i = 0; i < seq * D; i++) q[i] = (float)((rand() % 2000) - 1000) / 100.0f;
        for (int i = 0; i < seq * DIM; i++) { k[i] = (float)((rand() % 2000) - 1000) / 100.0f;
                                              v[i] = (float)((rand() % 2000) - 1000) / 100.0f; }
        cpu_attn(ref, q, k, v, seq, 256, is_full);
        double t0 = now_s();
        int ok = gpu_barun_attn(gpu, q, k, v, seq, HEADS, DIM, 256, is_full);
        double t1 = now_s();
        double maxd = 0, sumr = 0;
        for (int i = 0; i < seq * D; i++) {
            double d = fabs((double)gpu[i] - (double)ref[i]);
            if (d > maxd) maxd = d;
            sumr += fabs(ref[i]);
        }
        int pass = ok == 1 && maxd < 1e-3 * (sumr / (seq * D)) * 100;
        printf("  seq=%d full=%d rc=%d gpu %.2fms  max|gpu-cpu|=%.3e %s\n",
               seq, is_full, ok, (t1 - t0) * 1000.0, maxd,
               pass ? "OK" : "FAIL");
        free(q); free(k); free(v); free(ref); free(gpu);
        if (!pass) return 1;
    }
    printf("ALL GPU ATTENTION TESTS PASSED -- the tile matches the CPU oracle\n");
    return 0;
}

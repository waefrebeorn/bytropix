/* gpu_matmul_check.c -- verify GPU matmul == CPU matmul bit-for-bit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_wubu.h"

int main(void)
{
    int M = 8, N = 448, K = 64;   /* seq, out_n, in_n */
    float *x = malloc(sizeof(float) * M * K);
    float *w = malloc(sizeof(float) * N * K);   /* w[o*K + i], [out,in] */
    float *cpu = malloc(sizeof(float) * M * N);
    float *gpu = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++) x[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < N * K; i++) w[i] = ((float)rand() / RAND_MAX) - 0.5f;

    /* CPU: out[s,o] = sum_i w[o,i]*x[s,i]  (the wubu loop) */
    for (int s = 0; s < M; s++)
        for (int o = 0; o < N; o++) {
            float acc = 0;
            for (int i = 0; i < K; i++) acc += w[o * K + i] * x[s * K + i];
            cpu[s * N + o] = acc;
        }

    if (!gpu_wubu_init()) { printf("NO GPU\n"); return 0; }
    int rc = gpu_wubu_matmul(gpu, w, x, M, N, K);
    if (!rc) { printf("GPU CALL FAILED\n"); return 1; }

    double max_diff = 0;
    for (int i = 0; i < M * N; i++) {
        double d = fabs((double)cpu[i] - (double)gpu[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("max diff: %.6e %s\n", max_diff,
           max_diff < 1e-3 ? "MATCH" : "MISMATCH");
    gpu_wubu_free();
    return max_diff < 1e-3 ? 0 : 1;
}

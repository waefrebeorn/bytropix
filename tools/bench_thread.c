#include "thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    thread_pool_t *pool = thread_pool_global();
    if (!pool) { printf("FAIL: pool NULL\n"); return 1; }
    printf("Threads: %d\n", thread_pool_count(pool));

    // Small test: [M=16, K=32] @ [K=32, N=64]
    int M=16, K=32, N=64;
    float A[16*32], B[64*32], C[16*64], C_ref[16*64];
    for (int i=0; i<M*K; i++) A[i] = (float)(i % 7);
    for (int i=0; i<N*K; i++) B[i] = (float)(i % 11);

    // Single-thread ref
    for (int m=0; m<M; m++)
        for (int n=0; n<N; n++) {
            float sum = 0;
            for (int k=0; k<K; k++) sum += A[m*K+k] * B[n*K+k];
            C_ref[m*N+n] = sum;
        }

    memset(C, 0, M*N*sizeof(float));
    thread_pool_matmul_nt(pool, M, N, K, A, B, C);

    double md = 0;
    for (int i=0; i<M*N; i++) {
        double d = fabs((double)C[i] - (double)C_ref[i]);
        if (d > md) md = d;
    }
    printf("Small test:  max diff=%e  %s\n", md, md<1e-6?"PASS":"FAIL");

    // Medium test: [M=512, K=1024] @ [K=1024, N=2048] = 1 GFLOP
    M=512; K=1024; N=2048;
    float *A2 = (float*)malloc(M*K*sizeof(float));
    float *B2 = (float*)malloc(N*K*sizeof(float));
    float *C2 = (float*)calloc(M*N, sizeof(float));
    float *C2_ref = (float*)calloc(M*N, sizeof(float));

    srand(42);
    for (int i=0; i<M*K; i++) A2[i] = (float)(rand()%100)/100.0f;
    for (int i=0; i<N*K; i++) B2[i] = (float)(rand()%100)/100.0f;

    double t0 = now_sec();
    for (int m=0; m<M; m++)
        for (int n=0; n<N; n++) {
            float sum = 0;
            for (int k=0; k<K; k++) sum += A2[m*K+k] * B2[n*K+k];
            C2_ref[m*N+n] = sum;
        }
    double t_single = now_sec() - t0;

    t0 = now_sec();
    thread_pool_matmul_nt(pool, M, N, K, A2, B2, C2);
    double t_mt = now_sec() - t0;

    md = 0;
    for (int i=0; i<M*N; i++) {
        double d = fabs((double)C2[i] - (double)C2_ref[i]);
        if (d > md) md = d;
    }
    double gflop = 2.0 * M * N * K / 1e9;
    printf("Medium test: max diff=%e  %s\n", md, md<1e-5?"PASS":"FAIL");
    printf("  Single: %.3fs (%.1f GFLOP/s)\n", t_single, gflop/t_single);
    printf("  Multi:  %.3fs (%.1f GFLOP/s)\n", t_mt, gflop/t_mt);
    printf("  Speedup: %.1fx\n", t_single/t_mt);

    free(A2); free(B2); free(C2); free(C2_ref);
    printf("DONE\n");
    return 0;
}

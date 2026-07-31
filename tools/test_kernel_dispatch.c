/* test_kernel_dispatch.c — hardware-agnostic kernel dispatch tests */
#include "wubu_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPS 1e-5f
#define TM 4
#define TK 8
#define TN 6

static void nref_softmax(float *logits, int M, int N) {
    for (int i = 0; i < M; i++) {
        float *row = logits + i * N;
        float maxv = row[0];
        for (int j = 1; j < N; j++) if (row[j] > maxv) maxv = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) { row[j] = expf(row[j] - maxv); sum += row[j]; }
        for (int j = 0; j < N; j++) row[j] /= sum;
    }
}

static float max_diff(const float *a, const float *b, int n) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > md) md = d; }
    return md;
}

static float *gemm_ref(const float *A, const float *B, int M, int K, int N) {
    float *C = (float *)calloc((size_t)M * N, sizeof(float));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
    return C;
}

static float *gemv_ref(const float *A, const float *x, int M, int K) {
    float *y = (float *)calloc(M, sizeof(float));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            y[i] += A[i*K+k] * x[k];
    return y;
}

static void rmsnorm_ref(float *x, const float *gamma, const float *beta,
                           int M, int d, float eps) {
    for (int i = 0; i < M; i++) {
        float *row = x + i * d;
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) sum_sq += row[j] * row[j];
        float rsqrt = 1.0f / sqrtf(sum_sq / (float)d + eps);
        for (int j = 0; j < d; j++) row[j] = row[j] * rsqrt * gamma[j] + beta[j];
    }
}

int main(void) {
    int errors = 0;

    printf("[1] init = %d\n", wubu_kernel_init());

    /* 2. GEMM */
    {
        float A[TM*TK], B[TK*TN], C[TM*TN];
        float *ref = gemm_ref(A, B, TM, TK, TN);
        for (int i = 0; i < TM*TK; i++) A[i] = (float)(i % 7 - 3) * 0.1f;
        for (int i = 0; i < TK*TN; i++) B[i] = (float)(i % 5 - 2) * 0.2f;
        wubu_kernel_gemm_scalar(A, B, C, TM, TK, TN, 0.0f);
        float md = max_diff(C, ref, TM*TN);
        printf("[2] GEMM max_diff = %.8e %s\n", (double)md, md < EPS ? "PASS" : "FAIL");
        if (md > EPS) errors++;
        free(ref);
    }

    /* 3. GEMV */
    {
        float A[TM*TK], x[TK], y[TM];
        float *ref = gemv_ref(A, x, TM, TK);
        for (int i = 0; i < TM*TK; i++) A[i] = (float)(i % 11 - 5) * 0.1f;
        for (int i = 0; i < TK; i++) x[i] = (float)(i - TK/2) * 0.05f;
        wubu_kernel_gemv_scalar(A, x, y, TM, TK);
        float md = max_diff(y, ref, TM);
        printf("[3] GEMV max_diff = %.8e %s\n", (double)md, md < EPS ? "PASS" : "FAIL");
        if (md > EPS) errors++;
        free(ref);
    }

    /* 4. Softmax */
    {
        float logits[TM*TN], ref[TM*TN];
        for (int i = 0; i < TM*TN; i++) logits[i] = (float)(i % 13 - 6) * 0.5f;
        memcpy(ref, logits, TM*TN*sizeof(float));
        wubu_kernel_softmax_scalar(logits, TM, TN);
        nref_softmax(ref, TM, TN);
        float md = max_diff(logits, ref, TM*TN);
        printf("[4] Softmax max_diff = %.8e %s\n", (double)md, md < EPS ? "PASS" : "FAIL");
        if (md > EPS) errors++;
        /* Row sums = 1.0 */
        for (int i = 0; i < TM; i++) {
            float sum = 0.0f;
            for (int j = 0; j < TN; j++) sum += logits[i*TN+j];
            if (fabsf(sum - 1.0f) > EPS) { fprintf(stderr, "  FAIL row %d sum=%.8f\n", i, sum); errors++; }
        }
    }

    /* 5. RMSNorm */
    {
        int d = 16;
        float x[TM*d], gamma[d], beta[d], ref_x[TM*d];
        for (int i = 0; i < TM*d; i++) x[i] = (float)(i % 7 - 3) * 0.3f;
        for (int i = 0; i < d; i++) { gamma[i] = 1.0f; beta[i] = 0.0f; }
        memcpy(ref_x, x, TM*d*sizeof(float));
        wubu_kernel_rmsnorm_scalar(x, gamma, beta, TM, d, 1e-6f);
        rmsnorm_ref(ref_x, gamma, beta, TM, d, 1e-6f);
        float md = max_diff(x, ref_x, TM*d);
        printf("[5] RMSNorm max_diff = %.8e %s\n", (double)md, md < EPS ? "PASS" : "FAIL");
        if (md > EPS) errors++;
    }

    /* 6. Quantize/dequantize */
    {
        int bits = 8;
        float fp32[TM*TK], scales[TM], restored[TM*TK];
        int8_t q[TM*TK];
        for (int i = 0; i < TM*TK; i++) fp32[i] = (float)(i % 17 - 8) * 0.1f;
        wubu_kernel_quantize_scalar(fp32, q, scales, TM, TK, bits);
        wubu_kernel_dequantize_scalar(q, scales, NULL, restored, TM, TK, bits);
        float md = max_diff(fp32, restored, TM*TK);
        printf("[6] Q/DQ %d-bit max_diff = %.8e %s\n", bits, (double)md, md < 0.1f ? "PASS" : "FAIL");
        if (md > 0.1f) errors++;
    }

    /* 7. Backend registration */
    {
        wubu_kernel_backend_t mock = {
            .id = WUBU_BACKEND_BLAS, .name = "mock-blas",
            .gemm = wubu_kernel_gemm_scalar, .supports = NULL, .next = NULL
        };
        int rc = wubu_kernel_register(WUBU_BACKEND_BLAS, "mock-blas", &mock);
        printf("[7] Register = %d\n", rc);
        if (rc != 0) errors++;
        else {
            const char *name = wubu_kernel_active_backend(WUBU_KERN_GEMM);
            printf("    active GEMM backend: %s\n", name);
            if (strcmp(name, "mock-blas") != 0) { fprintf(stderr, "  FAIL\n"); errors++; }
            wubu_kernel_unregister(WUBU_BACKEND_BLAS);
        }
    }

    /* 8. Force + query */
    wubu_kernel_force_backend(WUBU_BACKEND_SCALAR);
    {
        const char *a = wubu_kernel_active_backend(WUBU_KERN_GEMM);
        printf("[8] Forced = %s %s\n", a, strcmp(a, "cpu-scalar") == 0 ? "PASS" : "FAIL");
        if (strcmp(a, "cpu-scalar") != 0) errors++;
    }

    /* 9. Shutdown safe */
    wubu_kernel_shutdown();
    printf("[9] shutdown OK\n");

    /* 10. CPU feature detection */
    {
        wubu_cpu_features_t cpu;
        wubu_cpu_detect(&cpu);
        printf("[10] CPU: AVX2=%d FMA=%d AVX512=%d cores=%d L1d=%dKB L2=%dKB L3=%dKB mem_bw=%.0fGB/s\n",
               cpu.has_avx2, cpu.has_fma, cpu.has_avx512, cpu.n_cores,
               cpu.l1d_kb, cpu.l2_kb, cpu.l3_kb, cpu.mem_bw_gbs);
        wubu_backend_id_t b = wubu_kernel_auto_select(WUBU_KERN_GEMM);
        printf("     auto-select GEMM backend: %s\n", wubu_backend_name(b));
        if (cpu.has_avx2 && cpu.has_fma && b != WUBU_BACKEND_CPU_SIMD) {
            fprintf(stderr, "  FAIL: should auto-select CPU_SIMD\n"); errors++;
        }
    }

    printf("=== errors: %d ===\n", errors);
    return errors;
}
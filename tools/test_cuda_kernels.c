// ================================================================
// Test: SSM scalar parallel scan + MoE dispatch CUDA kernels
//
// Tests correctness by comparing GPU output against CPU reference.
// Compiled as C, links with CUDA runtime and cuBLAS.
// ================================================================

#include "cuda_kernels.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ================================================================
// CPU reference: SSM scalar recurrence
// h[t] = A[t] * h[t-1] + B[t] * v[t]
// ================================================================
static void cpu_ssm_scalar_scan(const float *A, const float *B, const float *v,
                                 float *h_init, float *h_final, float *out,
                                 int B_, int T, int d) {
    for (int batch = 0; batch < B_; batch++) {
        float h[128];
        for (int i = 0; i < d; i++) h[i] = h_init[batch * d + i];

        for (int t = 0; t < T; t++) {
            float a = A[batch * T + t];
            float b_val = B[batch * T + t];
            for (int i = 0; i < d; i++) {
                h[i] = a * h[i] + b_val * v[(batch * T + t) * d + i];
                out[(batch * T + t) * d + i] = h[i];
            }
        }
        for (int i = 0; i < d; i++) h_final[batch * d + i] = h[i];
    }
}

// ================================================================
// CPU reference: MoE forward (matching GPU dispatch semantics)
// Processes token-by-token through routed experts + shared expert.
// ================================================================
static void cpu_moe_forward(const float *x, int B, int T,
                             const int *assignments, const float *weights,
                             const moe_weights_t *w,
                             float *output) {
    int N = B * T;
    float *temp = (float *)malloc(D_FF * 3 * sizeof(float));
    float *shared_gate = (float *)malloc(SHARED_D_FF * sizeof(float));
    float *shared_up = (float *)malloc(SHARED_D_FF * sizeof(float));
    float *shared_act = (float *)malloc(SHARED_D_FF * sizeof(float));

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *out_s = output + s * D_MODEL;

        // Shared expert
        for (int j = 0; j < SHARED_D_FF; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++)
                sum += x_s[k] * w->ffn_gate_shexp[k * SHARED_D_FF + j];
            shared_gate[j] = sum;
        }
        for (int j = 0; j < SHARED_D_FF; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++)
                sum += x_s[k] * w->ffn_up_shexp[k * SHARED_D_FF + j];
            shared_up[j] = sum;
        }
        for (int j = 0; j < SHARED_D_FF; j++) {
            float g = shared_gate[j];
            float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
            shared_act[j] = silu_g * shared_up[j];
        }
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int k = 0; k < SHARED_D_FF; k++)
                sum += shared_act[k] * w->ffn_down_shexp[k * D_MODEL + j];
            out_s[j] = sum;
        }

        // Routed experts
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int e = assignments[s * N_ACTIVE_EXPTS + k];
            float wgt = weights[s * N_ACTIVE_EXPTS + k];
            if (e < 0 || wgt < 1e-30f) continue;

            const float *gate_w = w->ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
            const float *up_w   = w->ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
            const float *down_w = w->ffn_down_exps  + (int64_t)e * D_FF * D_MODEL;

            // gate = x @ gate_weight
            for (int j = 0; j < D_FF; j++) {
                float sum = 0.0f;
                for (int i = 0; i < D_MODEL; i++)
                    sum += x_s[i] * gate_w[i * D_FF + j];
                temp[j] = sum;
            }
            // up = x @ up_weight
            for (int j = 0; j < D_FF; j++) {
                float sum = 0.0f;
                for (int i = 0; i < D_MODEL; i++)
                    sum += x_s[i] * up_w[i * D_FF + j];
                temp[D_FF + j] = sum;
            }
            // act = SiLU(gate) * up
            for (int j = 0; j < D_FF; j++) {
                float g = temp[j];
                float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
                temp[2 * D_FF + j] = silu_g * temp[D_FF + j];
            }
            // out = act @ down_weight
            for (int j = 0; j < D_MODEL; j++) {
                float sum = 0.0f;
                for (int i = 0; i < D_FF; i++)
                    sum += temp[2 * D_FF + i] * down_w[i * D_MODEL + j];
                out_s[j] += wgt * sum;
            }
        }
    }

    free(temp);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

// ================================================================
// Test helpers
// ================================================================
static float max_diff_val(const float *a, const float *b, int n) {
    float max_d = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = (float)fabs((double)(a[i] - b[i]));
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static int compare(const float *a, const float *b, int n, float eps, const char *label) {
    float md = max_diff_val(a, b, n);
    if (md > eps) {
        // Find worst index
        int worst = 0;
        float worst_d = 0.0f;
        for (int i = 0; i < n; i++) {
            float d = (float)fabs((double)(a[i] - b[i]));
            if (d > worst_d) { worst_d = d; worst = i; }
        }
        printf("  FAIL [%s]: max_diff=%.6e at idx=%d (a=%.6f b=%.6f)\n",
               label, md, worst, a[worst], b[worst]);
        return 0;
    }
    printf("  PASS [%s]: max_diff=%.6e\n", label, md);
    return 1;
}

// ================================================================
// Test 1: SSM scalar parallel scan
// ================================================================
static int test_ssm_scalar_scan(void) {
    printf("\n=== TEST 1: SSM Scalar Parallel Scan ===\n");

    const int B_ = 2;
    const int T = 8;
    const int d = 16;

    int n_vals = B_ * d;

    float *h_A = (float *)malloc(B_ * T * sizeof(float));
    float *h_B = (float *)malloc(B_ * T * sizeof(float));
    float *h_v = (float *)malloc(B_ * T * d * sizeof(float));
    float *h_h_init = (float *)malloc(B_ * d * sizeof(float));
    float *h_h_cpu = (float *)malloc(B_ * d * sizeof(float));
    float *h_h_gpu = (float *)malloc(B_ * d * sizeof(float));
    float *h_out_cpu = (float *)malloc(B_ * T * d * sizeof(float));
    float *h_out_gpu = (float *)malloc(B_ * T * d * sizeof(float));

    srand(42);
    for (int i = 0; i < B_ * T; i++) {
        h_A[i] = ((float)rand() / RAND_MAX) * 0.9f + 0.05f;
        h_B[i] = ((float)rand() / RAND_MAX) * 0.5f;
    }
    for (int i = 0; i < B_ * T * d; i++)
        h_v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < B_ * d; i++)
        h_h_init[i] = ((float)rand() / RAND_MAX) * 0.1f;

    cpu_ssm_scalar_scan(h_A, h_B, h_v, h_h_init, h_h_cpu, h_out_cpu, B_, T, d);

    float *d_A, *d_B, *d_v, *d_h, *d_out;
    cudaMalloc(&d_A, B_ * T * sizeof(float));
    cudaMalloc(&d_B, B_ * T * sizeof(float));
    cudaMalloc(&d_v, B_ * T * d * sizeof(float));
    cudaMalloc(&d_h, B_ * d * sizeof(float));
    cudaMalloc(&d_out, B_ * T * d * sizeof(float));

    cudaMemcpy(d_A, h_A, B_ * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_ * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, B_ * T * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h_init, B_ * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = NULL; // default stream
    wubu_cuda_ssm_scalar_scan(B_, T, d, d_A, d_B, d_v, d_h, d_out, stream);
    cudaDeviceSynchronize();

    cudaMemcpy(h_h_gpu, d_h, B_ * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_gpu, d_out, B_ * T * d * sizeof(float), cudaMemcpyDeviceToHost);

    int pass = 1;
    pass &= compare(h_h_cpu, h_h_gpu, B_ * d, 1e-5f, "h_final");
    pass &= compare(h_out_cpu, h_out_gpu, B_ * T * d, 1e-5f, "delta_out");

    free(h_A); free(h_B); free(h_v); free(h_h_init);
    free(h_h_cpu); free(h_h_gpu); free(h_out_cpu); free(h_out_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_v); cudaFree(d_h); cudaFree(d_out);

    return pass;
}

// ================================================================
// Test 2: MoE dispatch
// ================================================================
static int test_moe_dispatch(cublasHandle_t handle) {
    printf("\n=== TEST 2: MoE Dispatch ===\n");

    const int B = 1;
    const int T = 4;
    const int N = B * T;

    moe_weights_t w;
    memset(&w, 0, sizeof(w));
    w.loaded = true;

    w.ffn_gate_exps = (float *)malloc((size_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float));
    w.ffn_up_exps   = (float *)malloc((size_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float));
    w.ffn_down_exps = (float *)malloc((size_t)D_FF * D_MODEL * N_EXPERTS * sizeof(float));
    w.ffn_gate_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    w.ffn_up_shexp   = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    w.ffn_down_shexp = (float *)malloc(SHARED_D_FF * D_MODEL * sizeof(float));

    srand(123);
    for (size_t i = 0; i < (size_t)D_MODEL * D_FF * N_EXPERTS; i++) {
        w.ffn_gate_exps[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        w.ffn_up_exps[i]   = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (size_t i = 0; i < (size_t)D_FF * D_MODEL * N_EXPERTS; i++) {
        w.ffn_down_exps[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < D_MODEL * SHARED_D_FF; i++) {
        w.ffn_gate_shexp[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        w.ffn_up_shexp[i]   = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < SHARED_D_FF * D_MODEL; i++) {
        w.ffn_down_shexp[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }

    float *h_x = (float *)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++)
        h_x[i] = ((float)rand() / RAND_MAX) * 0.5f - 0.25f;

    int *h_assign = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    float *h_wgt = (float *)malloc(N * N_ACTIVE_EXPTS * sizeof(float));
    for (int s = 0; s < N; s++) {
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            h_assign[s * N_ACTIVE_EXPTS + k] = (s * N_ACTIVE_EXPTS + k) % N_EXPERTS;
            h_wgt[s * N_ACTIVE_EXPTS + k] = 1.0f / N_ACTIVE_EXPTS;
        }
    }

    float *h_out_cpu = (float *)calloc(N * D_MODEL, sizeof(float));
    cpu_moe_forward(h_x, B, T, h_assign, h_wgt, &w, h_out_cpu);

    float *d_x, *d_out_gpu, *d_scratch;
    int *d_assign;
    float *d_wgt;
    float *d_gate_exps, *d_up_exps, *d_down_exps;
    float *d_gshexp, *d_ushexp, *d_dshexp;
    float *h_out_gpu = (float *)malloc(N * D_MODEL * sizeof(float));

    cudaMalloc(&d_x, N * D_MODEL * sizeof(float));
    cudaMalloc(&d_out_gpu, N * D_MODEL * sizeof(float));
    cudaMalloc(&d_assign, N * N_ACTIVE_EXPTS * sizeof(int));
    cudaMalloc(&d_wgt, N * N_ACTIVE_EXPTS * sizeof(float));
    cudaMalloc(&d_gate_exps, (size_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float));
    cudaMalloc(&d_up_exps,   (size_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float));
    cudaMalloc(&d_down_exps, (size_t)D_FF * D_MODEL * N_EXPERTS * sizeof(float));
    cudaMalloc(&d_gshexp, D_MODEL * SHARED_D_FF * sizeof(float));
    cudaMalloc(&d_ushexp, D_MODEL * SHARED_D_FF * sizeof(float));
    cudaMalloc(&d_dshexp, SHARED_D_FF * D_MODEL * sizeof(float));

    size_t scratch_size = wubu_cuda_moe_dispatch_query_scratch(B, T);
    cudaMalloc(&d_scratch, scratch_size);

    cudaMemcpy(d_x, h_x, N * D_MODEL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assign, h_assign, N * N_ACTIVE_EXPTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wgt, h_wgt, N * N_ACTIVE_EXPTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate_exps, w.ffn_gate_exps, (size_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_exps, w.ffn_up_exps, (size_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_exps, w.ffn_down_exps, (size_t)D_FF * D_MODEL * N_EXPERTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gshexp, w.ffn_gate_shexp, D_MODEL * SHARED_D_FF * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ushexp, w.ffn_up_shexp, D_MODEL * SHARED_D_FF * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dshexp, w.ffn_down_shexp, SHARED_D_FF * D_MODEL * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = NULL;
    wubu_cuda_moe_dispatch(handle, stream, B, T,
        d_x, d_assign, d_wgt,
        d_gate_exps, d_up_exps, d_down_exps,
        d_gshexp, d_ushexp, d_dshexp,
        d_out_gpu, d_scratch);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu, d_out_gpu, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);

    int pass = compare(h_out_cpu, h_out_gpu, N * D_MODEL, 1e-4f, "MoE output");

    // Diagnostic: detailed per-token compare for first 2 tokens
    for (int s = 0; s < 2; s++) {
        float max_d = 0.0f; int max_j = 0;
        for (int j = 0; j < D_MODEL; j++) {
            float d = (float)fabs((double)(h_out_cpu[s*D_MODEL+j] - h_out_gpu[s*D_MODEL+j]));
            if (d > max_d) { max_d = d; max_j = j; }
        }
        printf("  Token %d: max_diff=%.6e at j=%d (cpu=%.6f gpu=%.6f)\n",
               s, max_d, max_j, h_out_cpu[s*D_MODEL+max_j], h_out_gpu[s*D_MODEL+max_j]);
    }

    // Report expert distribution
    printf("  Expert distribution: ");
    int total_assign = 0;
    for (int e = 0; e < N_EXPERTS; e++) {
        int cnt = 0;
        for (int s = 0; s < N; s++)
            for (int k = 0; k < N_ACTIVE_EXPTS; k++)
                if (h_assign[s * N_ACTIVE_EXPTS + k] == e) cnt++;
        if (cnt > 0) {
            printf("e%d:%d ", e, cnt);
            total_assign += cnt;
        }
    }
    printf("(total=%d)\n", total_assign);

    free(h_x); free(h_assign); free(h_wgt); free(h_out_cpu); free(h_out_gpu);
    wubu_moe_free_layer(&w);
    cudaFree(d_x); cudaFree(d_out_gpu); cudaFree(d_scratch);
    cudaFree(d_assign); cudaFree(d_wgt);
    cudaFree(d_gate_exps); cudaFree(d_up_exps); cudaFree(d_down_exps);
    cudaFree(d_gshexp); cudaFree(d_ushexp); cudaFree(d_dshexp);

    return pass;
}

// ================================================================
// Main
// ================================================================
int main() {
    cublasHandle_t handle;
    cudaStream_t stream;

    printf("Initializing CUDA...\n");
    if (!wubu_cuda_init(&handle, &stream)) {
        fprintf(stderr, "CUDA init failed\n");
        return 1;
    }
    printf("CUDA initialized OK\n");

    int passed = 0, total = 2;

    passed += test_ssm_scalar_scan();
    passed += test_moe_dispatch(handle);

    printf("\n=== Results: %d/%d tests passed ===\n", passed, total);

    wubu_cuda_destroy(handle, stream);
    return (passed == total) ? 0 : 1;
}

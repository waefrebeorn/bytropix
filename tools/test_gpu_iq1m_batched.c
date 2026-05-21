// test_gpu_iq1m_batched.c — Test batched IQ1_M GPU quant matmul.
#include "gguf_reader.h"
#include "gpu_quant_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256
#define MIN(a,b) (((a)<(b))?(a):(b))

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf";
    const char *tensor_name = argc > 2 ? argv[2] : "blk.0.ffn_gate_shexp.weight";

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    gguf_buffer_data(ctx);

    gguf_tensor_info *t = gguf_find_tensor(ctx, tensor_name);
    if (!t) { fprintf(stderr, "Tensor not found\n"); return 1; }
    int type = t->ggml_type;
    int64_t n_rows = t->dims[0];
    int64_t n_cols = t->dims[1] > 0 ? t->dims[1] : n_rows;
    printf("Tensor: %s type=%d dims=[%lld,%lld]\n", tensor_name, type,
           (long long)n_rows, (long long)n_cols);

    if (type != GGML_TYPE_IQ1_M) {
        fprintf(stderr, "Not IQ1_M type (got %d) — skipping\n", type);
        return 1;
    }

    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    const uint8_t *w_data = blob + t->data_offset;

    int test_rows = MIN(n_rows, 4096);
    int test_cols = MIN(n_cols, 16);
    int C = 3;  // batch size
    int64_t col_stride = gguf_raw_size(type, n_rows);

    // Generate random input: [C, test_rows]
    float *x = (float *)malloc((size_t)C * test_rows * sizeof(float));
    for (int i = 0; i < C * test_rows; i++) x[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
    
    // Verify x matches single-token test
    printf("x[0]=%f x[1]=%f x[2047]=%f\n", x[0], x[1], x[test_rows-1]);

    // CPU: dequantize each column and dot with each token
    float *y_cpu = (float *)calloc((size_t)C * test_cols, sizeof(float));
    float *deq_buf = (float *)malloc((size_t)QK_K * sizeof(float));
    
    // Debug: compare dequant of col 0, block 0 with single-token test
    {
        float d0[QK_K];
        gguf_dequantize(w_data, type, QK_K, d0);
        double sum_x0 = 0;
        for (int i = 0; i < QK_K; i++) sum_x0 += d0[i];
        double dot0 = 0;
        for (int i = 0; i < QK_K; i++) dot0 += (double)x[i] * (double)d0[i];
        printf("DEBUG: col0 block0 deq[0]=%f deq[1]=%f sum=%f dot_first_block=%f\n",
               d0[0], d0[1], (float)sum_x0, (float)dot0);
    }
    for (int c = 0; c < test_cols; c++) {
        const uint8_t *col_data = w_data + (int64_t)c * col_stride;
        for (int tok = 0; tok < C; tok++) {
            const uint8_t *cd = col_data;
            const float *xt = x + (size_t)tok * test_rows;
            int64_t remaining = test_rows;
            double dot = 0.0;
        while (remaining > 0) {
            int blk = (remaining > QK_K) ? QK_K : (int)remaining;
            gguf_dequantize(cd, type, blk, deq_buf);
            for (int i = 0; i < blk; i++)
                dot += (double)xt[i] * (double)deq_buf[i];
            cd += 56;
            xt += QK_K;
            remaining -= QK_K;
            }
            y_cpu[tok * test_cols + c] = (float)dot;
        }
    }

    // GPU setup
    cudaStream_t stream;
    cudaSetDevice(0);
    cudaStreamCreate(&stream);
    const uint64_t *grid = gguf_get_iq1s_grid();
    wubu_cuda_quant_matmul_set_iq1s_grid(grid);

    size_t w_bytes = (size_t)test_cols * col_stride;
    uint8_t *d_w;
    float *d_x, *d_y;
    cudaMalloc((void**)&d_w, w_bytes);
    cudaMalloc((void**)&d_x, (size_t)C * test_rows * sizeof(float));
    cudaMalloc((void**)&d_y, (size_t)C * test_cols * sizeof(float));
    cudaMemcpy(d_w, w_data, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, (size_t)C * test_rows * sizeof(float), cudaMemcpyHostToDevice);

    int ret = wubu_cuda_quant_matmul_batched(d_x, C, d_w, type,
        test_rows, test_cols, d_y, stream);
    if (!ret) { fprintf(stderr, "GPU batched returned 0\n"); return 1; }
    cudaStreamSynchronize(stream);

    float *y_gpu = (float *)malloc((size_t)C * test_cols * sizeof(float));
    cudaMemcpy(y_gpu, d_y, (size_t)C * test_cols * sizeof(float), cudaMemcpyDeviceToHost);

    double max_diff = 0.0;
    for (int tok = 0; tok < C; tok++) {
        double tok_max = 0.0;
        for (int c = 0; c < test_cols; c++) {
            double diff = fabs(y_gpu[tok * test_cols + c] - y_cpu[tok * test_cols + c]);
            if (diff > tok_max) tok_max = diff;
            if (diff > max_diff) max_diff = diff;
        }
        printf("  tok %d: cpu[0]=%8.5f gpu[0]=%8.5f diff=%e max=%e\n",
               tok, y_cpu[tok * test_cols], y_gpu[tok * test_cols],
               fabs(y_gpu[tok * test_cols] - y_cpu[tok * test_cols]), tok_max);
    }
    printf("Overall max diff: %e\n", max_diff);

    free(x); free(y_cpu); free(y_gpu); free(deq_buf);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
    cudaStreamDestroy(stream);
    gguf_close(ctx);

    printf("%s\n", max_diff < 1e-4 ? "PASS" : "FAIL");
    return max_diff < 1e-4 ? 0 : 1;
}

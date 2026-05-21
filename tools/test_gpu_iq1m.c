// test_gpu_iq1m.c — Verify GPU IQ1_M quant matmul against CPU dequant.
// Loads a column of IQ1_M weights, computes GPU and CPU dot products, reports diff.
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
    const char *tensor_name = argc > 2 ? argv[2] : "blk.3.attn_qkv.weight";

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    gguf_buffer_data(ctx);

    gguf_tensor_info *t = gguf_find_tensor(ctx, tensor_name);
    if (!t) { fprintf(stderr, "Tensor '%s' not found\n", tensor_name); return 1; }
    int type = t->ggml_type;
    int64_t n_rows = t->dims[0];
    int64_t n_cols = t->dims[1] > 0 ? t->dims[1] : n_rows;
    printf("Tensor: %s type=%d dims=[%lld,%lld]\n", tensor_name, type,
           (long long)n_rows, (long long)n_cols);

    if (type != GGML_TYPE_IQ1_M) {
        fprintf(stderr, "Not IQ1_M type (got %d) — test requires IQ1_M weights\n", type);
        // Try anyway
    }

    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    const uint8_t *w_data = blob + t->data_offset;

    // Pick a small number of rows and columns for test
    int test_rows = MIN(n_rows, 4096);   // 16 QK_K blocks
    int test_cols = MIN(n_cols, 4);
    int64_t col_stride = gguf_raw_size(type, n_rows);
    printf("Test: %d rows x %d cols (col_stride=%lld)\n",
           test_rows, test_cols, (long long)col_stride);

    // Generate random input
    float *x = (float *)malloc((size_t)test_rows * sizeof(float));
    for (int i = 0; i < test_rows; i++) x[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
    
    printf("x[0]=%f x[1]=%f x[2047]=%f\n", x[0], x[1], x[test_rows-1]);

    // CPU: dequantize each column and dot with x
    float *y_cpu = (float *)calloc(test_cols, sizeof(float));
    float *deq_buf = (float *)malloc((size_t)QK_K * sizeof(float));
    
    // Debug: compare dequant of col 0, block 0 with batched test
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
        int64_t remaining = test_rows;
        int64_t offset = 0;
        while (remaining > 0) {
            int blk = (remaining > QK_K) ? QK_K : (int)remaining;
            gguf_dequantize(col_data, type, blk, deq_buf);
            for (int i = 0; i < blk; i++)
                y_cpu[c] += x[offset + i] * deq_buf[i];
            col_data += 56; // IQ1_M block size: qs[32]+qh[16]+scales[8]
            offset += QK_K;
            remaining -= QK_K;
        }
    }

    // GPU: upload grid, weights, input, run matmul
    cudaError_t ce;
    cudaStream_t stream;
    cudaSetDevice(0);
    cudaStreamCreate(&stream);

    // Upload grid table
    const uint64_t *grid = gguf_get_iq1s_grid();
    wubu_cuda_quant_matmul_set_iq1s_grid(grid);

    // Upload weights (all columns)
    size_t w_bytes = (size_t)test_cols * col_stride;
    uint8_t *d_w;
    float *d_x, *d_y;
    cudaMalloc((void**)&d_w, w_bytes);
    cudaMalloc((void**)&d_x, (size_t)test_rows * sizeof(float));
    cudaMalloc((void**)&d_y, (size_t)test_cols * sizeof(float));
    cudaMemcpy(d_w, w_data, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, (size_t)test_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU matmul
    int ret = wubu_cuda_quant_matmul(d_x, d_w, type,
        test_rows, test_cols, d_y, NULL, 0, stream);
    if (!ret) {
        fprintf(stderr, "GPU quant_matmul returned 0 (unsupported type %d)\n", type);
        return 1;
    }
    cudaStreamSynchronize(stream);

    float *y_gpu = (float *)malloc((size_t)test_cols * sizeof(float));
    cudaMemcpy(y_gpu, d_y, (size_t)test_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    double max_diff = 0.0;
    for (int c = 0; c < test_cols; c++) {
        double diff = fabs((double)y_gpu[c] - (double)y_cpu[c]);
        if (diff > max_diff) max_diff = diff;
        printf("  col %d: cpu=%10.6f  gpu=%10.6f  diff=%e\n", c, y_cpu[c], y_gpu[c], diff);
    }
    printf("Max diff: %e\n", max_diff);

    // Cleanup
    free(x); free(y_cpu); free(y_gpu); free(deq_buf);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
    cudaStreamDestroy(stream);
    gguf_close(ctx);

    printf("%s\n", max_diff < 1e-4 ? "PASS" : "FAIL");
    return max_diff < 1e-4 ? 0 : 1;
}

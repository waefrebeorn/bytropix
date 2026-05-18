// tools/test_quantized_matmul.c
// Test: compare quantized_matmul output vs F32 SGEMM for a single layer's Q5_K weight.
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    // Open model to get quantized weight data
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("FAIL: gguf_open\n"); return 1; }
    
    // Buffer the data blob so we have access to quantized weights
    gguf_buffer_data(ctx);
    
    // Find a Q5_K weight tensor (blk.0.attn_qkv.weight)
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    if (!t) { printf("FAIL: tensor not found\n"); gguf_close(ctx); return 1; }
    
    printf("Tensor: %s, type=%d, dims=[%ld,%ld], data_offset=%lu\n",
           t->name, t->ggml_type, (long)t->dims[0], (long)t->dims[1],
           (unsigned long)t->data_offset);
    
    int64_t n_rows = t->dims[0];  // input dim (D_MODEL = 2048)
    int64_t n_cols = t->dims[1];  // output dim (qkv_dim = 8192)
    int64_t n_elems = n_rows * n_cols;
    
    // Get quantized data pointer
    const uint8_t *qdata = (const uint8_t *)ctx->data_blob + t->data_offset;
    int weight_type = t->ggml_type;  // should be Q5_K
    
    // Allocate F32 reference
    float *W_f32 = (float *)malloc(n_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, W_f32, n_elems);
    
    // Create a random input vector
    float *x = (float *)malloc(n_rows * sizeof(float));
    srand(42);
    for (int i = 0; i < n_rows; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    // Compute reference: F32 SGEMM y_ref[j] = sum_i x[i] * W[i][j]
    float *y_ref = (float *)calloc(n_cols, sizeof(float));
    for (int j = 0; j < n_cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < n_rows; i++) {
            sum += (double)x[i] * (double)W_f32[j * n_rows + i];  // column-major
        }
        y_ref[j] = (float)sum;
    }
    printf("F32 ref: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
           (double)y_ref[0], (double)y_ref[1], (double)y_ref[2], (double)y_ref[3], (double)y_ref[4]);
    
    // Compute quantized matmul
    float *y_q = (float *)calloc(n_cols, sizeof(float));
    quantized_matmul(x, qdata, weight_type, n_rows, n_cols, 0, y_q);
    printf("Qmatmul: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
           (double)y_q[0], (double)y_q[1], (double)y_q[2], (double)y_q[3], (double)y_q[4]);
    
    // Compute cosine similarity
    double dot = 0, n1 = 0, n2 = 0, max_err = 0;
    for (int j = 0; j < n_cols; j++) {
        dot += (double)y_ref[j] * (double)y_q[j];
        n1  += (double)y_ref[j] * (double)y_ref[j];
        n2  += (double)y_q[j]    * (double)y_q[j];
        double err = fabs((double)y_ref[j] - (double)y_q[j]);
        if (err > max_err) max_err = err;
    }
    double cos_sim = dot / (sqrt(n1) * sqrt(n2));
    printf("cos-sim = %.10f\n", cos_sim);
    printf("max_err = %.10f\n", max_err);
    
    int pass = (cos_sim > 0.9999f);
    printf("%s\n", pass ? "PASS" : "FAIL");
    
    free(W_f32); free(x); free(y_ref); free(y_q);
    gguf_close(ctx);
    return pass ? 0 : 1;
}

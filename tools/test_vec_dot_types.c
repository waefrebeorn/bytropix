// Quick test: verify Q4_K and Q6_K vec_dot against F32 SGEMM
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void test_one(gguf_ctx *ctx, const char *tensor_name) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, tensor_name);
    if (!t) { printf("  %s: NOT FOUND\n", tensor_name); return; }
    
    int64_t n_rows = t->dims[0];
    int64_t n_cols = t->dims[1];
    int64_t n_elems = n_rows * n_cols;
    const uint8_t *qdata = (const uint8_t *)ctx->data_blob + t->data_offset;
    
    float *W = (float *)malloc(n_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, W, n_elems);
    
    float *x = (float *)malloc(n_rows * sizeof(float));
    srand(12345);
    for (int i = 0; i < n_rows; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    float *ref = (float *)calloc(n_cols, sizeof(float));
    #pragma omp parallel for
    for (int j = 0; j < n_cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < n_rows; i++) sum += (double)x[i] * (double)W[j * n_rows + i];
        ref[j] = (float)sum;
    }
    
    float *qy = (float *)calloc(n_cols, sizeof(float));
    quantized_matmul(x, qdata, t->ggml_type, n_rows, n_cols, 0, qy);
    
    double dot = 0, n1 = 0, n2 = 0, max_err = 0;
    for (int j = 0; j < n_cols; j++) {
        dot += (double)ref[j] * (double)qy[j];
        n1  += (double)ref[j] * (double)ref[j];
        n2  += (double)qy[j]  * (double)qy[j];
        double err = fabs((double)ref[j] - (double)qy[j]);
        if (err > max_err) max_err = err;
    }
    double cos_sim = dot / (sqrt(n1) * sqrt(n2));
    
    printf("  %s (type=%d, [%ldx%ld]): cos-sim=%.10f max_err=%.6f\n",
           tensor_name, t->ggml_type, (long)n_rows, (long)n_cols, cos_sim, max_err);
    
    free(W); free(x); free(ref); free(qy);
}

int main(void) {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    // Test Q5_K, Q6_K, Q4_K
    test_one(ctx, "blk.0.attn_qkv.weight");     // Q5_K [2048,8192]
    test_one(ctx, "blk.0.ssm_out.weight");        // Q6_K [4096,2048]
    test_one(ctx, "blk.0.attn_q.weight");         // Q5_K [2048,8192]
    test_one(ctx, "output.weight");                // Q4_K [50272,2048]
    
    gguf_close(ctx);
    printf("Done.\n");
    return 0;
}

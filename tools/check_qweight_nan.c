#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.3.attn_q.weight");
    if (!t) { printf("NOT FOUND\n"); return 1; }
    
    int q_dim = 4096; // GQA_Q_HEADS * GQA_HEAD_DIM = 16 * 256
    // Weight is [D_MODEL=2048, q_dim*2=8192]
    float *weight = (float*)malloc(2048 * 8192 * sizeof(float));
    int nread = gguf_read_tensor_f32(ctx, t, weight, 2048 * 8192);
    printf("Read %d / %ld elements\n", nread, 2048L * 8192L);
    
    // Check weight[j=4096] across all input dims
    printf("Checking weight[i * 8192 + 4096] for i=0..2047:\n");
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < 2048; i++) {
        float v = weight[i * 8192 + 4096];
        if (isnan(v)) { nan_count++; if (nan_count <= 5) printf("  NaN at i=%d\n", i); }
        if (isinf(v)) { inf_count++; if (inf_count <= 5) printf("  Inf at i=%d\n", i); }
    }
    printf("  NaN: %d / 2048, Inf: %d / 2048\n", nan_count, inf_count);
    
    // Check a few sample values
    printf("  weight[  0*8192+4096] = %e\n", (double)weight[4096]);
    printf("  weight[100*8192+4096] = %e\n", (double)weight[100*8192+4096]);
    printf("  weight[500*8192+4096] = %e\n", (double)weight[500*8192+4096]);
    printf("  weight[1000*8192+4096] = %e\n", (double)weight[1000*8192+4096]);
    printf("  weight[1500*8192+4096] = %e\n", (double)weight[1500*8192+4096]);
    printf("  weight[2000*8192+4096] = %e\n", (double)weight[2000*8192+4096]);
    
    // Also check token s=2 in x @ W:
    // x = [-2.92, -0.34, -0.27, -1.49, ...]
    // The sum x @ W[:, 4096] should be finite
    double total = 0;
    for (int i = 0; i < 2048; i++) {
        total += weight[i * 8192 + 4096] * weight[i * 8192 + 4096]; // just check weight norm
    }
    printf("  Weight column 4096 norm: %.2f\n", sqrt(total));
    
    free(weight);
    gguf_close(ctx);
    return 0;
}

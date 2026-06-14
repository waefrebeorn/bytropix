#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    if (!t) { fprintf(stderr, "not found\n"); return 1; }
    printf("attn_q: type=%d dims=%lld %lld\n", t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);

    const uint8_t *weight_q = blob + t->data_offset;
    int K = t->dims[0];
    int N = t->dims[1];
    int blocks_per_col = (K + 255) / 256;
    int total_blocks = blocks_per_col * N;
    size_t q4k_size = (size_t)total_blocks * 144;
    size_t f32_size = (size_t)K * N * sizeof(float);

    // Test 1: Original host data
    printf("\n=== Test 1: Original host data (first 2 blocks = 512 elems) ===\n");
    float *host_out = (float*)malloc(512 * sizeof(float));
    gguf_dequantize(weight_q, t->ggml_type, 512, host_out);
    
    printf("First 16 elements:\n");
    for (int i = 0; i < 16; i++) {
        printf("  [%d] = %.6f\n", i, host_out[i]);
    }

    // Test 2: Simulate GPU download (malloc + memcpy)
    printf("\n=== Test 2: Simulated GPU download (first 2 blocks) ===\n");
    uint8_t *sim_h_q4k = (uint8_t*)malloc(q4k_size);
    memcpy(sim_h_q4k, weight_q, q4k_size);  // Simulate GPU download
    
    // Dequantize using gguf_dequantize on the simulated data
    printf("First 2 blocks of simulated GPU data:\n");
    float *sim_out = (float*)malloc(512 * sizeof(float));
    gguf_dequantize(sim_h_q4k, t->ggml_type, 512, sim_out);
    
    printf("First 16 elements:\n");
    for (int i = 0; i < 16; i++) {
        printf("  [%d] = %.6f\n", i, sim_out[i]);
    }

    // Compare
    printf("\n=== Comparison ===\n");
    double max_err = 0;
    for (int i = 0; i < 512; i++) {
        double err = fabs(host_out[i] - sim_out[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error between host and sim: %.10f\n", max_err);

    // Test 3: Full tensor dequantize using gguf_dequantize
    printf("\n=== Test 3: Full tensor dequantize ===\n");
    float *full_host = (float*)malloc(f32_size);
    float *full_sim = (float*)malloc(f32_size);
    
    gguf_dequantize(weight_q, t->ggml_type, (int64_t)K * N, full_host);
    gguf_dequantize(sim_h_q4k, t->ggml_type, (int64_t)K * N, full_sim);
    
    max_err = 0;
    int max_idx = 0;
    for (int64_t i = 0; i < (int64_t)K * N; i++) {
        double err = fabs(full_host[i] - full_sim[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("Max error on full tensor: %.10f at index %d\n", max_err, max_idx);

    // Check specific indices
    printf("Ref first 16: ");
    for (int i = 0; i < 16; i++) printf("%.6f ", full_host[i]);
    printf("\nSim first 16: ");
    for (int i = 0; i < 16; i++) printf("%.6f ", full_sim[i]);
    printf("\n");

    free(host_out);
    free(sim_out);
    free(full_host);
    free(full_sim);
    free(sim_h_q4k);
    gguf_close(ctx);
    return 0;
}

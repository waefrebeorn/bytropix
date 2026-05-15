#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // Open GGUF, buffer, find output.weight once
    gguf_ctx *ctx = gguf_open(path);
    gguf_buffer_data(ctx);
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    printf("output.weight at data_offset=%ld type=%d dims=[%ld,%ld]\n", 
           (long)t->data_offset, t->ggml_type, t->dims[0], t->dims[1]);
    
    // Read first 20 elements TWICE from the SAME context
    float *r1 = (float *)malloc(20 * sizeof(float));
    float *r2 = (float *)malloc(20 * sizeof(float));
    
    gguf_read_tensor_f32(ctx, t, r1, 20);
    printf("Read 1 (first 20):\n");
    for (int i = 0; i < 20; i++) printf("  [%d]=%.6e\n", i, r1[i]);
    
    gguf_read_tensor_f32(ctx, t, r2, 20);
    printf("Read 2 (first 20):\n");
    for (int i = 0; i < 20; i++) printf("  [%d]=%.6e\n", i, r2[i]);
    
    // Check if they match
    int diff = 0;
    for (int i = 0; i < 20; i++)
        if (fabsf(r1[i] - r2[i]) > 1e-6f) { diff++; }
    printf("Different elements: %d/20\n", diff);
    
    // Now read a LARGE block (10K elements) then read first 20 again
    int n_big = 10000;
    float *big = (float *)malloc(n_big * sizeof(float));
    gguf_read_tensor_f32(ctx, t, big, n_big);
    
    float *r3 = (float *)malloc(20 * sizeof(float));
    gguf_read_tensor_f32(ctx, t, r3, 20);
    printf("Read 3 (after reading 10K elements, first 20):\n");
    for (int i = 0; i < 20; i++) printf("  [%d]=%.6e\n", i, r3[i]);
    
    diff = 0;
    for (int i = 0; i < 20; i++)
        if (fabsf(r1[i] - r3[i]) > 1e-6f) { diff++; }
    printf("Different elements after large read: %d/20\n", diff);
    
    free(r1); free(r2); free(r3); free(big);
    gguf_close(ctx);
    printf("=== PASS ===\n");
    return 0;
}

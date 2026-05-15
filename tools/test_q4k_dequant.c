// Test Q4_K dequant: compare full model load vs fresh small read
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // Method 1: model_load
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) return 1;
    printf("Model output_weight[0..9]:\n");
    for (int i = 0; i < 10; i++) 
        printf("  [%d] = %.6e\n", i, model.output_weight[i]);
    
    printf("\nModel output_weight[2048..2057] (vocab[1] dim[0..9]):\n");
    for (int i = 0; i < 10; i++)
        printf("  [%d] = %.6e\n", 2048+i, model.output_weight[2048+i]);
    
    // Method 2: fresh read from raw data
    gguf_ctx *ctx2 = gguf_open(path);
    gguf_buffer_data(ctx2);
    gguf_tensor_info *t2 = gguf_find_tensor(ctx2, "output.weight");
    
    int64_t n = 4096; // 2 vocab entries
    float *w2 = (float *)malloc(n * sizeof(float));
    gguf_read_tensor_f32(ctx2, t2, w2, n);
    
    printf("\nFresh read w2[0..9]:\n");
    for (int i = 0; i < 10; i++)
        printf("  [%d] = %.6e\n", i, w2[i]);
    
    printf("\nFresh read w2[2048..2057] (vocab[1] dim[0..9]):\n");
    for (int i = 0; i < 10; i++)
        printf("  [%d] = %.6e\n", 2048+i, w2[2048+i]);
    
    printf("\nComparison model[0..9] vs fresh[0..9]:\n");
    for (int i = 0; i < 10; i++) {
        float diff = fabsf(model.output_weight[i] - w2[i]);
        printf("  [%d] model=%.6e fresh=%.6e diff=%.6e%s\n", 
               i, model.output_weight[i], w2[i], diff,
               diff > 1e-6 ? " *** DIFF ***" : "");
    }
    
    free(w2);
    gguf_close(ctx2);
    wubu_model_free(&model);
    printf("\n=== PASS ===\n");
    return 0;
}

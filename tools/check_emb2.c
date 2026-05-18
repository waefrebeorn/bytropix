// Directly compare GGUF token embedding vs saved file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const int D = 2048;
    const int BOS_ID = 248044;
    
    // Extract token 0 embedding directly from GGUF using raw tensor read
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { printf("ERROR: token_embd.weight not found\n"); return 1; }
    
    // Read just token 0 (first D floats)
    float *tok0 = (float *)malloc(D * sizeof(float));
    int n = gguf_read_tensor_f32(ctx, t, tok0, D);
    printf("Read token 0: %d floats\n", n);
    printf("Token 0 first 10: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", tok0[i]);
    printf("\n");
    
    // Now compare with the saved file
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (f) {
        float *file_tok0 = (float *)malloc(D * sizeof(float));
        fread(file_tok0, sizeof(float), D, f);
        fclose(f);
        
        float max_diff = 0;
        for (int i = 0; i < D; i++) {
            float diff = fabsf(tok0[i] - file_tok0[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max diff token 0 vs file: %.6f\n", max_diff);
        printf("File token 0 first 10: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", file_tok0[i]);
        printf("\n");
        free(file_tok0);
    }
    
    gguf_close(ctx);
    free(tok0);
    return 0;
}

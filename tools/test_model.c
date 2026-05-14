#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    printf("Loading model from %s...\n", model_path);
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Run a single-token forward pass
    int B = 1, T = 4;
    int tokens[] = {0, 1, 2, 3};  // dummy token IDs
    float *logits = (float *)malloc(B * T * 248320 * sizeof(float));
    
    printf("\nRunning forward pass (B=%d, T=%d)...\n", B, T);
    wubu_model_forward(&model, tokens, B, T, logits);
    
    printf("Done! Logits[0:8]:");
    for (int i = 0; i < 8; i++) printf(" %+.6f", logits[i]);
    printf("\n");
    
    // Also run Euclidean SSM only (for comparison)
    float *embd = (float *)malloc(B * T * D_MODEL * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (f) {
        for (int i = 0; i < B * T; i++) {
            int id = tokens[i];
            if (id >= 0 && id < model.vocab_size) {
                fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
                fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
            }
        }
        fclose(f);
        printf("Embeddings[0:4]: %.6f %.6f %.6f %.6f\n", embd[0], embd[1], embd[2], embd[3]);
    }
    
    free(logits);
    free(embd);
    wubu_model_free(&model);
    return 0;
}

/**
 * test_nan_logit.c — Find NaN in output projection specifically
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main() {
    wubu_model_t model;
    if (!wubu_model_init(&model, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;

    // Load text embeddings
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { perror("fopen"); return 1; }
    float embd[8*2048];
    fread(embd, sizeof(float), 8*2048, f);
    fclose(f);

    int B = 1, T = 8, N = B*T, D = D_MODEL;
    
    // Do full model forward
    float *logits = (float *)malloc(B * T * model.vocab_size * sizeof(float));
    wubu_model_forward_from_embd(&model, embd, B, T, logits);

    // Find NaN positions
    int first_nan = -1, nan_count = 0;
    for (int i = 0; i < B * T * model.vocab_size; i++) {
        if (isnan(logits[i])) {
            nan_count++;
            if (first_nan < 0) {
                first_nan = i;
                printf("First NaN at global_idx=%d (tok=%d, vocab=%d)\n",
                       i, i / model.vocab_size, i % model.vocab_size);
            }
        }
    }
    printf("Total NaN: %d / %d (%.4f%%)\n", nan_count, B*T*model.vocab_size,
           100.0*nan_count/(B*T*model.vocab_size));

    // Check if NaN comes from specific vocab positions
    int vocab_nan[256] = {0};
    int total_per_bucket[256] = {0};
    for (int t = 0; t < B*T; t++) {
        for (int v = 0; v < model.vocab_size; v++) {
            total_per_bucket[v / 1000]++;
            if (isnan(logits[t * model.vocab_size + v]))
                vocab_nan[v / 1000]++;
        }
    }
    printf("\nNaN distribution by vocab bucket (each 1000 tokens):\n");
    for (int b = 0; b < 248; b++) {
        if (vocab_nan[b] > 0)
            printf("  bucket[%d-%d]: %d/%d NaN\n", b*1000, (b+1)*1000-1, vocab_nan[b], total_per_bucket[b]);
    }

    // Check if same vocab positions are NaN across tokens
    if (B*T > 1) {
        int same_pos_nan = 0;
        int total_nan_in_first = 0;
        for (int v = 0; v < model.vocab_size; v++) {
            if (isnan(logits[v])) total_nan_in_first++;
            int all_nan = 1;
            for (int t = 0; t < B*T; t++)
                if (!isnan(logits[t * model.vocab_size + v])) { all_nan = 0; break; }
            if (all_nan) same_pos_nan++;
        }
        printf("\nVocab positions NaN across ALL tokens: %d\n", same_pos_nan);
        printf("Vocab positions NaN in first token: %d\n", total_nan_in_first);
    }

    wubu_model_free(&model);
    free(logits);
    return 0;
}

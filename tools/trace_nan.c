#include "gguf_reader.h"
#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = "data/train_data.bin";
    const char *embed_path = "data/qwen36_embeddings_c.bin.raw";

    // Load model
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) return 1;
    printf("Model: %d layers\n", model.n_layers);

    // Load corpus
    FILE *f = fopen(corpus_path, "rb");
    fseek(f, 0, SEEK_END);
    long data_size = ftell(f);
    int total_tokens = (int)(data_size / sizeof(int));
    fseek(f, 0, SEEK_SET);
    int *tokens = (int *)malloc(data_size);
    fread(tokens, 1, data_size, f);
    fclose(f);
    printf("Corpus: %d tokens\n", total_tokens);
    printf("First 8 token IDs: ");
    for (int i = 0; i < 8 && i < total_tokens; i++) printf("%d ", tokens[i]);
    printf("\n");

    // Load embeddings from file
    f = fopen(embed_path, "rb");
    fseek(f, 0, SEEK_END);
    long emb_size = ftell(f);
    int emb_tokens = (int)(emb_size / (D_MODEL * sizeof(float)));
    fclose(f);
    printf("Embeddings: %d tokens available (%ld MB)\n", emb_tokens, emb_size/(1024*1024));

    // Load first few embeddings
    float test_emb[D_MODEL];
    for (int ti = 0; ti < 4 && ti < total_tokens; ti++) {
        int id = tokens[ti];
        if (id < 0 || id >= emb_tokens) id = 0;
        f = fopen(embed_path, "rb");
        fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
        fread(test_emb, sizeof(float), D_MODEL, f);
        fclose(f);
        printf("Token %d (id=%d) emb[0:4]: %+.6f %+.6f %+.6f %+.6f  norm=%.4f\n",
               ti, id, test_emb[0], test_emb[1], test_emb[2], test_emb[3],
               sqrtf(test_emb[0]*test_emb[0] + test_emb[1]*test_emb[1]));
    }

    // Run forward layer by layer, check for NaN
    int B = 1, T = 4, N = B * T;
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    float *hidden = (float *)malloc(N * D_MODEL * sizeof(float));

    // Load embeddings for first N tokens
    for (int i = 0; i < N; i++) {
        int id = tokens[i];
        if (id < 0 || id >= emb_tokens) id = 0;
        f = fopen(embed_path, "rb");
        fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
        fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
        fclose(f);
    }

    printf("\n--- Running layer-by-layer ---\n");
    memcpy(hidden, embd, N * D_MODEL * sizeof(float));

    for (int l = 0; l < model.n_layers && l < 42; l++) {
        if (l < model.n_layers) {
            if (model.layers[l].type == LAYER_SSM) {
                wubu_ssm_forward(hidden, B, T, &model.layers[l].weights.ssm,
                                 model.ssm_states[l]);
            } else {
                wubu_gqa_forward(hidden, B, T, &model.layers[l].weights.gqa);
            }
            // Add residual
            for (int i = 0; i < N * D_MODEL; i++) {
                hidden[i] = hidden[i]; // identity for now — shows raw output
            }
        }
        // Check for NaN
        int nan_count = 0;
        for (int i = 0; i < N * D_MODEL && i < 10; i++) {
            if (isnan(hidden[i])) nan_count++;
        }
        printf("  Layer %d (%s): hidden[0:4]=%+.4f %+.4f %+.4f %+.4f  NaN=%s\n",
               l, (l < model.n_layers && model.layers[l].type == LAYER_SSM) ? "SSM" : "GQA",
               hidden[0], hidden[1], hidden[2], hidden[3],
               nan_count > 0 ? "YES" : "no");
        if (nan_count > 0) {
            printf("    *** NaN detected at layer %d! ***\n", l);
            break;
        }

        // After layer 0, check again
        if (l == 0) {
            printf("    Res norm after layer 0: ");
            float sum = 0;
            for (int i = 0; i < N * D_MODEL; i++) sum += hidden[i] * hidden[i];
            printf("%.4f\n", sqrtf(sum / (N * D_MODEL)));
        }
    }

    free(tokens); free(embd); free(hidden);
    wubu_model_free(&model);
    return 0;
}

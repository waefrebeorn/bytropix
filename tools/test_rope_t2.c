// Test GQA RoPE by comparing with reference for T=2
// Uses same token twice (sine and cosine should alternate positions)
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = true;

    int D = D_MODEL;
    int vs = mdl.vocab_size;
    int T = 2;
    int B = 1;

    // Load 2 token embeddings
    float *x = (float *)malloc(B * T * D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { printf("ERROR: can't open emb file\n"); return 1; }
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);          // token 0
        fread(x + D, sizeof(float), D, f);       // token 1 (next token)
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
        memcpy(x + D, mdl.token_embd + 248045LL * D, D * sizeof(float));  // next token
    }

    // Run forward
    float *logits = (float *)malloc(B * T * vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, B, T, logits);

    printf("T=2 forward complete\n");
    printf("Token 0 top-5:\n");
    {
        float *cpy = (float *)malloc(vs * sizeof(float));
        memcpy(cpy, logits, vs * sizeof(float));
        for (int k = 0; k < 5; k++) {
            float best = -1e30f; int best_idx = -1;
            for (int i = 0; i < vs; i++) if (cpy[i] > best) { best = cpy[i]; best_idx = i; }
            cpy[best_idx] = -1e30f;
            printf("  [%d] val=%.6f\n", best_idx, (double)best);
        }
        free(cpy);
    }
    printf("Token 1 top-5:\n");
    {
        float *cpy = (float *)malloc(vs * sizeof(float));
        memcpy(cpy, logits + vs, vs * sizeof(float));
        for (int k = 0; k < 5; k++) {
            float best = -1e30f; int best_idx = -1;
            for (int i = 0; i < vs; i++) if (cpy[i] > best) { best = cpy[i]; best_idx = i; }
            cpy[best_idx] = -1e30f;
            printf("  [%d] val=%.6f\n", best_idx, (double)best);
        }
        free(cpy);
    }

    // Compare T=1 vs reference (token 0 should match)
    float *ref = (float *)malloc(vs * sizeof(float));
    FILE *f = fopen("/tmp/llama_logits_new.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), vs, f);
        fclose(f);
        double dot = 0, n_our = 0, n_ref = 0;
        for (int i = 0; i < vs; i++) {
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("\nToken 0 cos-sim vs T=1 reference: %.10f\n", cos);
        printf("  (should be ~0.997 — T=1 was 0.99696)\n");
    }

    free(ref);
    free(logits);
    free(x);
    wubu_model_free(&mdl);
    return 0;
}

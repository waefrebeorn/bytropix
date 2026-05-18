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
    mdl.enable_moe = false;  // NO MoE
    
    int D = D_MODEL;
    int vs = mdl.vocab_size;
    
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { printf("ERROR: can't open emb file\n"); return 1; }
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    float *logits = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    // Compare vs reference
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
        printf("cos-sim (no MoE): %.10f\n", cos);
    }
    
    free(ref);
    free(logits);
    free(x);
    wubu_model_free(&mdl);
    return 0;
}

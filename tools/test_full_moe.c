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
    
    // Run full forward (will produce per-layer dump)
    float *logits = (float *)malloc(mdl.vocab_size * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    // Compare vs reference logits
    float *ref = (float *)malloc(mdl.vocab_size * sizeof(float));
    FILE *f = fopen("/tmp/llama_logits_new.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), mdl.vocab_size, f);
        fclose(f);
        double dot = 0, n_our = 0, n_ref = 0;
        double max_diff = 0; int max_diff_idx = -1;
        for (int i = 0; i < mdl.vocab_size; i++) {
            double d = (double)logits[i] - (double)ref[i];
            if (fabs(d) > max_diff) { max_diff = fabs(d); max_diff_idx = i; }
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("cos-sim vs reference: %.10f\n", cos);
        printf("max diff: %.10f at idx %d\n", max_diff, max_diff_idx);
    }
    
    free(ref);
    free(logits);
    free(x);
    wubu_model_free(&mdl);
    return 0;
}

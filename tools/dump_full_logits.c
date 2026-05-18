/* Full model + MoE, dump logits for comparison */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    mdl.enable_moe = true;
    
    int vs = mdl.vocab_size;
    int D = D_MODEL;
    
    // Get BOS embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        fseek(emb_f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, emb_f);
        fclose(emb_f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    float *logits = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    // Dump logits
    FILE *f = fopen("/tmp/our_full_logits.bin", "wb");
    fwrite(logits, sizeof(float), vs, f);
    fclose(f);
    
    // Top-5
    int top[5] = {0}; float tv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    for (int i = 0; i < vs; i++) {
        if (logits[i] > tv[4]) {
            tv[4] = logits[i]; top[4] = i;
            for (int k = 3; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float t = tv[k]; tv[k] = tv[k+1]; tv[k+1] = t;
                    int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
                }
            }
        }
    }
    printf("Our full model (40L+MoE) top-5:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    
    // Compare vs reference
    float *ref = (float *)malloc(vs * sizeof(float));
    f = fopen("/tmp/ref_logits.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), vs, f);
        fclose(f);
        double dot = 0, n_our = 0, n_ref = 0;
        int n = vs < 50000 ? vs : 50000;
        for (int i = 0; i < n; i++) {
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos_sim = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("Cos-sim vs ref (first %d logits): %.6f\n", n, cos_sim);
        free(ref);
    }
    
    free(x); free(logits);
    wubu_model_free(&mdl);
    return 0;
}

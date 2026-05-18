/* Full model forward (40 layers, no MoE) with fresh embedding, compare vs ref */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = false;
    
    int D = D_MODEL;
    int vs = mdl.vocab_size;
    
    // Get embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Allocate logits
    float *logits = (float *)malloc(vs * sizeof(float));
    
    // Run full forward (40 layers, no MoE)
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
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
    printf("Full model (40L, no MoE) top-5:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    
    // Save our logits
    FILE *f = fopen("/tmp/our_logits_40l.bin", "wb");
    fwrite(logits, sizeof(float), 50000, f);
    fclose(f);
    
    // Compare vs ref
    float *ref = (float *)malloc(50000 * sizeof(float));
    f = fopen("/tmp/ref_logits.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), 50000, f);
        fclose(f);
        double dot=0, n_our=0, n_ref=0;
        for (int i = 0; i < 50000; i++) {
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos_sim = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("Cos-sim vs ref (first 50000 logits): %.6f\n", cos_sim);
        
        // Also compare top-5 overlap
        printf("\nReference top-5:\n");
        for (int i = 0; i < 50000; i++) {
            if (ref[i] > tv[4]) {
                tv[4] = ref[i]; top[4] = i;
                for (int k = 3; k >= 0; k--) {
                    if (tv[k] < tv[k+1]) {
                        float t = tv[k]; tv[k] = tv[k+1]; tv[k+1] = t;
                        int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
                    }
                }
            }
        }
        // tv and top now have ref's top-5
        for (int k = 0; k < 5; k++)
            printf("  [%d] val=%.4f\n", top[k], tv[k]);
    }
    
    free(x); free(logits); free(ref);
    wubu_model_free(&mdl);
    return 0;
}

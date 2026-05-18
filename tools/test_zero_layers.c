/* Zero layers: just embedding + final norm + output projection */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"

int main() {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    int D = D_MODEL;
    int vs = mdl.vocab_size;
    
    // Get BOS embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Final RMSNorm only (no layers)
    float *final = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, mdl.norm_weight, 1e-6f, final);
    
    // Output projection
    float *logits = (float *)malloc(vs * sizeof(float));
    for (int j = 0; j < vs; j++) {
        double sum = 0.0;
        for (int k = 0; k < D; k++)
            sum += (double)final[k] * (double)mdl.output_weight[j * D + k];
        logits[j] = (float)sum;
    }
    
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
    printf("0-layer (BOS emb + final norm + output) top-5:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    
    // Save logits for comparison
    FILE *f = fopen("/tmp/our_l0_logits.bin", "wb");
    fwrite(logits, sizeof(float), 5000, f);
    fclose(f);
    
    // Compare against ref logits (full 40-layer llama.cpp)
    float *ref = (float *)malloc(vs * sizeof(float));
    f = fopen("/tmp/ref_logits.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), vs, f);
        fclose(f);
        double dot = 0, n_our = 0, n_ref = 0;
        for (int i = 0; i < 5000; i++) {
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos_sim = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("Cos-sim vs 40-layer ref (first 5000 logits): %.6f\n", cos_sim);
        free(ref);
    }
    
    free(x); free(final); free(logits);
    wubu_model_free(&mdl);
    return 0;
}

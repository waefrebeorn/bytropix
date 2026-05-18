/* Run only 1 layer (SSM layer 0) and compare logits */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    
    int D = D_MODEL;
    int vs = mdl.vocab_size;
    
    // Get BOS embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *f = fopen(emb_path, "rb");
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Run JUST layer 0 (SSM)
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, mdl.layers[0].attn_norm_weight, 1e-6f, normed);
    
    float *attn_out = (float *)malloc(D * sizeof(float));
    memset(attn_out, 0, D * sizeof(float));
    float *ssm_state = mdl.ssm_states + 0 * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    float *conv_state = mdl.conv_states + 0 * (CONV_KERNEL - 1) * CONV_DIM;
    wubu_ssm_forward(normed, 1, 1, &mdl.layers[0].ssm, ssm_state, conv_state, attn_out);
    
    // Residual
    for (int i = 0; i < D; i++) x[i] += attn_out[i];
    
    // Post-attention RMSNorm + MoE pass-through (no MoE loaded)
    float *normed2 = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, mdl.layers[0].post_attn_norm_weight, 1e-6f, normed2);
    for (int i = 0; i < D; i++) x[i] += normed2[i];
    
    // Final RMSNorm
    float *final = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, mdl.norm_weight, 1e-6f, final);
    
    // Output projection (full vocab)
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
    printf("1-layer (SSM L0) top-5:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    
    free(x); free(normed); free(attn_out); free(normed2); free(final); free(logits);
    wubu_model_free(&mdl);
    return 0;
}

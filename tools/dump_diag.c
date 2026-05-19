/* Full model with per-layer diagnostics: dump output stats per layer */
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
    
    int D = D_MODEL;
    
    // Get BOS embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (!emb_f) { printf("ERROR: can't open emb file\n"); return 1; }
        fseek(emb_f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, emb_f);
        fclose(emb_f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Print embedding stats
    float m=0,s=0;
    for(int i=0;i<D;i++){m+=x[i];s+=x[i]*x[i];}
    printf("EmbL0: mean=%+.6f std=%.6f norm=%.6f first5=[%.4f %.4f %.4f %.4f %.4f]\n",
           m/D, sqrtf(s/D - (m/D)*(m/D)), sqrtf(s), x[0],x[1],x[2],x[3],x[4]);
    
    // Run forward, saving layer outputs
    float *output = (float *)malloc(D * sizeof(float));
    memcpy(output, x, D * sizeof(float));
    
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];
        
        // Pre-attention RMSNorm
        float *normed = (float *)malloc(D * sizeof(float));
        wubu_rms_norm(1, 1, D, output, layer->attn_norm_weight, 1e-6f, normed);
        
        // Print normed stats
        float nm=0,ns=0;
        for(int i=0;i<D;i++){nm+=normed[i];ns+=normed[i]*normed[i];}
        
        // Attention
        float *attn_out = (float *)malloc(D * sizeof(float));
        
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }
        
        // Print attn output stats
        float am=0,as_v=0;
        for(int i=0;i<D;i++){am+=attn_out[i];as_v+=attn_out[i]*attn_out[i];}
        float a_norm = sqrtf(as_v);
        
        // Residual
        for (int i = 0; i < D; i++) output[i] += attn_out[i];
        free(attn_out);
        
        // Post-attention RMSNorm
        float *normed2 = (float *)malloc(D * sizeof(float));
        wubu_rms_norm(1, 1, D, output, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        float n2m=0,n2s=0;
        for(int i=0;i<D;i++){n2m+=normed2[i];n2s+=normed2[i]*normed2[i];}
        
        // MoE forward
        float *ffn_out = (float *)malloc(D * sizeof(float));
        if (mdl.enable_moe && mdl.gguf_ctx &&
            (mdl.moe_max_layers == 0 || l < mdl.moe_max_layers)) {
            if (wubu_moe_load_layer(mdl.gguf_ctx, l, &layer->moe)) {
                wubu_moe_forward(normed2, 1, 1, &layer->moe, ffn_out, NULL);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, D * sizeof(float));
            }
        } else {
            memcpy(ffn_out, normed2, D * sizeof(float));
        }
        
        float fm=0,fs=0;
        for(int i=0;i<D;i++){fm+=ffn_out[i];fs+=ffn_out[i]*ffn_out[i];}
        float f_norm = sqrtf(fs);
        
        // Residual
        for (int i = 0; i < D; i++) output[i] += ffn_out[i];
        
        free(normed);
        free(normed2);
        free(ffn_out);
        
        // Check for NaN
        int nan_found = 0;
        for (int i = 0; i < D; i++) if (isnan(output[i])) { nan_found = 1; break; }
        
        printf("L%02d %s: attn_norm norm=%.2f attn_norm=%.2f ffn_norm=%.2f %s\n",
               l, layer->is_ssm ? "SSM" : "GQA",
               a_norm, sqrtf(as_v), f_norm,
               nan_found ? "*** NAN ***" : "");
        
        if (nan_found) break;
    }
    
    // Final RMSNorm
    if (mdl.norm_weight) {
        float *final_normed = (float *)malloc(D * sizeof(float));
        wubu_rms_norm(1, 1, D, output, mdl.norm_weight, 1e-6f, final_normed);
        memcpy(output, final_normed, D * sizeof(float));
        free(final_normed);
    }
    
    // Output projection
    float *logits = (float *)malloc(mdl.vocab_size * sizeof(float));
    #pragma omp parallel for
    for (int j = 0; j < mdl.vocab_size; j++) {
        double sum = 0.0;
        for (int k = 0; k < D; k++)
            sum += (double)output[k] * (double)mdl.output_weight[j * D + k];
        logits[j] = (float)sum;
    }
    
    // Top-5
    int top[5] = {0}; float tv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    for (int i = 0; i < mdl.vocab_size; i++) {
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
    printf("\nFinal top-5:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    
    // Save logits
    FILE *f = fopen("/tmp/our_diag_logits.bin", "wb");
    fwrite(logits, sizeof(float), mdl.vocab_size, f);
    fclose(f);
    
    // Compare vs reference
    float *ref = (float *)malloc(mdl.vocab_size * sizeof(float));
    f = fopen("/tmp/ref_logits.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), mdl.vocab_size, f);
        fclose(f);
        double dot = 0, n_our = 0, n_ref = 0;
        for (int i = 0; i < 50000; i++) {
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos_sim = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("Cos-sim vs ref (first 50000 logits): %.6f\n", cos_sim);
        free(ref);
    }
    
    free(x); free(output); free(logits);
    wubu_model_free(&mdl);
    return 0;
}

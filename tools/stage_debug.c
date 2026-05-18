/* Dump intermediate states before and after each stage for one token */
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
    
    // Print embedding stats
    double em=0, es=0;
    for(int i=0;i<D;i++){em+=x[i];es+=x[i]*x[i];}
    printf("Embedding after load: mean=%.6f std=%.6f\n", em/D, sqrt(es/D - (em/D)*(em/D)));
    
    // Run 40 layers manually, saving after each stage
    float *output = (float *)malloc(D * sizeof(float));
    memcpy(output, x, D * sizeof(float));
    
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];
        
        // Pre-attention RMSNorm
        float *normed = (float *)malloc(D * sizeof(float));
        wubu_rms_norm(1, 1, D, output, layer->attn_norm_weight, 1e-6f, normed);
        
        // Stats of normed
        double nm=0, ns=0;
        for(int i=0;i<D;i++){nm+=normed[i];ns+=normed[i]*normed[i];}
        if (l < 3) printf("L%d pre-attn norm: mean=%.6f std=%.6f rms=%.6f\n", l, nm/D, sqrt(ns/D-(nm/D)*(nm/D)), sqrt(ns/D));
        
        // Attention
        float *attn_out = (float *)malloc(D * sizeof(float));
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out);
        } else {
            wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out);
        }
        
        double am=0, as_v=0;
        for(int i=0;i<D;i++){am+=attn_out[i];as_v+=attn_out[i]*attn_out[i];}
        if (l < 3) printf("L%d attn_out: mean=%.6f std=%.6f\n", l, am/D, sqrt(as_v/D-(am/D)*(am/D)));
        
        // Residual
        for (int i = 0; i < D; i++) output[i] += attn_out[i];
        free(attn_out);
        
        // After-attention norm
        float *normed2 = (float *)malloc(D * sizeof(float));
        wubu_rms_norm(1, 1, D, output, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        double n2m=0, n2s=0;
        for(int i=0;i<D;i++){n2m+=normed2[i];n2s+=normed2[i]*normed2[i];}
        if (l < 3) printf("L%d post-attn norm: mean=%.6f std=%.6f\n", l, n2m/D, sqrt(n2s/D-(n2m/D)*(n2m/D)));
        
        // MoE
        float *ffn_out = (float *)malloc(D * sizeof(float));
        if (mdl.enable_moe && mdl.gguf_ctx) {
            if (wubu_moe_load_layer(mdl.gguf_ctx, l, &layer->moe)) {
                wubu_moe_forward(normed2, 1, 1, &layer->moe, ffn_out);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, D * sizeof(float));
            }
        } else {
            memcpy(ffn_out, normed2, D * sizeof(float));
        }
        
        double fm=0, fs=0;
        for(int i=0;i<D;i++){fm+=ffn_out[i];fs+=ffn_out[i]*ffn_out[i];}
        if (l < 3) printf("L%d ffn_out: mean=%.6f std=%.6f\n", l, fm/D, sqrt(fs/D-(fm/D)*(fm/D)));
        
        // Residual
        for (int i = 0; i < D; i++) output[i] += ffn_out[i];
        
        double xm=0, xs=0;
        for(int i=0;i<D;i++){xm+=output[i];xs+=output[i]*output[i];}
        if (l < 3) printf("L%d residual: mean=%.6f std=%.6f\n", l, xm/D, sqrt(xs/D-(xm/D)*(xm/D)));
        
        free(normed);
        free(normed2);
        free(ffn_out);
    }
    
    // Final norm
    double xm=0, xs=0;
    for(int i=0;i<D;i++){xm+=output[i];xs+=output[i]*output[i];}
    printf("\nBefore final norm: mean=%.6f std=%.6f\n", xm/D, sqrt(xs/D-(xm/D)*(xm/D)));
    
    if (mdl.norm_weight) {
        float *final_normed = (float *)malloc(D * sizeof(float));
        wubu_rms_norm(1, 1, D, output, mdl.norm_weight, 1e-6f, final_normed);
        
        double fnm=0, fns=0;
        for(int i=0;i<D;i++){fnm+=final_normed[i];fns+=final_normed[i]*final_normed[i];}
        printf("After final norm: mean=%.6f std=%.6f rms=%.6f\n", fnm/D, sqrt(fns/D-(fnm/D)*(fnm/D)), sqrt(fns/D));
        
        memcpy(output, final_normed, D * sizeof(float));
        free(final_normed);
    }
    
    // Output projection - save hidden state and logits
    FILE *f = fopen("/tmp/our_hidden.bin", "wb");
    fwrite(output, sizeof(float), D, f);
    fclose(f);
    printf("Hidden state saved to /tmp/our_hidden.bin\n");
    
    float *logits = (float *)malloc(mdl.vocab_size * sizeof(float));
    #pragma omp parallel for
    for (int j = 0; j < mdl.vocab_size; j++) {
        double sum = 0.0;
        for (int k = 0; k < D; k++)
            sum += (double)output[k] * (double)mdl.output_weight[j * D + k];
        logits[j] = (float)sum;
    }
    
    double lm=0, ls=0;
    for(int i=0;i<50000;i++){lm+=logits[i];ls+=logits[i]*logits[i];}
    printf("Logits (first 50000): mean=%.6f std=%.6f\n", lm/50000, sqrt(ls/50000-(lm/50000)*(lm/50000)));
    
    // Compare vs ref
    float *ref = (float *)malloc(mdl.vocab_size * sizeof(float));
    f = fopen("/tmp/ref_logits_fresh.bin", "rb");
    fread(ref, sizeof(float), mdl.vocab_size, f);
    fclose(f);
    
    double dot = 0, n_our = 0, n_ref = 0;
    for (int i = 0; i < 50000; i++) {
        dot += (double)logits[i] * (double)ref[i];
        n_our += (double)logits[i] * (double)logits[i];
        n_ref += (double)ref[i] * (double)ref[i];
    }
    printf("Cos-sim: %.6f\n", dot/(sqrt(n_our)*sqrt(n_ref)));
    
    free(x); free(output); free(logits); free(ref);
    wubu_model_free(&mdl);
    return 0;
}

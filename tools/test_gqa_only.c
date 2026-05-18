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
    mdl.enable_moe = false;
    
    int D = D_MODEL;
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { printf("ERROR\n"); return 1; }
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }

    // Record what reference shows for layers 0-2 (post-MoE residual)
    // Actually the reference llm_layer_X.bin is AFTER full layer (attn+residual+ffn+residual)
    // We need to match what llama.cpp does for layer 3 GQA

    // First run layers 0-2 to get the correct input to GQA layer 3
    float *hidden = (float *)malloc(D * sizeof(float));
    memcpy(hidden, x, D * sizeof(float));
    
    // Read ref layer 2 output (our input to layer 3)
    FILE *rf = fopen("/tmp/dump_layers/ref_layer_2.bin", "rb");
    float *ref_l2 = (float *)malloc(D * sizeof(float));
    fread(ref_l2, sizeof(float), D, rf);
    fclose(rf);
    
    // Read our layer 2 output
    FILE *of = fopen("/tmp/dump_layers/our_layer_2.bin", "rb");
    float *our_l2 = (float *)malloc(D * sizeof(float));
    fread(our_l2, sizeof(float), D, of);
    fclose(of);

    printf("=== REPLACING hidden with ref L2 output (ground truth GQA input) ===\n");
    memcpy(hidden, ref_l2, D * sizeof(float));  // Use reference hidden as input to GQA
    
    // Layer 3 is GQA
    wubu_layer_t *layer = &mdl.layers[3];
    
    // Pre-attention RMSNorm
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, hidden, layer->attn_norm_weight, 1e-6f, normed);
    
    // Dump normed for debug
    FILE *d1 = fopen("/tmp/gqa_normed.bin", "wb");
    fwrite(normed, sizeof(float), D, d1);
    fclose(d1);
    
    // GQA forward
    float *attn_out = (float *)malloc(D * sizeof(float));
    wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out);
    
    printf("GQA output stats: mean=%.6f max=%.6f min=%.6f\n",
           attn_out[0], attn_out[0], attn_out[0]);
    double m=0,mx=-1e30,mn=1e30;
    for(int i=0;i<D;i++){m+=attn_out[i];if(attn_out[i]>mx)mx=attn_out[i];if(attn_out[i]<mn)mn=attn_out[i];}
    printf("  Full: mean=%.6f max=%.6f min=%.6f\n", m/D, mx, mn);
    
    // Save GQA output
    FILE *dg = fopen("/tmp/gqa_attn_out.bin", "wb");
    fwrite(attn_out, sizeof(float), D, dg);
    fclose(dg);
    
    // Residual: x += attn_out
    float *post_attn = (float *)malloc(D * sizeof(float));
    memcpy(post_attn, hidden, D * sizeof(float));
    for (int i = 0; i < D; i++) post_attn[i] += attn_out[i];
    
    // Post-attention norm
    float *normed2 = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, post_attn, layer->post_attn_norm_weight, 1e-6f, normed2);
    
    // MoE pass-through
    float *ffn_out = (float *)malloc(D * sizeof(float));
    memcpy(ffn_out, normed2, D * sizeof(float));
    
    // Final residual
    for (int i = 0; i < D; i++) hidden[i] += ffn_out[i];
    
    // Save our L3 output
    FILE *d3 = fopen("/tmp/gqa_our_layer3.bin", "wb");
    fwrite(hidden, sizeof(float), D, d3);
    fclose(d3);
    
    // Compare with ref L3
    float *ref_l3 = (float *)malloc(D * sizeof(float));
    rf = fopen("/tmp/dump_layers/ref_layer_3.bin", "rb");
    fread(ref_l3, sizeof(float), D, rf);
    fclose(rf);
    
    double dot=0, n1=0, n2=0;
    for (int i = 0; i < D; i++) {
        dot += (double)hidden[i] * (double)ref_l3[i];
        n1 += (double)hidden[i] * (double)hidden[i];
        n2 += (double)ref_l3[i] * (double)ref_l3[i];
    }
    printf("Our L3 (with ref L2 input) vs ref L3: cos=%.6f\n", 
           dot / (sqrt(n1) * sqrt(n2)));
    
    free(x); free(hidden); free(normed); free(attn_out);
    free(post_attn); free(normed2); free(ffn_out);
    free(ref_l2); free(our_l2); free(ref_l3);
    wubu_model_free(&mdl);
    return 0;
}

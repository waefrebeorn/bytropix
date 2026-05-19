/**
 * tools/ssm_layer0_dump.c
 *
 * Run SSM Layer 0 of the model for a single token and dump intermediates.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void dump_bin(const char *path, const float *data, int n) {
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
    else fprintf(stderr, "WARN: can't write %s\n", path);
}

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = false;
    
    int D = D_MODEL;
    
    // Get BOS embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *emb_f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_f) { printf("ERROR: can't open emb file\n"); return 1; }
        fseek(emb_f, 248044LL * D * sizeof(float), SEEK_SET);
        size_t nr = fread(x, sizeof(float), D, emb_f);
        if (nr != (size_t)D) { printf("ERROR: emb read failed\n"); return 1; }
        fclose(emb_f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    dump_bin("/tmp/dbg_emb.bin", x, D);
    printf("Emb[0..4]: %.6f %.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], x[3], x[4]);
    
    // Layer 0
    wubu_layer_t *layer = &mdl.layers[0];
    
    // Pre-attention RMSNorm
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, layer->attn_norm_weight, 1e-6f, normed);
    dump_bin("/tmp/dbg_norm.bin", normed, D);
    {
        float nm = 0, ns = 0;
        for (int i = 0; i < D; i++) { nm += normed[i]; ns += normed[i] * normed[i]; }
        printf("Norm: mean=%.6f std=%.6f\n", nm/D, sqrtf(ns/D - (nm/D)*(nm/D)));
    }
    
    // SSM forward (B=1, T=1)
    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *attn_out = (float *)malloc(D * sizeof(float));
    
    setenv("DUMP_SSM_DEBUG", "1", 1);
    wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out);
    unsetenv("DUMP_SSM_DEBUG");
    
    printf("\nSSM output[0..4]: %.6f %.6f %.6f %.6f %.6f\n", 
           attn_out[0], attn_out[1], attn_out[2], attn_out[3], attn_out[4]);
    {
        float om = 0, os = 0;
        for (int i = 0; i < D; i++) { om += attn_out[i]; os += attn_out[i] * attn_out[i]; }
        printf("SSM out: mean=%.6f std=%.6f\n", om/D, sqrtf(os/D - (om/D)*(om/D)));
    }
    
    // Residual
    for (int i = 0; i < D; i++) x[i] += attn_out[i];
    dump_bin("/tmp/dbg_full_out.bin", x, D);
    
    printf("\nFinal output[0..4]: %.6f %.6f %.6f %.6f %.6f\n", 
           x[0], x[1], x[2], x[3], x[4]);
    {
        float fm = 0, fs = 0;
        for (int i = 0; i < D; i++) { fm += x[i]; fs += x[i] * x[i]; }
        printf("Final: mean=%.6f std=%.6f\n", fm/D, sqrtf(fs/D - (fm/D)*(fm/D)));
    }
    
    free(x);
    free(normed);
    free(ssm_state);
    free(conv_state);
    free(attn_out);
    wubu_model_free(&mdl);
    return 0;
}

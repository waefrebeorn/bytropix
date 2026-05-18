#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Test F32 fallback path by temporarily clearing quantized pointers
static void clear_quantized_ptrs(wubu_model_t *mdl) {
    for (int l = 0; l < mdl->n_layers; l++) {
        wubu_layer_t *layer = &mdl->layers[l];
        if (layer->is_ssm) {
            layer->ssm.attn_qkv_weight_q = NULL;
            layer->ssm.attn_gate_weight_q = NULL;
            layer->ssm.ssm_out_weight_q = NULL;
        } else {
            layer->gqa.attn_q_weight_q = NULL;
            layer->gqa.attn_k_weight_q = NULL;
            layer->gqa.attn_v_weight_q = NULL;
            layer->gqa.attn_output_weight_q = NULL;
        }
    }
    mdl->output_weight_q = NULL; // Also force F32 output proj
}

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    // Override: force F32 fallback for ALL projections
    clear_quantized_ptrs(&mdl);
    mdl.enable_moe = false;
    
    int D = D_MODEL;
    int vs = mdl.vocab_size;
    
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
    
    float *logits = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    float *ref = (float *)malloc(vs * sizeof(float));
    FILE *f = fopen("/tmp/llama_logits_new.bin", "rb");
    if (f) {
        fread(ref, sizeof(float), vs, f);
        fclose(f);
        double dot=0, n_our=0, n_ref=0, max_diff=0;
        int max_idx = -1;
        for (int i = 0; i < vs; i++) {
            double d = (double)logits[i] - (double)ref[i];
            if (fabs(d) > max_diff) { max_diff = fabs(d); max_idx = i; }
            dot += (double)logits[i] * (double)ref[i];
            n_our += (double)logits[i] * (double)logits[i];
            n_ref += (double)ref[i] * (double)ref[i];
        }
        double cos = dot / (sqrt(n_our) * sqrt(n_ref));
        printf("F32 FALLBACK - cos-sim: %.10f\n", cos);
        printf("max diff: %.10f at idx %d\n", max_diff, max_idx);
    }
    
    free(ref); free(logits); free(x);
    wubu_model_free(&mdl);
    return 0;
}

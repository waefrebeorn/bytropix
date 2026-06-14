/**
 * dump_hidden.c — Compare CPU vs GPU hidden states per-layer to find error accumulation.
 * GPU MoE enabled. State properly isolated between runs.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = true;
    
    int gpu_ok = 0;
    if (getenv("GPU")) {
        gpu_ok = wubu_model_gpu_init(&mdl, 4096, 256);
        fprintf(stderr, "GPU init: %s\n", gpu_ok ? "OK" : "FAILED");
    }
    if (!gpu_ok) { wubu_model_free(&mdl); return 1; }
    
    float embd[D_MODEL];
    if (mdl.token_embd)
        memcpy(embd, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    else
        memset(embd, 0, D_MODEL * sizeof(float));
    
    // Save initial state
    size_t ssm_sz = (size_t)mdl.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float);
    size_t conv_sz = (size_t)mdl.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float);
    float *saved_ssm = (float *)malloc(ssm_sz);
    float *saved_conv = (float *)malloc(conv_sz);
    memcpy(saved_ssm, mdl.ssm_states, ssm_sz);
    memcpy(saved_conv, mdl.conv_states, conv_sz);
    int saved_cache = mdl.gqa_cache_len;
    
    // Manually run CPU forward layer by layer, saving intermediate hidden states
    // (We can't use wubu_model_forward_from_embd because it internally does MoE)
    float *cpu_x = (float *)malloc(D_MODEL * sizeof(float));
    memcpy(cpu_x, embd, D_MODEL * sizeof(float));
    float *tmp = (float *)malloc(D_MODEL * sizeof(float));
    float *tmp2 = (float *)malloc(D_MODEL * sizeof(float));
    float *tmp3 = (float *)malloc(D_MODEL * sizeof(float));
    float **layer_hidden = (float **)calloc(mdl.n_layers, sizeof(float *));
    
    fprintf(stderr, "CPU forward...\n");
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];
        wubu_rms_norm(1, 1, D_MODEL, cpu_x, layer->attn_norm_weight, 1e-6f, tmp);
        if (layer->is_ssm) {
            float *ss = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *cs = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(tmp, 1, 1, &layer->ssm, ss, cs, tmp2, NULL, NULL);
        } else {
            int li = 0; for (int j = 0; j < l; j++) if (!mdl.layers[j].is_ssm) li++;
            int64_t off = (int64_t)li * GQA_MAX_CTX * GQA_KV_DIM;
            void *kc = (uint8_t *)mdl.gqa_k_cache + off * sizeof(block_q4_0_cache);
            void *vc = (uint8_t *)mdl.gqa_v_cache + off * sizeof(block_q4_0_cache);
            void *ko = (mdl.gqa_cache_len == 0) ? kc : (uint8_t*)kc + mdl.gqa_cache_len * GQA_KV_DIM * sizeof(block_q4_0_cache);
            void *vo = ko;
            wubu_gqa_forward(tmp, 1, 1, &layer->gqa, tmp2, NULL, NULL, 0, ko, vo, layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads);
            mdl.gqa_cache_len++;
        }
        for (int i = 0; i < D_MODEL; i++) cpu_x[i] += tmp2[i];
        wubu_rms_norm(1, 1, D_MODEL, cpu_x, layer->post_attn_norm_weight, 1e-6f, tmp);
        
        // CPU MoE — ensure gpu_ctx is NULL
        layer->moe.gpu_ctx = NULL;
        wubu_moe_forward(tmp, 1, 1, &layer->moe, tmp2, NULL);
        layer->moe.gpu_ctx = NULL;
        
        // Save layer output
        layer_hidden[l] = (float *)malloc(D_MODEL * sizeof(float));
        for (int i = 0; i < D_MODEL; i++) {
            cpu_x[i] += tmp2[i];
            layer_hidden[l][i] = cpu_x[i];
        }
    }
    
    // Restore state for GPU run
    fprintf(stderr, "Restoring state for GPU...\n");
    memcpy(mdl.ssm_states, saved_ssm, ssm_sz);
    memcpy(mdl.conv_states, saved_conv, conv_sz);
    mdl.gqa_cache_len = saved_cache;
    for (int l = 0; l < mdl.n_layers; l++) {
        if (mdl.layers[l].is_ssm) {
            wubu_gpu_sync_ssm_state_to_gpu(mdl.gpu_ctx, l,
                mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE,
                mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM);
        }
    }
    
    // GPU forward
    fprintf(stderr, "GPU forward...\n");
    float *gpu_x = (float *)malloc(D_MODEL * sizeof(float));
    memcpy(gpu_x, embd, D_MODEL * sizeof(float));
    
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];
        wubu_rms_norm(1, 1, D_MODEL, gpu_x, layer->attn_norm_weight, 1e-6f, tmp);
        if (layer->is_ssm) {
            float *ss = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *cs = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_gpu_sync_ssm_state_to_gpu(mdl.gpu_ctx, l, ss, cs);
            wubu_gpu_set_ssm_hybrid(mdl.gpu_ctx, l, &layer->ssm);
            wubu_ssm_forward(tmp, 1, 1, &layer->ssm, ss, cs, tmp2, NULL, NULL);
            layer->ssm.gpu_ssm_state = NULL;
            layer->ssm.gpu_stream = NULL;
        } else {
            int li = 0; for (int j = 0; j < l; j++) if (!mdl.layers[j].is_ssm) li++;
            int64_t off = (int64_t)li * GQA_MAX_CTX * GQA_KV_DIM;
            void *kc = (uint8_t *)mdl.gqa_k_cache + off * sizeof(block_q4_0_cache);
            void *vc = (uint8_t *)mdl.gqa_v_cache + off * sizeof(block_q4_0_cache);
            void *ko = (mdl.gqa_cache_len == 0) ? kc : (uint8_t*)kc + mdl.gqa_cache_len * GQA_KV_DIM * sizeof(block_q4_0_cache);
            void *vo = ko;
            wubu_gqa_forward(tmp, 1, 1, &layer->gqa, tmp2, NULL, NULL, 0, ko, vo, layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads);
            mdl.gqa_cache_len++;
        }
        for (int i = 0; i < D_MODEL; i++) gpu_x[i] += tmp2[i];
        wubu_rms_norm(1, 1, D_MODEL, gpu_x, layer->post_attn_norm_weight, 1e-6f, tmp);
        
        // GPU MoE
        layer->moe.gpu_ctx = (void *)&mdl;
        wubu_moe_forward(tmp, 1, 1, &layer->moe, tmp2, NULL);
        layer->moe.gpu_ctx = NULL;
        
        for (int i = 0; i < D_MODEL; i++) gpu_x[i] += tmp2[i];
        
        // Compare with CPU layer output
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < D_MODEL; i++) {
            dot += (double)layer_hidden[l][i] * (double)gpu_x[i];
            n1  += (double)layer_hidden[l][i] * (double)layer_hidden[l][i];
            n2  += (double)gpu_x[i] * (double)gpu_x[i];
        }
        printf("Layer %2d (%s): cos-sim=%.6f\n",
               l, layer->is_ssm ? "SSM" : "GQA",
               n1 > 0 && n2 > 0 ? dot / (sqrt(n1) * sqrt(n2)) : 1.0);
    }
    
    // Final cos-sim
    double dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < D_MODEL; i++) {
        dot += (double)layer_hidden[mdl.n_layers-1][i] * (double)gpu_x[i];
        n1  += (double)layer_hidden[mdl.n_layers-1][i] * (double)layer_hidden[mdl.n_layers-1][i];
        n2  += (double)gpu_x[i] * (double)gpu_x[i];
    }
    printf("\nFinal cos-sim: %.6f\n", dot / (sqrt(n1) * sqrt(n2)));
    
    for (int l = 0; l < mdl.n_layers; l++) free(layer_hidden[l]);
    free(layer_hidden); free(cpu_x); free(gpu_x); free(tmp); free(tmp2); free(tmp3);
    free(saved_ssm); free(saved_conv);
    wubu_model_free(&mdl);
    return 0;
}

/**
 * test_moe_accum.c — Compare CPU vs GPU MoE error accumulation across layers.
 * Runs one full model forward with CPU MoE, saves all layer MoE inputs.
 * Then runs with GPU MoE, comparing per-layer MoE outputs.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
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
        printf("GPU init: %s\n", gpu_ok ? "OK" : "FAILED");
    }
    if (!gpu_ok) { wubu_model_free(&mdl); return 1; }

    // Use embedding of token 279
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

    // === CPU run: save per-layer MoE input (normed2) ===
    float **cpu_moe_in = (float **)calloc(mdl.n_layers, sizeof(float *));
    float **cpu_moe_out = (float **)calloc(mdl.n_layers, sizeof(float *));
    
    // Forward without GPU context
    mdl.skip_output_proj = true;
    
    // We need to hack into the forward to capture MoE inputs/outputs
    // Easiest: modify wubu_model.c to save them? No, too intrusive.
    // Instead, let's manually run layer-by-layer.
    
    printf("CPU: running layer-by-layer forward...\n");
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    memcpy(x, embd, D_MODEL * sizeof(float));
    float *normed = (float *)malloc(D_MODEL * sizeof(float));
    float *attn_out = (float *)malloc(D_MODEL * sizeof(float));
    float *normed2 = (float *)malloc(D_MODEL * sizeof(float));
    float *ffn_out = (float *)malloc(D_MODEL * sizeof(float));
    
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];
        
        // Pre-attention RMSNorm
        wubu_rms_norm(1, 1, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        
        // Attention
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
        } else {
            // GQA (CPU)
            int l_gqa = 0;
            for (int li = 0; li < l; li++) if (!mdl.layers[li].is_ssm) l_gqa++;
            int64_t layer_cache_off = (int64_t)l_gqa * GQA_MAX_CTX * GQA_KV_DIM;
            void *k_cache = (uint8_t *)mdl.gqa_k_cache + layer_cache_off * sizeof(block_q4_0_cache);
            void *v_cache = (uint8_t *)mdl.gqa_v_cache + layer_cache_off * sizeof(block_q4_0_cache);
            void *k_out = (mdl.gqa_cache_len == 0) ? k_cache : NULL;
            void *v_out = k_out;
            wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out,
                NULL, NULL, 0, k_out, v_out);
            mdl.gqa_cache_len++;
        }
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) x[i] += attn_out[i];
        
        // Post-attention RMSNorm
        wubu_rms_norm(1, 1, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        // Save MoE input for later comparison
        cpu_moe_in[l] = (float *)malloc(D_MODEL * sizeof(float));
        cpu_moe_out[l] = (float *)malloc(D_MODEL * sizeof(float));
        memcpy(cpu_moe_in[l], normed2, D_MODEL * sizeof(float));
        
        // MoE (CPU — gpu_ctx is NULL since we never initialized GPU in this path)
        // But wait: mdl.gpu_ctx IS set since we called wubu_model_gpu_init!
        // We need to force CPU MoE by not setting moe.gpu_ctx
        // Actually, the MoE forward checks w->gpu_ctx which is layer->moe.gpu_ctx
        // This is set by wubu_model.c during wubu_model_forward_from_embd, not here.
        // So layer->moe.gpu_ctx should be NULL since we haven't set it.
        mdl.layers[l].moe.gpu_ctx = NULL;  // force CPU
        wubu_moe_forward(normed2, 1, 1, &layer->moe, ffn_out, NULL);
        memcpy(cpu_moe_out[l], ffn_out, D_MODEL * sizeof(float));
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) x[i] += ffn_out[i];
    }
    printf("CPU done.\n");
    
    // === Restore state for GPU run ===
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
    
    // === GPU run: per-layer MoE ===
    printf("GPU: running layer-by-layer forward...\n");
    memcpy(x, embd, D_MODEL * sizeof(float));
    mdl.gqa_cache_len = 0;  // Reset GQA cache
    
    double total_dot = 0, total_n1 = 0, total_n2 = 0;
    
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *layer = &mdl.layers[l];
        
        // Pre-attention RMSNorm
        wubu_rms_norm(1, 1, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        
        // Attention
        if (layer->is_ssm) {
            float *ssm_state = mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            // GPU hybrid path
            if (mdl.gpu_ctx) {
                wubu_gpu_sync_ssm_state_to_gpu(mdl.gpu_ctx, l, ssm_state, conv_state);
                wubu_gpu_set_ssm_hybrid(mdl.gpu_ctx, l, &layer->ssm);
                wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
                layer->ssm.gpu_ssm_state = NULL;
                layer->ssm.gpu_stream = NULL;
            } else {
                wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
            }
        } else {
            // GQA (CPU for T=1)
            int l_gqa = 0;
            for (int li = 0; li < l; li++) if (!mdl.layers[li].is_ssm) l_gqa++;
            int64_t layer_cache_off = (int64_t)l_gqa * GQA_MAX_CTX * GQA_KV_DIM;
            void *k_cache = (uint8_t *)mdl.gqa_k_cache + layer_cache_off * sizeof(block_q4_0_cache);
            void *v_cache = (uint8_t *)mdl.gqa_v_cache + layer_cache_off * sizeof(block_q4_0_cache);
            void *k_out = (mdl.gqa_cache_len == 0) ? k_cache : NULL;
            void *v_out = k_out;
            wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out,
                NULL, NULL, 0, k_out, v_out);
            mdl.gqa_cache_len++;
        }
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) x[i] += attn_out[i];
        
        // Post-attention RMSNorm
        wubu_rms_norm(1, 1, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        // GPU MoE
        mdl.layers[l].moe.gpu_ctx = (void *)&mdl;
        wubu_moe_forward(normed2, 1, 1, &layer->moe, ffn_out, NULL);
        mdl.layers[l].moe.gpu_ctx = NULL;
        
        // Compare MoE input and output with CPU
        double moe_out_dot = 0, moe_out_n1 = 0, moe_out_n2 = 0;
        double max_diff = 0;
        for (int i = 0; i < D_MODEL; i++) {
            float cv = cpu_moe_out[l][i];
            float gv = ffn_out[i];
            moe_out_dot += (double)cv * (double)gv;
            moe_out_n1 += (double)cv * (double)cv;
            moe_out_n2 += (double)gv * (double)gv;
            double diff = fabs((double)cv - (double)gv);
            if (diff > max_diff) max_diff = diff;
        }
        double cos = moe_out_dot / (sqrt(moe_out_n1) * sqrt(moe_out_n2));
        
        // Also compare the full residual state x
        for (int i = 0; i < D_MODEL; i++) {
            total_dot += (double)cpu_moe_out[l][i] * (double)(x[i] - ffn_out[i] + cpu_moe_out[l][i]);
        }
        // Actually simpler: just accumulate for final hidden state
        
        printf("Layer %2d (%s): MoE cos-sim=%.6f max_diff=%.2e\n",
               l, layer->is_ssm ? "SSM" : "GQA", cos, max_diff);
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) x[i] += ffn_out[i];
    }
    
    // Compare final hidden state
    // Reconstruct CPU final hidden: start from embd, add attn_out + ffn_out per layer
    float cpu_final[D_MODEL];
    memcpy(cpu_final, embd, D_MODEL * sizeof(float));
    // Actually we can't reconstruct easily. Let's just dump the final x.
    // CPU final x = sum over layers of attention(L_i) + MoE(L_i)
    // GPU final x = same but with GPU MoE
    
    free(x); free(normed); free(attn_out); free(normed2); free(ffn_out);
    for (int l = 0; l < mdl.n_layers; l++) { free(cpu_moe_in[l]); free(cpu_moe_out[l]); }
    free(cpu_moe_in); free(cpu_moe_out);
    free(saved_ssm); free(saved_conv);
    wubu_model_free(&mdl);
    return 0;
}

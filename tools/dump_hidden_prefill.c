/**
 * dump_hidden_prefill.c — Compare CPU vs GPU hidden states for prefill (N>1).
 * Tests GPU GQA (active for N>1) and GPU SSM (full forward or hybrid).
 * GPU MoE is also active.
 * PROPER state isolation between CPU and GPU runs.
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
        printf("GPU init: %s\n", gpu_ok ? "OK" : "FAILED");
    }
    
    // Prefill with N=3 tokens
    const int N_TOKENS = 3;
    float embd[N_TOKENS * D_MODEL];
    if (mdl.token_embd) {
        // Tokens 279 (" the"), 264 ("of"), 315 ("to")
        int tokens[3] = {279, 264, 315};
        for (int i = 0; i < N_TOKENS; i++)
            memcpy(embd + i * D_MODEL, mdl.token_embd + (long long)tokens[i] * D_MODEL, D_MODEL * sizeof(float));
    } else {
        memset(embd, 0, N_TOKENS * D_MODEL * sizeof(float));
    }
    
    // Save initial state
    size_t ssm_state_bytes = (size_t)mdl.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float);
    size_t conv_state_bytes = (size_t)mdl.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float);
    
    float *saved_ssm_state = (float *)malloc(ssm_state_bytes);
    float *saved_conv_state = (float *)malloc(conv_state_bytes);
    memcpy(saved_ssm_state, mdl.ssm_states, ssm_state_bytes);
    memcpy(saved_conv_state, mdl.conv_states, conv_state_bytes);
    int saved_cache_len = mdl.gqa_cache_len;
    
    // === CPU forward (skip output proj) ===
    float cpu_hidden[D_MODEL * N_TOKENS];
    mdl.skip_output_proj = true;
    wubu_model_forward_from_embd(&mdl, embd, 1, N_TOKENS, cpu_hidden);
    printf("CPU[0]: max=%.4f [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
           (float)(fabsf(cpu_hidden[0])), cpu_hidden[0], cpu_hidden[1], cpu_hidden[2], cpu_hidden[3]);
    printf("CPU[1]: max=%.4f [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
           (float)(fabsf(cpu_hidden[D_MODEL])), cpu_hidden[D_MODEL], cpu_hidden[D_MODEL+1], cpu_hidden[D_MODEL+2], cpu_hidden[D_MODEL+3]);
    
    // Restore state for GPU run
    memcpy(mdl.ssm_states, saved_ssm_state, ssm_state_bytes);
    memcpy(mdl.conv_states, saved_conv_state, conv_state_bytes);
    mdl.gqa_cache_len = saved_cache_len;
    
    if (gpu_ok) {
        for (int l = 0; l < mdl.n_layers; l++) {
            if (mdl.layers[l].is_ssm) {
                wubu_gpu_sync_ssm_state_to_gpu(mdl.gpu_ctx, l,
                    mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE,
                    mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM);
            }
        }
    }
    
    // === GPU forward ===
    if (gpu_ok) {
        float gpu_hidden[D_MODEL * N_TOKENS];
        wubu_model_forward_from_embd(&mdl, embd, 1, N_TOKENS, gpu_hidden);
        printf("GPU[0]: max=%.4f [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
               (float)(fabsf(gpu_hidden[0])), gpu_hidden[0], gpu_hidden[1], gpu_hidden[2], gpu_hidden[3]);
        printf("GPU[1]: max=%.4f [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
               (float)(fabsf(gpu_hidden[D_MODEL])), gpu_hidden[D_MODEL], gpu_hidden[D_MODEL+1], gpu_hidden[D_MODEL+2], gpu_hidden[D_MODEL+3]);
        
        // Cos-sim for first token only
        double dot0 = 0, n1_0 = 0, n2_0 = 0;
        for (int i = 0; i < D_MODEL; i++) {
            dot0 += (double)cpu_hidden[i] * (double)gpu_hidden[i];
            n1_0 += (double)cpu_hidden[i] * (double)cpu_hidden[i];
            n2_0 += (double)gpu_hidden[i] * (double)gpu_hidden[i];
        }
        printf("Cos-sim token[0]: %.6f\n", dot0 / (sqrt(n1_0) * sqrt(n2_0)));
        
        // Cos-sim for second token
        double dot1 = 0, n1_1 = 0, n2_1 = 0;
        for (int i = 0; i < D_MODEL; i++) {
            dot1 += (double)cpu_hidden[D_MODEL + i] * (double)gpu_hidden[D_MODEL + i];
            n1_1 += (double)cpu_hidden[D_MODEL + i] * (double)cpu_hidden[D_MODEL + i];
            n2_1 += (double)gpu_hidden[D_MODEL + i] * (double)gpu_hidden[D_MODEL + i];
        }
        printf("Cos-sim token[1]: %.6f\n", dot1 / (sqrt(n1_1) * sqrt(n2_1)));
    }
    
    free(saved_ssm_state);
    free(saved_conv_state);
    wubu_model_free(&mdl);
    return 0;
}

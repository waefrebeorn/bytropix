/**
 * Debug: compare GPU vs CPU SSM after each token.
 * Run N tokens, compare output + state after each.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz);
extern "C" int wubu_model_gpu_ssm_forward_full(wubu_model_t *model, int layer_idx,
    const float *h_norm, int C, float *h_attn_out);
extern "C" void wubu_model_gpu_free(wubu_model_t *model);

static double cos_sim(const float *a, const float *b, int n) {
    double dot=0, na=0, nb=0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na)*sqrt(nb) + 1e-30);
}

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    wubu_model_t mdl;
    memset(&mdl, 0, sizeof(mdl));
    if (!wubu_model_init(&mdl, gguf_path)) return 1;
    if (!wubu_model_gpu_init(&mdl, 4096, 1024)) return 1;

    const int L = 0, N = 1, N_TOKENS = 5;
    float *h_in = (float*)malloc(N * D_MODEL * sizeof(float));
    float *gpu_out = (float*)calloc(N * D_MODEL, sizeof(float));
    float *cpu_out = (float*)calloc(N * D_MODEL, sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    // CPU state buffers
    float *ssm_state = mdl.ssm_states + L * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    float *conv_state = mdl.conv_states + L * (CONV_KERNEL - 1) * CONV_DIM;
    
    for (int step = 0; step < N_TOKENS; step++) {
        // Same random-ish input
        for (int i = 0; i < D_MODEL; i++) {
            h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        }
        
        // GPU
        wubu_model_gpu_ssm_forward_full(&mdl, L, h_in, N, gpu_out);
        
        // CPU
        memset(ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
        memset(conv_state, 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
        wubu_ssm_forward(h_in, 1, N, &mdl.layers[L].ssm, ssm_state, conv_state, cpu_out, NULL, NULL);
        
        double cs = cos_sim(gpu_out, cpu_out, D_MODEL);
        printf("Step %d: cos-sim=%.6f  gpu[0]=%.6f cpu[0]=%.6f  gpu_max=%.4f cpu_max=%.4f\n",
               step, cs, gpu_out[0], cpu_out[0],
               (double)fabs(gpu_out[0]), (double)fabs(cpu_out[0]));
        
        // Re-seed for same input, but now GPU state persisted and CPU state was reset
        // Next step needs SAME input on both but only GPU has accumulated state
        break;  // Just do first step for now - need to fix the comparison
    }

    // Now test: both start with zero state, same input, step 1
    printf("\n--- Same initial state test ---\n");
    srand(42);
    for (int i = 0; i < D_MODEL; i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    
    // Reset ALL states to zero
    for (int l = 0; l < mdl.n_layers; l++) {
        if (mdl.layers[l].is_ssm) {
            memset(mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, 0,
                   SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
            memset(mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM, 0,
                   (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
        }
    }
    
    // GPU states are reset by re-initializing
    // Hmm, need GPU state reset too. Let me just compare first call output more carefully.
    
    // Use debug approach: dump intermediates from both
    float *gpu_first = (float*)calloc(D_MODEL, sizeof(float));
    float *cpu_first = (float*)calloc(D_MODEL, sizeof(float));
    
    // Reset CPU state
    memset(ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    memset(conv_state, 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    
    // GPU first call with fresh input
    for (int i = 0; i < D_MODEL; i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    wubu_model_gpu_ssm_forward_full(&mdl, L, h_in, N, gpu_first);
    
    // Reset seed for CPU with same input
    srand(42);
    // Skip first rand() call... wait, srand(42) was already called above. The first call consumed the input.
    
    free(gpu_first);
    free(cpu_first);
    
    free(h_in); free(gpu_out); free(cpu_out);
    wubu_model_gpu_free(&mdl);
    wubu_model_free(&mdl);
    return 0;
}

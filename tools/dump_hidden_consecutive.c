/**
 * dump_hidden_consecutive.c — Check if GPU with MoE stays correct across multiple decode steps.
 * Compares: GPU forward(token1) + GPU forward(token2) vs GPU forward(token1) + GPU forward(token2)
 * If they match, GPU MoE doesn't diverge.
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
    if (!gpu_ok) { wubu_model_free(&mdl); return 1; }
    
    // Save initial state BEFORE any forward
    size_t ssm_sz = (size_t)mdl.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float);
    size_t conv_sz = (size_t)mdl.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float);
    float *saved_ssm = (float *)malloc(ssm_sz);
    float *saved_conv = (float *)malloc(conv_sz);
    memcpy(saved_ssm, mdl.ssm_states, ssm_sz);
    memcpy(saved_conv, mdl.conv_states, conv_sz);
    int saved_cache_len = mdl.gqa_cache_len;
    
    // Get embedding
    float embd[D_MODEL];
    if (mdl.token_embd) {
        memcpy(embd, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    } else {
        memset(embd, 0, D_MODEL * sizeof(float));
    }
    printf("embd[0]=%.6f embd[1]=%.6f\n", embd[0], embd[1]);
    
    // === Run A: process token once ===
    float hidden_a[D_MODEL];
    mdl.skip_output_proj = true;
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, hidden_a);
    printf("A (1st tok): [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f\n",
           hidden_a[0], hidden_a[1], hidden_a[2], hidden_a[3]);
    
    // Run A second token
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, hidden_a);
    printf("A (2nd tok): [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f\n",
           hidden_a[0], hidden_a[1], hidden_a[2], hidden_a[3]);
    
    // === Restore state ===
    memcpy(mdl.ssm_states, saved_ssm, ssm_sz);
    memcpy(mdl.conv_states, saved_conv, conv_sz);
    mdl.gqa_cache_len = saved_cache_len;
    for (int l = 0; l < mdl.n_layers; l++) {
        if (mdl.layers[l].is_ssm) {
            wubu_gpu_sync_ssm_state_to_gpu(mdl.gpu_ctx, l,
                mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE,
                mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM);
        }
    }
    
    // === Run B: process token once (should match A's 1st tok) ===
    float hidden_b[D_MODEL];
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, hidden_b);
    printf("B (1st tok): [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f\n",
           hidden_b[0], hidden_b[1], hidden_b[2], hidden_b[3]);
    
    double dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < D_MODEL; i++) {
        dot += (double)hidden_a[i] * (double)hidden_b[i];
        n1 += (double)hidden_a[i] * (double)hidden_a[i];
        n2 += (double)hidden_b[i] * (double)hidden_b[i];
    }
    // a is 2nd tok, b is 1st tok — they SHOULD differ since state is different (A has state from tok1, B has fresh state)
    printf("A(2nd) vs B(1st) Cos-sim: %.6f (expected: different)\\n", dot / (sqrt(n1) * sqrt(n2)));
    
    // Now save B's state and run second token
    float *b_ssm = (float *)malloc(ssm_sz);
    float *b_conv = (float *)malloc(conv_sz);
    memcpy(b_ssm, mdl.ssm_states, ssm_sz);
    memcpy(b_conv, mdl.conv_states, conv_sz);
    int b_cache = mdl.gqa_cache_len;
    
    // B second token
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, hidden_b);
    printf("B (2nd tok): [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f\n",
           hidden_b[0], hidden_b[1], hidden_b[2], hidden_b[3]);
    
    // Compare A(2nd) vs B(2nd) — should match
    dot = 0; n1 = 0; n2 = 0;
    for (int i = 0; i < D_MODEL; i++) {
        dot += (double)hidden_a[i] * (double)hidden_b[i];
        n1 += (double)hidden_a[i] * (double)hidden_a[i];
        n2 += (double)hidden_b[i] * (double)hidden_b[i];
    }
    printf("A(2nd) vs B(2nd) Cos-sim: %.6f (expected: 1.0)\\n", dot / (sqrt(n1) * sqrt(n2)));
    
    free(saved_ssm); free(saved_conv);
    free(b_ssm); free(b_conv);
    wubu_model_free(&mdl);
    return 0;
}

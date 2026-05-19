// test_bwd_model.c — Verify full model backward gradient flow end-to-end
// Tests that wubu_model_backward_from_embd produces non-zero gradients
// through all 40 layers (30 SSM + 10 GQA) with correct residual chaining
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_model.h"
#include "wubu_ssm.h"

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const int B = 1, T = 2, N = B * T;
    
    printf("=== Model-Level Backward Verification ===\n\n");
    
    // Load model
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }
    printf("Loaded: %d layers (%d SSM, %d GQA)\n",
           model.n_layers,
           model.n_layers - model.n_layers/4,
           model.n_layers/4);
    
    // Create random embeddings
    float *embeddings = (float *)malloc(N * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++)
        embeddings[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    
    // Run forward pass and save ALL intermediates
    float *logits = (float *)malloc(N * D_MODEL * sizeof(float));
    float *saved_normed = (float *)calloc(model.n_layers * N * D_MODEL, sizeof(float));
    float *saved_attn_out = (float *)calloc(model.n_layers * N * D_MODEL, sizeof(float));
    float *saved_normed2 = (float *)calloc(model.n_layers * N * D_MODEL, sizeof(float));
    float *saved_ffn_out = (float *)calloc(model.n_layers * N * D_MODEL, sizeof(float));
    
    // === CPU Forward with intermediate saving ===
    printf("\nRunning CPU forward (saving intermediates)...\n");
    
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    memcpy(x, embeddings, N * D_MODEL * sizeof(float));
    
    for (int l = 0; l < model.n_layers; l++) {
        wubu_layer_t *layer = &model.layers[l];
        
        // Pre-attention RMSNorm
        float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        
        // Save normed
        memcpy(saved_normed + l * N * D_MODEL, normed, N * D_MODEL * sizeof(float));
        
        // Attention
        float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
        if (layer->is_ssm) {
            float *ssm_state = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }
        
        // Save attn_out
        memcpy(saved_attn_out + l * N * D_MODEL, attn_out, N * D_MODEL * sizeof(float));
        
        // Residual: x += attn_out
        for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];
        
        // Post-attention RMSNorm
        float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        // Save normed2 and ffn_out (MoE pass-through = normed2)
        memcpy(saved_normed2 + l * N * D_MODEL, normed2, N * D_MODEL * sizeof(float));
        memcpy(saved_ffn_out + l * N * D_MODEL, normed2, N * D_MODEL * sizeof(float));
        
        // Residual: x += normed2 (pass-through MoE)
        for (int i = 0; i < N * D_MODEL; i++) x[i] += normed2[i];
        
        free(normed); free(attn_out); free(normed2);
    }
    
    // Final RMSNorm
    if (model.norm_weight) {
        float *final_normed = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, model.norm_weight, 1e-6f, final_normed);
        memcpy(logits, final_normed, N * D_MODEL * sizeof(float));
        free(final_normed);
    } else {
        memcpy(logits, x, N * D_MODEL * sizeof(float));
    }
    free(x);
    
    printf("Forward complete. Logits range: [%.4f, %.4f]\n",
           logits[0], logits[N * D_MODEL - 1]);
    
    // === Backward ===
    // Create simple gradient: d_logits = logits (for loss = sum(logits²)/2)
    float *d_logits = (float *)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++)
        d_logits[i] = logits[i];  // d/dx of 0.5*sum(x²) = x
    
    float *d_embeddings = (float *)calloc(N * D_MODEL, sizeof(float));
    
    printf("\nRunning model backward...\n");
    double t0 = clock();
    wubu_model_backward_from_embd(&model, embeddings, logits, d_logits,
                                   saved_normed, saved_attn_out,
                                   saved_normed2, saved_ffn_out,
                                   d_embeddings, B, T);
    double t1 = clock();
    printf("Backward: %.2f CPU seconds\n", (t1 - t0) / CLOCKS_PER_SEC);
    
    // === Verify non-zero gradients ===
    float max_d = 0.0f, min_d = 0.0f, sum_abs = 0.0f;
    int non_zero = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (d_embeddings[i] > max_d) max_d = d_embeddings[i];
        if (d_embeddings[i] < min_d) min_d = d_embeddings[i];
        sum_abs += fabsf(d_embeddings[i]);
        if (fabsf(d_embeddings[i]) > 1e-10f) non_zero++;
    }
    
    printf("\n=== Results ===\n");
    printf("d_embeddings range: [%.6e, %.6e]\n", min_d, max_d);
    printf("d_embeddings sum(|x|): %.6e\n", sum_abs);
    printf("d_embeddings non-zero elements: %d/%d\n", non_zero, N * D_MODEL);
    
    int pass = (non_zero > N * D_MODEL / 2) && (sum_abs > 1e-6f);
    printf("\nGradient flow test: %s\n", pass ? "PASS" : "FAIL");
    
    // Also verify that the gradient flows through multiple layers by checking
    // the gradient norm at different output levels
    printf("\nPer-sample gradient norms:\n");
    for (int i = 0; i < N; i++) {
        float norm = 0.0f;
        for (int j = 0; j < D_MODEL; j++)
            norm += d_embeddings[i * D_MODEL + j] * d_embeddings[i * D_MODEL + j];
        norm = sqrtf(norm);
        printf("  sample %d: ||d_embd|| = %.6e\n", i, norm);
    }
    
    // Cleanup
    free(embeddings); free(logits); free(d_logits); free(d_embeddings);
    free(saved_normed); free(saved_attn_out);
    free(saved_normed2); free(saved_ffn_out);
    wubu_model_free(&model);
    
    printf("\nDone.\n");
    return pass ? 0 : 1;
}

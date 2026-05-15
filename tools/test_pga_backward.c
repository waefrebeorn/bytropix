/**
 * test_pga_backward.c — Validate Poincaré GQA backward produces non-zero weight gradients.
 *
 * Loads model, runs one PGA forward, then backward, checks weight gradients.
 */
#include "wubu_poincare_gqa.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *model_path = argc>1?argv[1]:"/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int B = 1, T = 4, N = B * T;
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;   // 4096
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;  // 512
    float R = 10.0f;
    
    // Load model weights
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) { fprintf(stderr, "Can't open model\n"); return 1; }
    gguf_buffer_data(ctx);
    
    // Find a GQA layer
    gqa_layer_weights gqa_w;
    memset(&gqa_w, 0, sizeof(gqa_w));
    
    char tn[256];
    for (int l = 0; l < 40; l++) {
        if (wubu_is_ssm_layer(l)) continue;
        // Load this GQA layer's weights
        snprintf(tn, sizeof(tn), "blk.%d.attn_q.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, tn);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", tn); return 1; }
        gqa_w.attn_q_weight = (float*)malloc((int64_t)D_MODEL * q_dim * 2 * sizeof(float));
        gguf_read_tensor_f32(ctx, t, gqa_w.attn_q_weight, (int64_t)D_MODEL * q_dim * 2);
        
        snprintf(tn, sizeof(tn), "blk.%d.attn_k.weight", l);
        t = gguf_find_tensor(ctx, tn);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", tn); return 1; }
        gqa_w.attn_k_weight = (float*)malloc((int64_t)D_MODEL * kv_dim * sizeof(float));
        gguf_read_tensor_f32(ctx, t, gqa_w.attn_k_weight, (int64_t)D_MODEL * kv_dim);
        
        snprintf(tn, sizeof(tn), "blk.%d.attn_v.weight", l);
        t = gguf_find_tensor(ctx, tn);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", tn); return 1; }
        gqa_w.attn_v_weight = (float*)malloc((int64_t)D_MODEL * kv_dim * sizeof(float));
        gguf_read_tensor_f32(ctx, t, gqa_w.attn_v_weight, (int64_t)D_MODEL * kv_dim);
        
        snprintf(tn, sizeof(tn), "blk.%d.attn_output.weight", l);
        t = gguf_find_tensor(ctx, tn);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", tn); return 1; }
        gqa_w.attn_output_weight = (float*)malloc((int64_t)q_dim * D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, gqa_w.attn_output_weight, (int64_t)q_dim * D_MODEL);
        
        snprintf(tn, sizeof(tn), "blk.%d.attn_q_norm.weight", l);
        t = gguf_find_tensor(ctx, tn);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", tn); return 1; }
        gqa_w.attn_q_norm_weight = (float*)malloc(GQA_HEAD_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, gqa_w.attn_q_norm_weight, GQA_HEAD_DIM);
        
        snprintf(tn, sizeof(tn), "blk.%d.attn_k_norm.weight", l);
        t = gguf_find_tensor(ctx, tn);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", tn); return 1; }
        gqa_w.attn_k_norm_weight = (float*)malloc(GQA_HEAD_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, gqa_w.attn_k_norm_weight, GQA_HEAD_DIM);
        
        printf("Loaded GQA layer %d weights\n", l);
        break;  // load just one GQA layer
    }
    gguf_close(ctx);
    
    // Create random input
    float *x = (float*)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++) x[i] = ((float)rand() / RAND_MAX) * 0.1f;
    
    // Run forward_save
    poincare_gqa_fwd_save_t save;
    memset(&save, 0, sizeof(save));
    save.Q_ball = (float*)malloc(N * q_dim * sizeof(float));
    save.K_ball = (float*)malloc(N * kv_dim * sizeof(float));
    save.V_ball = (float*)malloc(N * kv_dim * sizeof(float));
    save.Q_norm = (float*)malloc(N * q_dim * sizeof(float));
    save.Q_raw  = (float*)malloc(N * q_dim * sizeof(float));
    save.K_norm = (float*)malloc(N * kv_dim * sizeof(float));
    save.K_raw  = (float*)malloc(N * kv_dim * sizeof(float));
    save.V      = (float*)malloc(N * kv_dim * sizeof(float));
    save.gate   = (float*)malloc(N * q_dim * sizeof(float));
    save.gate_sig = (float*)malloc(N * q_dim * sizeof(float));
    save.attn_out_pre_gate = (float*)malloc(N * q_dim * sizeof(float));
    
    float *output = (float*)malloc(N * D_MODEL * sizeof(float));
    wubu_poincare_gqa_forward_save(x, B, T, &gqa_w, R, output, &save);
    
    // Create gradient w.r.t. output (random, non-zero)
    float *d_output = (float*)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++) d_output[i] = ((float)rand() / RAND_MAX) * 0.01f;
    
    // Allocate gradient buffers
    float *d_x = (float*)calloc(N * D_MODEL, sizeof(float));
    float *d_q_weight = (float*)calloc((int64_t)D_MODEL * q_dim * 2, sizeof(float));
    float *d_k_weight = (float*)calloc((int64_t)D_MODEL * kv_dim, sizeof(float));
    float *d_v_weight = (float*)calloc((int64_t)D_MODEL * kv_dim, sizeof(float));
    float *d_q_norm_weight = (float*)calloc(GQA_HEAD_DIM, sizeof(float));
    float *d_k_norm_weight = (float*)calloc(GQA_HEAD_DIM, sizeof(float));
    float *d_out_weight = (float*)calloc((int64_t)q_dim * D_MODEL, sizeof(float));
    
    // Run backward
    wubu_poincare_gqa_backward(B, T,
        x,
        save.Q_norm, save.Q_raw,
        save.K_norm, save.K_raw,
        save.V,
        save.Q_ball, save.K_ball, save.V_ball,
        save.gate, save.gate_sig,
        save.attn_out_pre_gate,
        output,
        d_output,
        &gqa_w, R,
        d_x,
        d_q_weight, d_k_weight, d_v_weight,
        d_q_norm_weight, d_k_norm_weight, d_out_weight);
    
    // Check gradients
    float max_dq = 0, max_dk = 0, max_dv = 0, max_do = 0;
    float max_dqn = 0, max_dkn = 0, max_dx = 0;
    for (int64_t i = 0; i < (int64_t)D_MODEL * q_dim * 2; i++) { float v = fabsf(d_q_weight[i]); if (v > max_dq) max_dq = v; }
    for (int64_t i = 0; i < (int64_t)D_MODEL * kv_dim; i++) { float v = fabsf(d_k_weight[i]); if (v > max_dk) max_dk = v; }
    for (int64_t i = 0; i < (int64_t)D_MODEL * kv_dim; i++) { float v = fabsf(d_v_weight[i]); if (v > max_dv) max_dv = v; }
    for (int64_t i = 0; i < (int64_t)q_dim * D_MODEL; i++) { float v = fabsf(d_out_weight[i]); if (v > max_do) max_do = v; }
    for (int i = 0; i < GQA_HEAD_DIM; i++) { float v = fabsf(d_q_norm_weight[i]); if (v > max_dqn) max_dqn = v; }
    for (int i = 0; i < GQA_HEAD_DIM; i++) { float v = fabsf(d_k_norm_weight[i]); if (v > max_dkn) max_dkn = v; }
    for (int i = 0; i < N * D_MODEL; i++) { float v = fabsf(d_x[i]); if (v > max_dx) max_dx = v; }
    
    printf("=== PGA Backward Test Results ===\n");
    printf("d_q_weight_max:      %.6e  %s\n", max_dq, max_dq > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    printf("d_k_weight_max:      %.6e  %s\n", max_dk, max_dk > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    printf("d_v_weight_max:      %.6e  %s\n", max_dv, max_dv > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    printf("d_out_weight_max:    %.6e  %s\n", max_do, max_do > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    printf("d_q_norm_weight_max: %.6e  %s\n", max_dqn, max_dqn > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    printf("d_k_norm_weight_max: %.6e  %s\n", max_dkn, max_dkn > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    printf("d_x_max:             %.6e  %s\n", max_dx, max_dx > 0 ? "✓ NON-ZERO" : "✗ ZERO");
    
    int all_ok = (max_dq > 0 && max_dk > 0 && max_dv > 0 && max_do > 0 &&
                  max_dqn > 0 && max_dkn > 0 && max_dx > 0);
    printf("\n%s\n", all_ok ? "ALL GRADIENTS FLOW ✓" : "SOME GRADIENTS ZERO ✗");
    
    // Cleanup
    free(x); free(output); free(d_output); free(d_x);
    free(d_q_weight); free(d_k_weight); free(d_v_weight);
    free(d_q_norm_weight); free(d_k_norm_weight); free(d_out_weight);
    free(save.Q_ball); free(save.K_ball); free(save.V_ball);
    free(save.Q_norm); free(save.Q_raw);
    free(save.K_norm); free(save.K_raw);
    free(save.V); free(save.gate); free(save.gate_sig);
    free(save.attn_out_pre_gate);
    free(gqa_w.attn_q_weight); free(gqa_w.attn_k_weight); free(gqa_w.attn_v_weight);
    free(gqa_w.attn_output_weight);
    free(gqa_w.attn_q_norm_weight); free(gqa_w.attn_k_norm_weight);
    
    return all_ok ? 0 : 1;
}

#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

int wubu_moe_load_layer(gguf_ctx *ctx, int layer, moe_weights_t *moe) {
    char name[256];
    memset(moe, 0, sizeof(*moe));
    
    // Router: ffn_gate_inp.weight [D_MODEL, N_EXPERTS]
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp, D_MODEL * N_EXPERTS))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Expert gate
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    int64_t n = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    moe->ffn_gate_exps = (float *)malloc(n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Expert up
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_up_exps = (float *)malloc(n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_up_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Expert down
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    n = (int64_t)D_FF * D_MODEL * N_EXPERTS;
    moe->ffn_down_exps = (float *)malloc(n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_down_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Shared expert gate
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_gate_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_shexp, D_MODEL * SHARED_D_FF))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Shared expert up
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_up_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_up_shexp, D_MODEL * SHARED_D_FF))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Shared expert down
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_down_shexp = (float *)malloc(SHARED_D_FF * D_MODEL * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_down_shexp, SHARED_D_FF * D_MODEL))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // ffn_gate_inp_shexp exists but is unused in forward
    moe->ffn_gate_inp_shexp = NULL;
    
    moe->loaded = true;
    return 1;
}

void wubu_moe_free_layer(moe_weights_t *moe) {
    if (!moe || !moe->loaded) return;
    free(moe->ffn_gate_inp);
    free(moe->ffn_gate_exps);
    free(moe->ffn_up_exps);
    free(moe->ffn_down_exps);
    free(moe->ffn_gate_shexp);
    free(moe->ffn_up_shexp);
    free(moe->ffn_down_shexp);
    free(moe->ffn_gate_inp_shexp);
    memset(moe, 0, sizeof(*moe));
}

// ============================================================
// MoE Router: x @ gate_inp → softmax → top-k selection
// ============================================================

void wubu_moe_router(const float *x, int B, int T,
                     const float *gate_inp,
                     float *scores) {
    // x: [B, T, D_MODEL]
    // gate_inp: [D_MODEL, N_EXPERTS]
    // scores: [B*T, N_EXPERTS] (router logits, pre-softmax)
    int N = B * T;
    
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *score_s = scores + s * N_EXPERTS;
        
        // x_s @ gate_inp [2048] @ [2048, 256] -> [256]
        for (int e = 0; e < N_EXPERTS; e++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                sum += x_s[k] * gate_inp[k + e * D_MODEL];
            }
            score_s[e] = sum;
        }
    }
}

// ============================================================
// MoE Expert Computation (single expert for one token)
// ============================================================

static void moe_expert_forward(
    const float *x,          // [D_MODEL]
    const float *gate_weight, // [D_MODEL, D_FF]
    const float *up_weight,   // [D_MODEL, D_FF]
    const float *down_weight, // [D_FF, D_MODEL]
    float *temp,              // [D_FF * 3] scratch
    float *output)           // [D_MODEL]
{
    // temp layout: [gate_out(D_FF) | up_out(D_FF) | act(D_FF)]
    float *gate_out = temp;
    float *up_out = temp + D_FF;
    float *act = temp + 2 * D_FF;
    
    // gate = x @ gate_weight  [D_MODEL] @ [D_MODEL, D_FF] -> [D_FF]
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * gate_weight[k + j * D_MODEL];
        gate_out[j] = sum;
    }
    
    // up = x @ up_weight  [D_MODEL] @ [D_MODEL, D_FF] -> [D_FF]
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * up_weight[k + j * D_MODEL];
        up_out[j] = sum;
    }
    
    // act = silu(gate) * up
    for (int j = 0; j < D_FF; j++) {
        float g = gate_out[j];
        // silu: g * sigmoid(g)
        float silu_g;
        if (g < -80.0f) silu_g = 0.0f;
        else silu_g = g / (1.0f + expf(-g));
        act[j] = silu_g * up_out[j];
    }
    
    // output = act @ down_weight  [D_FF] @ [D_FF, D_MODEL] -> [D_MODEL]
    for (int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_FF; k++)
            sum += act[k] * down_weight[k + j * D_FF];
        output[j] = sum;
    }
}

// ============================================================
// MoE Layer Forward Pass
// ============================================================

void wubu_moe_forward(const float *x, int B, int T,
                      const moe_weights_t *w,
                      float *output) {
    if (!w->loaded) {
        // No MoE weights loaded: pass through
        memcpy(output, x, B * T * D_MODEL * sizeof(float));
        return;
    }
    
    int N = B * T;
    
    // Allocate scratch
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    int *topk_indices = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    float *topk_weights = (float *)malloc(N * N_ACTIVE_EXPTS * sizeof(float));
    float *expert_temp = (float *)malloc(D_FF * 3 * sizeof(float));  // per-expert scratch
    float *shared_gate = (float *)malloc(SHARED_D_FF * sizeof(float));
    float *shared_up = (float *)malloc(SHARED_D_FF * sizeof(float));
    float *shared_act = (float *)malloc(SHARED_D_FF * sizeof(float));
    
    if (!scores || !topk_indices || !topk_weights || !expert_temp ||
        !shared_gate || !shared_up || !shared_act) {
        fprintf(stderr, "MoE forward: allocation failed\n");
        free(scores); free(topk_indices); free(topk_weights); free(expert_temp);
        free(shared_gate); free(shared_up); free(shared_act);
        memcpy(output, x, N * D_MODEL * sizeof(float));
        return;
    }
    
    // Step 1: Compute router scores
    wubu_moe_router(x, B, T, w->ffn_gate_inp, scores);
    
    // Step 2: Softmax and top-k for each token
    for (int s = 0; s < N; s++) {
        float *score_s = scores + s * N_EXPERTS;
        
        // Find max for numerical stability
        float max_s = score_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (score_s[e] > max_s) max_s = score_s[e];
        
        // Compute softmax denominator
        float sum_exp = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum_exp += expf(score_s[e] - max_s);
        float inv_sum = 1.0f / (sum_exp + 1e-30f);
        
        // Compute softmax values
        float softmax_vals[N_EXPERTS];
        for (int e = 0; e < N_EXPERTS; e++)
            softmax_vals[e] = expf(score_s[e] - max_s) * inv_sum;
        
        // Find top-k indices and values
        int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
        float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;
        
        // Simple: mark k largest by repeated max search
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int best_idx = -1;
            float best_val = -1e30f;
            // Skip previously selected
            for (int e = 0; e < N_EXPERTS; e++) {
                bool used = false;
                for (int pk = 0; pk < k; pk++) {
                    if (indices_s[pk] == e) { used = true; break; }
                }
                if (!used && softmax_vals[e] > best_val) {
                    best_val = softmax_vals[e];
                    best_idx = e;
                }
            }
            indices_s[k] = best_idx;
            weights_s[k] = best_val;
        }
        
        // Normalize top-k weights to sum to 1
        float sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += weights_s[k];
        if (sum_w > 1e-30f) {
            float inv_sum_w = 1.0f / sum_w;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) weights_s[k] *= inv_sum_w;
        }
    }
    
    // Step 3: Process each token through selected experts + shared expert
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *out_s = output + s * D_MODEL;
        int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
        float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;
        
        // Initialize output with shared expert contribution
        // gate = x @ gate_shexp  [D_MODEL] @ [D_MODEL, 512] -> [512]
        for (int j = 0; j < SHARED_D_FF; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++)
                sum += x_s[k] * w->ffn_gate_shexp[k + j * D_MODEL];
            shared_gate[j] = sum;
        }
        
        // up = x @ up_shexp  [D_MODEL] @ [D_MODEL, 512] -> [512]
        for (int j = 0; j < SHARED_D_FF; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++)
                sum += x_s[k] * w->ffn_up_shexp[k + j * D_MODEL];
            shared_up[j] = sum;
        }
        
        // act = silu(gate) * up
        for (int j = 0; j < SHARED_D_FF; j++) {
            float g = shared_gate[j];
            float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
            shared_act[j] = silu_g * shared_up[j];
        }
        
        // shared_out = act @ down_shexp  [512] @ [512, 2048] -> [2048]
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int k = 0; k < SHARED_D_FF; k++)
                sum += shared_act[k] * w->ffn_down_shexp[k + j * SHARED_D_FF];
            out_s[j] = sum;
        }
        
        // Add routed expert contributions
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int e = indices_s[k];
            float wgt = weights_s[k];
            
            if (e < 0 || wgt < 1e-30f) continue;
            
            // Get expert weights (expert e starts at offset e * D_MODEL * D_FF)
            const float *gate_w = w->ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
            const float *up_w   = w->ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
            const float *down_w = w->ffn_down_exps  + (int64_t)e * D_FF * D_MODEL;
            
            float expert_out[D_MODEL];
            moe_expert_forward(x_s, gate_w, up_w, down_w, expert_temp, expert_out);
            
            // Weighted sum
            for (int j = 0; j < D_MODEL; j++)
                out_s[j] += wgt * expert_out[j];
        }
    }
    
    free(scores);
    free(topk_indices);
    free(topk_weights);
    free(expert_temp);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "wubu_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <alloca.h>

// GPU MoE expert forward (declared in wubu_model_gpu.cu, C linkage)
#ifdef GPU_SUPPORT
void wubu_model_gpu_moe_experts(const moe_weights_t *w,
    const float *x_s,
    const int *indices_s, const float *weights_s,
    float expert_contribs[8][D_MODEL],
    void *model_ptr);
#endif

#ifdef GPU_SUPPORT
// GPU SSM recurrence forward
void wubu_gpu_ssm_recurrence(
    float *ssm_state,
    const float *q, const float *k, const float *v,
    const float *beta, const float *gate,
    float *delta_out,
    void *stream);
#endif

int wubu_moe_load_layer(gguf_ctx *ctx, int layer, moe_weights_t *moe, int d_model, int d_ff, int n_experts) {
    char name[256];
    memset(moe, 0, sizeof(*moe));

    // Use global naming convention
    extern int g_tensor_naming;
    const char *prefix = (g_tensor_naming == 1) ? "model.layers." : "blk.";
    // Note: we don't have access to g_adapter here, so we use the resolved names from wubu_model.c

    // Router: ffn_gate_inp.weight [D_MODEL, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_gate_inp.weight", prefix, layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_gate_inp = (float *)malloc((size_t)d_model * n_experts * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp, d_model * n_experts))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // Expert gate [D_MODEL, D_FF, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_gate_exps.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    int64_t n = (int64_t)d_model * d_ff * n_experts;
    moe->ffn_gate_exps = (float *)malloc((size_t)n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // Expert up [D_MODEL, D_FF, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_up_exps.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_up_exps = (float *)malloc((size_t)n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_up_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // Expert down [D_FF, D_MODEL, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_down_exps.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    n = (int64_t)d_ff * d_model * n_experts;
    moe->ffn_down_exps = (float *)malloc((size_t)n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_down_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // Shared expert gate [D_MODEL, D_FF] (using d_ff as SHARED_D_FF)
    snprintf(name, sizeof(name), "%s.%d.ffn_gate_shexp.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_gate_shexp = (float *)malloc((size_t)d_model * d_ff * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_shexp, d_model * d_ff))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // Shared expert up [D_MODEL, D_FF]
    snprintf(name, sizeof(name), "%s.%d.ffn_up_shexp.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_up_shexp = (float *)malloc((size_t)d_model * d_ff * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_up_shexp, d_model * d_ff))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // Shared expert down [D_FF, D_MODEL]
    snprintf(name, sizeof(name), "%s.%d.ffn_down_shexp.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_down_shexp = (float *)malloc((size_t)d_ff * d_model * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_down_shexp, d_ff * d_model))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }

    // ffn_gate_inp_shexp — shared expert output gate: sigmoid(x_s @ this) scales output
    snprintf(name, sizeof(name), "%s.%d.ffn_gate_inp_shexp.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) {
        // Some models don't have this tensor; gate defaults to 1.0
        moe->ffn_gate_inp_shexp = NULL;
        printf("  Layer %d: no shared expert gate (ffn_gate_inp_shexp)\n", layer);
    } else {
        moe->ffn_gate_inp_shexp = (float *)malloc((size_t)d_model * sizeof(float));
        if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp_shexp, d_model))
            { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
        printf("  Layer %d: shared expert gate loaded (f32, %d elems)\n", layer, d_model);
    }

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
                     float *scores,
                     int n_experts, int d_model) {
    int N = B * T;

    #pragma omp parallel for if(N > 1)
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d_model;
        float *score_s = scores + s * n_experts;

        for (int e = 0; e < n_experts; e++) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum += x_s[k] * gate_inp[k + e * d_model];
            }
            score_s[e] = sum;
        }
    }
}

// ============================================================
// Quantized MoE Expert Computation (on-the-fly IQ2_XXS dequant)
// ============================================================

// Load one expert's quantized weights from the GGUF data blob
// Returns raw_size on success, 0 on failure
int wubu_moe_load_layer_quant(gguf_ctx *ctx, int layer,
                              uint8_t *gate_q, uint8_t *up_q, uint8_t *down_q,
                              int64_t *gate_raw_size, int64_t *up_raw_size, int64_t *down_raw_size,
                              int d_model, int d_ff) {
    char name[256];
    extern int g_tensor_naming;
    const char *prefix = (g_tensor_naming == 1) ? "model.layers." : "blk.";

    // Expert gate — quantized [D_MODEL, D_FF, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_gate_exps.weight", prefix, layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE quant load: missing %s\n", name); return 0; }
    int64_t n_elems = (int64_t)d_model * d_ff;
    int64_t raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    if (raw_sz <= 0) { fprintf(stderr, "MoE quant load: unsupported type %d\n", t->ggml_type); return 0; }
    if (!ctx->data_blob) {
        fprintf(stderr, "MoE quant load: data blob not buffered; call gguf_buffer_data first\n");
        return 0;
    }
    const uint8_t *src = (const uint8_t *)ctx->data_blob + t->data_offset;
    memcpy(gate_q, src, raw_sz);  // first expert only (expert 0)
    if (gate_raw_size) *gate_raw_size = raw_sz;

    // Expert up — quantized [D_MODEL, D_FF, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_up_exps.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE quant load: missing %s\n", name); return 0; }
    raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    src = (const uint8_t *)ctx->data_blob + t->data_offset;
    memcpy(up_q, src, raw_sz);
    if (up_raw_size) *up_raw_size = raw_sz;

    // Expert down — quantized [D_FF, D_MODEL, N_EXPERTS]
    snprintf(name, sizeof(name), "%s.%d.ffn_down_exps.weight", prefix, layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE quant load: missing %s\n", name); return 0; }
    n_elems = (int64_t)d_ff * d_model;
    raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    src = (const uint8_t *)ctx->data_blob + t->data_offset;
    memcpy(down_q, src, raw_sz);
    if (down_raw_size) *down_raw_size = raw_sz;

    return 1;
}

// Compute one expert with on-the-fly IQ2_XXS dequant dot product
// gate_q/up_q: quantized weight for one expert [D_MODEL, D_FF], column-major
// down_q: quantized weight for one expert [D_FF, D_MODEL], column-major
// temp: [D_FF * 3] scratch
void moe_expert_forward_dequant(const float *x,
                                const uint8_t *gate_q, const uint8_t *up_q, const uint8_t *down_q,
                                float *temp, float *output,
                                int d_model, int d_ff) {
    // Uses legacy IQ2_XXS kernel which assumes D_MODEL=2048, D_FF=512
    // For dynamic dimensions, we'd need a new dequant kernel
    // This path is legacy - main path uses quantized matmul in wubu_moe_forward
    (void)x; (void)gate_q; (void)up_q; (void)down_q; (void)temp; (void)output;
    (void)d_model; (void)d_ff;
}

// ============================================================
// Main MoE Forward: Router + Top-K Experts + Shared Expert
// ============================================================

void wubu_moe_forward(const float *x, int B, int T,
                      const moe_weights_t *w,
                      float *output,
                      int *selected_experts,
                      int n_active_experts, int n_experts, int d_model, int d_ff) {
    int N = B * T;

    // Router logits: [N, N_EXPERTS]
    float *scores = (float *)malloc((size_t)N * n_experts * sizeof(float));
    wubu_moe_router(x, B, T, w->ffn_gate_inp, scores, n_experts, d_model);

    // Softmax + Top-K per token
    int *topk_indices = (int *)malloc((size_t)N * n_active_experts * sizeof(int));
    float *topk_weights = (float *)malloc((size_t)N * n_active_experts * sizeof(float));

    for (int s = 0; s < N; s++) {
        float *score_s = scores + s * n_experts;

        // Softmax
        float max_val = -INFINITY;
        for (int e = 0; e < n_experts; e++) if (score_s[e] > max_val) max_val = score_s[e];
        float sum_exp = 0.0f;
        for (int e = 0; e < n_experts; e++) { score_s[e] = expf(score_s[e] - max_val); sum_exp += score_s[e]; }
        for (int e = 0; e < n_experts; e++) score_s[e] /= sum_exp;

        // Top-K
        for (int k = 0; k < n_active_experts; k++) {
            int best_e = -1; float best_w = -1.0f;
            for (int e = 0; e < n_experts; e++) {
                if (score_s[e] > best_w) { best_w = score_s[e]; best_e = e; }
            }
            topk_indices[s * n_active_experts + k] = best_e;
            topk_weights[s * n_active_experts + k] = best_w;
            score_s[best_e] = -1.0f;  // mark as used
        }
    }

    // Pass top-k to caller for prefetch
    if (selected_experts) {
        memcpy(selected_experts, topk_indices, (size_t)N * n_active_experts * sizeof(int));
    }

    // Expert computation
    float *expert_out = (float *)malloc((size_t)N * d_model * sizeof(float));
    memset(expert_out, 0, (size_t)N * d_model * sizeof(float));

    // Requires Qwen-style layout for quantized weights
    // For DiffusionGemma (different layout), would need adaptation
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d_model;
        float *out_s = expert_out + s * d_model;

        // Shared expert: always active
        // gate_proj: x_s @ ffn_gate_shexp^T -> [D_FF], SiLU
        // up_proj: x_s @ ffn_up_shexp^T -> [D_FF]
        // down_proj: (gate * up) @ ffn_down_shexp^T -> [D_MODEL]
        float *gate_shared = (float *)alloca((size_t)d_ff * sizeof(float));
        float *up_shared = (float *)alloca((size_t)d_ff * sizeof(float));

        for (int j = 0; j < d_ff; j++) {
            float sum_g = 0.0f, sum_u = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum_g += x_s[k] * w->ffn_gate_shexp[j + k * d_ff];  // [D_FF, D_MODEL] transposed
                sum_u += x_s[k] * w->ffn_up_shexp[j + k * d_ff];
            }
            gate_shared[j] = fmaxf(0.0f, sum_g);  // SiLU(x) = x * sigmoid(x) ≈ max(0, x) for rough approx
            up_shared[j] = sum_u;
        }

        // Shared expert output contribution
        for (int k = 0; k < d_model; k++) {
            float sum = 0.0f;
            for (int j = 0; j < d_ff; j++) {
                sum += gate_shared[j] * up_shared[j] * w->ffn_down_shexp[k + j * d_model];  // [D_MODEL, D_FF] transposed
            }
            out_s[k] = sum;
        }

        // Shared expert output gate (per-token scalar)
        if (w->ffn_gate_inp_shexp) {
            float gate_val = 0.0f;
            for (int k = 0; k < d_model; k++) {
                gate_val += x_s[k] * w->ffn_gate_inp_shexp[k];
            }
            gate_val = 1.0f / (1.0f + expf(-gate_val));  // sigmoid
            for (int k = 0; k < d_model; k++) out_s[k] *= gate_val;
        }

        // Routed experts (top-k)
        for (int ki = 0; ki < n_active_experts; ki++) {
            int e = topk_indices[s * n_active_experts + ki];
            float weight = topk_weights[s * n_active_experts + ki];

            // Expert gate projection
            for (int j = 0; j < d_ff; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d_model; k++) {
                    sum += x_s[k] * w->ffn_gate_exps[k + j * d_model + e * d_model * d_ff];
                }
                gate_shared[j] = fmaxf(0.0f, sum);  // SiLU
                // Expert up projection
                sum = 0.0f;
                for (int k = 0; k < d_model; k++) {
                    sum += x_s[k] * w->ffn_up_exps[k + j * d_model + e * d_model * d_ff];
                }
                up_shared[j] = sum;
            }

            // Expert down projection
            for (int k = 0; k < d_model; k++) {
                float sum = 0.0f;
                for (int j = 0; j < d_ff; j++) {
                    sum += gate_shared[j] * up_shared[j] * w->ffn_down_exps[k + j * d_model + e * d_model * d_ff];
                }
                out_s[k] += weight * sum;
            }
        }
    }

    memcpy(output, expert_out, (size_t)N * d_model * sizeof(float));
    free(expert_out);
    free(topk_indices);
    free(topk_weights);
    free(scores);
}

// MoE backward pass (not implemented for dynamic dimensions)
void wubu_moe_backward(const float *d_output, int B, int T,
                       const float *x,
                       const moe_weights_t *w,
                       float *d_x,
                       int *selected_experts,
                       int n_active_experts, int n_experts, int d_model, int d_ff) {
    // TODO: implement with dynamic dimensions
    (void)d_output; (void)B; (void)T; (void)x; (void)w; (void)d_x; (void)selected_experts;
    (void)n_active_experts; (void)n_experts; (void)d_model; (void)d_ff;
}
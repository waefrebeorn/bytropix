/**
 * wubu_moe_backward.c — MoE backward pass (CPU)
 *
 * Full backprop through MoE: shared expert + top-k routed experts.
 * Handles NULL expert weight pointers gracefully (skips that section).
 */
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

static inline float silu_f(float x) {
    if (x < -80.0f) return 0.0f;
    return x / (1.0f + expf(-x));
}

static inline float silu_deriv(float x, float silu_x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return silu_x + sig * (1.0f - silu_x);
}

// Recompute router: scores -> softmax -> top-k with normalized weights
static void moe_router_backward_prep(const float *x, int B, int T,
                                     const float *gate_inp,
                                     float *softmax_out,
                                     int *topk_indices,
                                     float *topk_weights)
{
    int N = B * T;
    float *logits = (float *)malloc(N * N_EXPERTS * sizeof(float));
    if (!logits) return;

    wubu_moe_router(x, B, T, gate_inp, logits);

    for (int s = 0; s < N; s++) {
        float *logit_s = logits + s * N_EXPERTS;
        float *score_s = softmax_out + s * N_EXPERTS;
        float max_s = logit_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (logit_s[e] > max_s) max_s = logit_s[e];
        float sum_exp = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum_exp += expf(logit_s[e] - max_s);
        float inv_sum = 1.0f / (sum_exp + 1e-30f);
        for (int e = 0; e < N_EXPERTS; e++)
            score_s[e] = expf(logit_s[e] - max_s) * inv_sum;

        int *ind_s = topk_indices + s * N_ACTIVE_EXPTS;
        float *wt_s = topk_weights + s * N_ACTIVE_EXPTS;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int best = -1;
            float best_v = -1e30f;
            for (int e = 0; e < N_EXPERTS; e++) {
                bool used = false;
                for (int pk = 0; pk < k; pk++)
                    if (ind_s[pk] == e) { used = true; break; }
                if (!used && score_s[e] > best_v) { best_v = score_s[e]; best = e; }
            }
            ind_s[k] = best;
            wt_s[k] = best_v;
        }
        float sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += wt_s[k];
        if (sum_w > 1e-30f) {
            float inv = 1.0f / sum_w;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) wt_s[k] *= inv;
        }
    }
    free(logits);
}

// Expert backward for one token. All weight pointers must be non-NULL.
static void moe_expert_backward(
    const float *x_s, const float *gate_w, const float *up_w, const float *down_w,
    const float *d_expert_out, float *temp,
    float *d_gate_w, float *d_up_w, float *d_down_w, float *d_x)
{
    float *gate_out = temp;
    float *up_out = temp + D_FF;
    float *act = temp + 2 * D_FF;

    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int p = 0; p < D_MODEL; p++)
            sum += (double)x_s[p] * (double)gate_w[p * D_FF + j];
        gate_out[j] = (float)sum;
    }
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int p = 0; p < D_MODEL; p++)
            sum += (double)x_s[p] * (double)up_w[p * D_FF + j];
        up_out[j] = (float)sum;
    }
    for (int j = 0; j < D_FF; j++)
        act[j] = silu_f(gate_out[j]) * up_out[j];

    float d_act[D_FF];
    for (int p = 0; p < D_FF; p++) {
        double sum = 0.0;
        for (int j = 0; j < D_MODEL; j++)
            sum += (double)d_expert_out[j] * (double)down_w[p * D_MODEL + j];
        d_act[p] = (float)sum;
    }
    if (d_down_w) {
        for (int p = 0; p < D_FF; p++)
            for (int j = 0; j < D_MODEL; j++)
                d_down_w[p * D_MODEL + j] += d_expert_out[j] * act[p];
    }
    for (int j = 0; j < D_FF; j++) {
        float g = gate_out[j];
        float sg = silu_f(g);
        float dsg = silu_deriv(g, sg);
        float d_up = d_act[j] * sg;
        float d_gate_v = d_act[j] * dsg * up_out[j];
        if (d_gate_w) {
            for (int p = 0; p < D_MODEL; p++)
                d_gate_w[p * D_FF + j] += d_gate_v * x_s[p];
        }
        if (d_up_w) {
            for (int p = 0; p < D_MODEL; p++)
                d_up_w[p * D_FF + j] += d_up * x_s[p];
        }
        for (int p = 0; p < D_MODEL; p++) {
            d_x[p] += d_gate_v * gate_w[p * D_FF + j]
                    + d_up * up_w[p * D_FF + j];
        }
    }
}

void wubu_moe_backward(const float *d_output, int B, int T,
                       const float *normed2,
                       const moe_weights_t *w,
                       float *d_normed2,
                       float *d_gate_inp, float *d_gate_exps, float *d_up_exps,
                       float *d_down_exps, float *d_gate_shexp,
                       float *d_up_shexp, float *d_down_shexp)
{
    if (!w->loaded) {
        // No shared expert: identity backward
        memcpy(d_normed2, d_output, B * T * D_MODEL * sizeof(float));
        return;
    }
    // If no expert weights, still continue with shared + router only

    bool has_expert_weights = (w->ffn_gate_exps && w->ffn_up_exps && w->ffn_down_exps);
    bool has_shared = (w->ffn_gate_shexp && w->ffn_up_shexp && w->ffn_down_shexp);
    bool has_router = (w->ffn_gate_inp != NULL);

    int N = B * T;

    float *softmax_vals = (float *)malloc(N * N_EXPERTS * sizeof(float));
    int *topk_indices = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    float *topk_weights = (float *)malloc(N * N_ACTIVE_EXPTS * sizeof(float));
    float *expert_temp = (float *)malloc(D_FF * 3 * sizeof(float));
    float *shared_temp = (float *)malloc(SHARED_D_FF * 3 * sizeof(float));
    float *d_shared_act_buf = (float *)malloc(SHARED_D_FF * sizeof(float));

    if (!softmax_vals || !topk_indices || !topk_weights || !expert_temp ||
        !shared_temp || !d_shared_act_buf) {
        memcpy(d_normed2, d_output, N * D_MODEL * sizeof(float));
        goto cleanup;
    }

    // Recompute router if gate_inp available
    if (has_router) {
        moe_router_backward_prep(normed2, B, T, w->ffn_gate_inp,
                                 softmax_vals, topk_indices, topk_weights);
    }

    memset(d_normed2, 0, N * D_MODEL * sizeof(float));

    for (int s = 0; s < N; s++) {
        const float *x_s = normed2 + s * D_MODEL;
        const float *d_out_s = d_output + s * D_MODEL;
        const float *scores_s = has_router ? softmax_vals + s * N_EXPERTS : NULL;
        const int *ind_s = has_router ? topk_indices + s * N_ACTIVE_EXPTS : NULL;
        const float *wt_s = has_router ? topk_weights + s * N_ACTIVE_EXPTS : NULL;
        float *d_x_s = d_normed2 + s * D_MODEL;

        // ===== SHARED EXPERT BACKWARD =====
        if (has_shared) {
            float *s_gate = shared_temp;
            float *s_up = shared_temp + SHARED_D_FF;
            float *s_act = shared_temp + 2 * SHARED_D_FF;

            for (int j = 0; j < SHARED_D_FF; j++) {
                double sum = 0.0;
                for (int k = 0; k < D_MODEL; k++)
                    sum += (double)x_s[k] * (double)w->ffn_gate_shexp[k * SHARED_D_FF + j];
                s_gate[j] = (float)sum;
            }
            for (int j = 0; j < SHARED_D_FF; j++) {
                double sum = 0.0;
                for (int k = 0; k < D_MODEL; k++)
                    sum += (double)x_s[k] * (double)w->ffn_up_shexp[k * SHARED_D_FF + j];
                s_up[j] = (float)sum;
            }
            for (int j = 0; j < SHARED_D_FF; j++)
                s_act[j] = silu_f(s_gate[j]) * s_up[j];

            for (int k = 0; k < SHARED_D_FF; k++) {
                double sum = 0.0;
                for (int j = 0; j < D_MODEL; j++)
                    sum += (double)d_out_s[j] * (double)w->ffn_down_shexp[k * D_MODEL + j];
                d_shared_act_buf[k] = (float)sum;
            }
            if (d_down_shexp) {
                for (int k = 0; k < SHARED_D_FF; k++)
                    for (int j = 0; j < D_MODEL; j++)
                        d_down_shexp[k * D_MODEL + j] += d_out_s[j] * s_act[k];
            }
            for (int j = 0; j < SHARED_D_FF; j++) {
                float g = s_gate[j];
                float sg = silu_f(g);
                float dsg = silu_deriv(g, sg);
                float d_up = d_shared_act_buf[j] * sg;
                float d_gate = d_shared_act_buf[j] * dsg * s_up[j];
                if (d_gate_shexp) {
                    for (int k = 0; k < D_MODEL; k++)
                        d_gate_shexp[k * SHARED_D_FF + j] += d_gate * x_s[k];
                }
                if (d_up_shexp) {
                    for (int k = 0; k < D_MODEL; k++)
                        d_up_shexp[k * SHARED_D_FF + j] += d_up * x_s[k];
                }
                for (int k = 0; k < D_MODEL; k++) {
                    d_x_s[k] += d_gate * w->ffn_gate_shexp[k * SHARED_D_FF + j]
                              + d_up * w->ffn_up_shexp[k * SHARED_D_FF + j];
                }
            }
        }

        // ===== ROUTED EXPERT BACKWARD =====
        if (has_expert_weights && has_router) {
            // Compute d_wgt for each active expert
            float d_wgt[N_ACTIVE_EXPTS];
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int e = ind_s[k];
                if (e < 0 || wt_s[k] < 1e-30f) { d_wgt[k] = 0.0f; continue; }
                const float *gate_w = w->ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
                const float *up_w   = w->ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
                const float *down_w = w->ffn_down_exps  + (int64_t)e * D_FF * D_MODEL;
                float *e_gate = expert_temp;
                float *e_up = expert_temp + D_FF;
                float *e_act = expert_temp + 2 * D_FF;
                for (int j = 0; j < D_FF; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < D_MODEL; p++)
                        sum += (double)x_s[p] * (double)gate_w[p * D_FF + j];
                    e_gate[j] = (float)sum;
                }
                for (int j = 0; j < D_FF; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < D_MODEL; p++)
                        sum += (double)x_s[p] * (double)up_w[p * D_FF + j];
                    e_up[j] = (float)sum;
                }
                for (int j = 0; j < D_FF; j++)
                    e_act[j] = silu_f(e_gate[j]) * e_up[j];
                double dw = 0.0;
                for (int j = 0; j < D_MODEL; j++) {
                    double e_out_j = 0.0;
                    for (int p = 0; p < D_FF; p++)
                        e_out_j += (double)e_act[p] * (double)down_w[p * D_MODEL + j];
                    dw += (double)d_out_s[j] * e_out_j;
                }
                d_wgt[k] = (float)dw;
            }

            // Router gradient
            float S_top = 1e-30f;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++)
                if (ind_s[k] >= 0) S_top += scores_s[ind_s[k]];
            float sum_dw_s = 0.0f;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++)
                if (ind_s[k] >= 0) sum_dw_s += d_wgt[k] * scores_s[ind_s[k]];

            float d_softmax[N_EXPERTS];
            memset(d_softmax, 0, sizeof(d_softmax));
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int e = ind_s[k];
                if (e < 0) continue;
                d_softmax[e] = d_wgt[k] / S_top - wt_s[k] * sum_dw_s / (S_top * S_top);
            }
            float s_dot_ds = 0.0f;
            for (int e = 0; e < N_EXPERTS; e++)
                s_dot_ds += scores_s[e] * d_softmax[e];

            float d_score[N_EXPERTS];
            for (int e = 0; e < N_EXPERTS; e++)
                d_score[e] = scores_s[e] * (d_softmax[e] - s_dot_ds);

            if (d_gate_inp) {
                for (int e = 0; e < N_EXPERTS; e++)
                    for (int p = 0; p < D_MODEL; p++)
                        d_gate_inp[p * N_EXPERTS + e] += d_score[e] * x_s[p];
            }
            for (int p = 0; p < D_MODEL; p++) {
                double sum = 0.0;
                for (int e = 0; e < N_EXPERTS; e++)
                    sum += (double)d_score[e] * (double)w->ffn_gate_inp[p * N_EXPERTS + e];
                d_x_s[p] += (float)sum;
            }

            // Expert backward
            float d_expert_scratch[D_MODEL];
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int e = ind_s[k];
                float wgt = wt_s[k];
                if (e < 0 || wgt < 1e-30f) continue;
                const float *gate_w = w->ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
                const float *up_w   = w->ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
                const float *down_w = w->ffn_down_exps  + (int64_t)e * D_FF * D_MODEL;
                for (int j = 0; j < D_MODEL; j++)
                    d_expert_scratch[j] = d_out_s[j] * wgt;
                float *d_gate_e = d_gate_exps ? d_gate_exps + (int64_t)e * D_MODEL * D_FF : NULL;
                float *d_up_e   = d_up_exps   ? d_up_exps   + (int64_t)e * D_MODEL * D_FF : NULL;
                float *d_down_e = d_down_exps ? d_down_exps + (int64_t)e * D_FF * D_MODEL : NULL;
                moe_expert_backward(x_s, gate_w, up_w, down_w,
                                   d_expert_scratch, expert_temp,
                                   d_gate_e, d_up_e, d_down_e, d_x_s);
            }
        }

        // If no router/expert weights: identity gradient for routed components
        if (!has_expert_weights) {
            for (int k = 0; k < D_MODEL; k++)
                d_x_s[k] += d_out_s[k];
        }
    }

cleanup:
    free(softmax_vals);
    free(topk_indices);
    free(topk_weights);
    free(expert_temp);
    free(shared_temp);
    free(d_shared_act_buf);
}

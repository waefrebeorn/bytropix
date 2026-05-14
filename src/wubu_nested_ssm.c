/**
 * Nested SSM Forward Implementation
 *
 * Product of K Poincaré balls with K curvatures.
 * Generalizes wubu_poincare_ssm_forward to K independent balls,
 * each with its own radius R_k, combined via learned gating weights.
 *
 * Architecture:
 *   Steps 1-8: IDENTICAL to Poincaré SSM (projections, conv, norm)
 *   Step 9:   K independent Poincaré recurrences, one per ball
 *   Step 9g:  Combine K ball outputs via gating weights
 *   Steps 10-11: IDENTICAL to Poincaré SSM (gate norm, output proj)
 */

#include "wubu_nested_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal: safe exponential with clamping to prevent float32 overflow
static inline float nested_safe_expf(float x) {
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) return expf(80.0f);
    return expf(x);
}

// Internal: compute softmax normalization for gating weights
static inline void nested_softmax_weights(const float *raw, int K, float *out) {
    float max_w = raw[0];
    for (int i = 1; i < K; i++) if (raw[i] > max_w) max_w = raw[i];
    double sum = 0.0;
    for (int i = 0; i < K; i++) {
        out[i] = expf(raw[i] - max_w);
        sum += out[i];
    }
    double inv_sum = 1.0 / (sum + 1e-30);
    for (int i = 0; i < K; i++) out[i] = (float)(out[i] * inv_sum);
}

// ============================================================
// Main nested SSM forward pass
// ============================================================
void wubu_nested_ssm_forward(const float *x, int B, int T,
                              const ssm_layer_weights *w,
                              wubu_nested_ssm_state_t *nested_state,
                              float *conv_state,
                              const wubu_nested_ssm_gating_t *gating,
                              float *output) {
    const int K = nested_state->K;
    const float *R = nested_state->R;
    const int N = B * T;
    const int C = KEY_DIM * 2 + VALUE_DIM;  // = 8192
    const int HEAD_STATE_SZ = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;

    if (K < 1 || K > NESTED_SSM_MAX_K) {
        fprintf(stderr, "Nested SSM: invalid K=%d (must be 1..%d)\n", K, NESTED_SSM_MAX_K);
        return;
    }

    // Validate input
    if (!nested_state->states) {
        fprintf(stderr, "Nested SSM: states not initialized\n");
        return;
    }

    // ========== Steps 1-8: IDENTICAL to Poincaré SSM ==========

    // Allocate intermediates
    float *qkv_all = (float *)malloc(N * C * sizeof(float));
    float *z_all = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *beta_raw = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_raw = (float *)malloc(N * DT_RANK * sizeof(float));
    float *conv_input = (float *)malloc(B * (T + CONV_KERNEL - 1) * C * sizeof(float));
    float *conv_output = (float *)malloc(N * C * sizeof(float));
    float *q_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *k_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *v_conv = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *q_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *k_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *delta_out = (float *)calloc(N * VALUE_DIM, sizeof(float));  // combined output
    float *z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));

    // Per-ball accumulators: each ball produces a delta_out
    float **ball_deltas = (float **)malloc(K * sizeof(float *));
    if (ball_deltas) {
        for (int k = 0; k < K; k++) {
            ball_deltas[k] = (float *)calloc(N * VALUE_DIM, sizeof(float));
        }
    }

    if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
        !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
        !delta_out || !z_silu || !beta_flat || !gate_flat || !ball_deltas) {
        fprintf(stderr, "Nested SSM forward: allocation failed\n");
        free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
        free(conv_input); free(conv_output);
        free(q_conv); free(k_conv); free(v_conv);
        free(q_norm); free(k_norm);
        free(delta_out); free(z_silu);
        free(beta_flat); free(gate_flat);
        if (ball_deltas) {
            for (int k = 0; k < K; k++) free(ball_deltas[k]);
            free(ball_deltas);
        }
        return;
    }

    // Step 1: Fused QKV projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *qkv_s = qkv_all + s * C;
        for (int j = 0; j < C; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_qkv_weight[i * C + j];
            qkv_s[j] = sum;
        }
    }

    // Step 2: z gate projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *z_s = z_all + s * VALUE_DIM;
        for (int j = 0; j < VALUE_DIM; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_gate_weight[i * VALUE_DIM + j];
            z_s[j] = sum;
        }
    }

    // Step 3: beta/alpha projections
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *beta_s = beta_raw + s * DT_RANK;
        float *alpha_s = alpha_raw + s * DT_RANK;
        for (int j = 0; j < DT_RANK; j++) {
            float sum_b = 0.0f, sum_a = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                sum_b += x_s[i] * w->ssm_beta_weight[i * DT_RANK + j];
                sum_a += x_s[i] * w->ssm_alpha_weight[i * DT_RANK + j];
            }
            beta_s[j] = sum_b;
            alpha_s[j] = sum_a;
        }
    }

    // Step 4: Compute beta and gate
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);

    float *alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));

    if (!alpha_biased || !alpha_softplus) {
        fprintf(stderr, "Nested SSM: alpha alloc failed\n");
        free(alpha_biased); free(alpha_softplus);
        goto cleanup_nested;
    }

    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            alpha_biased[s * DT_RANK + j] = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
        }
    }
    wubu_softplus(N * DT_RANK, alpha_biased, alpha_softplus);

    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            gate_flat[s * DT_RANK + j] = alpha_softplus[s * DT_RANK + j] * w->ssm_a[j];
        }
    }

    // Steps 5-8: Conv, split, L2 norm
    for (int b = 0; b < B; b++) {
        memcpy(conv_input + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        memcpy(conv_input + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C,
               T * C * sizeof(float));
    }
    wubu_conv1d(B, T, C, CONV_KERNEL, conv_input, w->ssm_conv1d_weight, conv_output);
    wubu_silu(N * C, conv_output, conv_output);

    for (int b = 0; b < B; b++) {
        float *ci = conv_input + (b * (T + CONV_KERNEL - 1) + T) * C;
        memcpy(conv_state + b * (CONV_KERNEL - 1) * C, ci,
               (CONV_KERNEL - 1) * C * sizeof(float));
    }

    for (int s = 0; s < N; s++) {
        const float *cv = conv_output + s * C;
        memcpy(q_conv + s * KEY_DIM, cv, KEY_DIM * sizeof(float));
        memcpy(k_conv + s * KEY_DIM, cv + KEY_DIM, KEY_DIM * sizeof(float));
        memcpy(v_conv + s * VALUE_DIM, cv + 2 * KEY_DIM, VALUE_DIM * sizeof(float));
    }

    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, 1e-12f, q_norm);
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, 1e-12f, k_norm);

    // ========== Step 9: K INDEPENDENT POINCARÉ RECURRENCES ==========

    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;  // 2

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;
            float *gate_s = gate_flat + s * DT_RANK;

            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / repeat_factor;
                float bg = beta_s[kh];
                float gg = nested_safe_expf(gate_s[kh]);

                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;

                // Run recurrence for each ball k with its curvature R_k
                for (int k = 0; k < K; k++) {
                    float R_k = R[k];
                    float *h = nested_state->states + (k * HEAD_STATE_SZ + vh * SSM_D_STATE * SSM_D_STATE);

                    // Temporary buffers (stack allocated for performance)
                    float temp_h[SSM_D_STATE];
                    float hk_tan[SSM_D_STATE];
                    float v_tan[SSM_D_STATE];
                    float diff_tan[SSM_D_STATE];
                    float update_ball[SSM_D_STATE];

                    // Step 9a: Decay state in Poincaré ball
                    // h_decayed = gg ⊗ h  (Möbius scalar multiplication)
                    for (int i = 0; i < SSM_D_STATE; i++) {
                        const float *h_row = h + i * SSM_D_STATE;
                        wubu_mobius_scalar_mul(gg, h_row, SSM_D_STATE, R_k, temp_h);
                        memcpy(h + i * SSM_D_STATE, temp_h, SSM_D_STATE * sizeof(float));
                    }

                    // Step 9b: Predict v from h using tangent-space inner product
                    memset(hk_tan, 0, SSM_D_STATE * sizeof(float));
                    for (int i = 0; i < SSM_D_STATE; i++) {
                        const float *h_row = h + i * SSM_D_STATE;
                        wubu_log_map(h_row, SSM_D_STATE, R_k, temp_h);
                        hk_tan[i] = wubu_dot(temp_h, k_vh, SSM_D_STATE);
                    }

                    // Step 9c: Map V to tangent space
                    wubu_log_map(v_vh, SSM_D_STATE, R_k, v_tan);

                    // Step 9d: Diff in tangent space
                    for (int i = 0; i < SSM_D_STATE; i++) {
                        diff_tan[i] = v_tan[i] - hk_tan[i];
                    }

                    // Step 9e: Outer product update in tangent space
                    memset(update_ball, 0, SSM_D_STATE * sizeof(float));
                    for (int i = 0; i < SSM_D_STATE; i++) {
                        float sum = 0.0f;
                        for (int j = 0; j < SSM_D_STATE; j++) {
                            sum += k_vh[i] * diff_tan[j] * bg;
                        }
                        update_ball[i] = sum;
                    }

                    // Step 9f: Map update to ball
                    float upd_ball[SSM_D_STATE];
                    wubu_exp_map(update_ball, SSM_D_STATE, R_k, upd_ball);

                    // Step 9g: h[t] = h[t-1] ⊕ update_ball (Möbius addition)
                    // Then clamp to ensure state stays within the ball (float32 safety)
                    for (int i = 0; i < SSM_D_STATE; i++) {
                        const float *h_row = h + i * SSM_D_STATE;
                        wubu_mobius_add(h_row, upd_ball, SSM_D_STATE, R_k, temp_h);
                        // Clamp: scale down if too close to boundary
                        float row_norm_sq = 0.0f;
                        for (int j = 0; j < SSM_D_STATE; j++)
                            row_norm_sq += temp_h[j] * temp_h[j];
                        float Rk_sq = R_k * R_k;
                        if (row_norm_sq >= Rk_sq * 0.9999f) {
                            float scale = sqrtf(Rk_sq * 0.9998f / row_norm_sq);
                            for (int j = 0; j < SSM_D_STATE; j++)
                                temp_h[j] *= scale;
                        }
                        memcpy(h + i * SSM_D_STATE, temp_h, SSM_D_STATE * sizeof(float));
                    }

                    // Step 9h: ball output = h @ q
                    float *ball_out = ball_deltas[k] + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                    memset(ball_out, 0, SSM_D_STATE * sizeof(float));
                    for (int i = 0; i < SSM_D_STATE; i++) {
                        const float *h_row = h + i * SSM_D_STATE;
                        for (int j = 0; j < SSM_D_STATE; j++) {
                            ball_out[i] += h_row[j] * q_vh[j];
                        }
                    }
                }  // end ball loop (k)
            }  // end head loop (vh)
        }  // end time loop (t)
    }  // end batch loop (b)

    // ========== Step 9g: COMBINE K BALL OUTPUTS VIA GATING ==========

    // Compute softmax-normalized gating weights
    float w_norm[NESTED_SSM_MAX_K];
    const float *raw_weights = NULL;
    float uniform_w[NESTED_SSM_MAX_K];

    if (gating) {
        raw_weights = gating->ball_weights;
    } else {
        // Uniform weighting: 1/K
        for (int k = 0; k < K; k++) uniform_w[k] = 0.0f;
        raw_weights = uniform_w;
    }

    // Softmax normalize
    nested_softmax_weights(raw_weights, K, w_norm);

    // Weighted sum: delta_out = sum_k w_norm[k] * ball_deltas[k]
    for (int i = 0; i < N * VALUE_DIM; i++) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += (double)w_norm[k] * (double)ball_deltas[k][i];
        }
        delta_out[i] = (float)sum;
    }

    // ========== Steps 10-11: IDENTICAL to Poincaré SSM ==========

    wubu_silu(N * VALUE_DIM, z_all, z_silu);

    // Step 10: Gated normalization
    for (int s = 0; s < N; s++) {
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            float *out_vh = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float *z_vh = z_silu + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float sum_sq = 0.0f;
            for (int i = 0; i < SSM_D_STATE; i++) sum_sq += out_vh[i] * out_vh[i];
            float rms = sqrtf(sum_sq / SSM_D_STATE + 1e-6f);
            float scale = 1.0f / rms;
            for (int i = 0; i < SSM_D_STATE; i++) {
                out_vh[i] = (out_vh[i] * scale * w->ssm_norm_weight[i]) * z_vh[i];
            }
        }
    }

    // Step 11: Output projection
    for (int s = 0; s < N; s++) {
        const float *inp = delta_out + s * VALUE_DIM;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int i = 0; i < VALUE_DIM; i++) {
                sum += inp[i] * w->ssm_out_weight[i * D_MODEL + j];
            }
            out[j] = sum;
        }
    }

cleanup_nested:
    free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
    free(conv_input); free(conv_output);
    free(q_conv); free(k_conv); free(v_conv);
    free(q_norm); free(k_norm);
    free(delta_out); free(z_silu);
    free(beta_flat); free(gate_flat);
    free(alpha_biased); free(alpha_softplus);
    if (ball_deltas) {
        for (int k = 0; k < K; k++) free(ball_deltas[k]);
        free(ball_deltas);
    }
}

// ============================================================
// Initialization: allocate and zero states
// ============================================================
int wubu_nested_ssm_init(wubu_nested_ssm_state_t *state, int K, const float *R) {
    if (!state || K < 1 || K > NESTED_SSM_MAX_K) return -1;

    state->K = K;
    for (int k = 0; k < K; k++) {
        state->R[k] = R[k];
    }

    const int HEAD_STATE_SZ = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    state->states = (float *)calloc(K * HEAD_STATE_SZ, sizeof(float));
    if (!state->states) {
        state->K = 0;
        return -1;
    }

    return 0;
}

// ============================================================
// Free nested SSM state memory
// ============================================================
void wubu_nested_ssm_free(wubu_nested_ssm_state_t *state) {
    if (state && state->states) {
        free(state->states);
        state->states = NULL;
    }
    if (state) state->K = 0;
}

// ============================================================
// Validate: check for NaN and within-radius constraints
// ============================================================
int wubu_nested_ssm_validate(const wubu_nested_ssm_state_t *state) {
    if (!state || !state->states || state->K < 1) return 1;

    const int HEAD_STATE_SZ = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;

    for (int k = 0; k < state->K; k++) {
        float R_k = state->R[k];
        float R_k_sq = R_k * R_k;
        const float *ball = state->states + k * HEAD_STATE_SZ;

        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            for (int i = 0; i < SSM_D_STATE; i++) {
                const float *h_row = ball + (vh * SSM_D_STATE + i) * SSM_D_STATE;

                // Check for NaN
                for (int j = 0; j < SSM_D_STATE; j++) {
                    if (isnan(h_row[j]) || isinf(h_row[j])) {
                        fprintf(stderr, "Nested SSM: ball %d, head %d, row %d has NaN/Inf\n",
                                k, vh, i);
                        return 1;
                    }
                }

                // Check within radius
                float norm_sq = 0.0f;
                for (int j = 0; j < SSM_D_STATE; j++) {
                    norm_sq += h_row[j] * h_row[j];
                }
                if (norm_sq >= R_k_sq * 0.99999f) {
                    fprintf(stderr, "Nested SSM: ball %d, head %d, row %d near boundary "
                            "(norm^2=%e, R^2=%e)\n",
                            k, vh, i, norm_sq, R_k_sq);
                    return 1;
                }
            }
        }
    }

    return 0;
}

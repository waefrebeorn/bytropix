#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include "thread_pool.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

// ============================================================
// Utility: Activation Functions
// ============================================================

void wubu_softplus(int n, const float *x, float *out) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v > 80.0f) out[i] = v;          // linear region
        else if (v < -80.0f) out[i] = 0.0f; // zero region
        else out[i] = logf(1.0f + expf(v));
    }
}

void wubu_silu(int n, const float *x, float *out) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < -80.0f) out[i] = 0.0f;
        else out[i] = v / (1.0f + expf(-v));
    }
}

void wubu_sigmoid(int n, const float *x, float *out) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < -80.0f) out[i] = 0.0f;
        else if (v > 80.0f) out[i] = 1.0f;
        else out[i] = 1.0f / (1.0f + expf(-v));
    }
}

// ============================================================
// Utility: Normalization
// ============================================================

void wubu_l2_norm(int B, int T, int n_heads, int d,
                  const float *x, float eps, float *out) {
    // x: [B, T, n_heads, d]
    // out: [B, T, n_heads, d]
    int seq_len = B * T;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *inp = x + (s * n_heads + h) * d;
            float *oup = out + (s * n_heads + h) * d;
            float sum_sq = 0.0f;
            for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
            float scale = 1.0f / sqrtf(sum_sq + eps);
            for (int i = 0; i < d; i++) oup[i] = inp[i] * scale;
        }
    }
}

void wubu_rms_norm(int B, int T, int d,
                   const float *x, const float *weight,
                   float eps, float *out) {
    // x: [B, T, d]
    // weight: [d]
    // out: [B, T, d]
    int seq_len = B * T;
    for (int s = 0; s < seq_len; s++) {
        const float *inp = x + s * d;
        float *oup = out + s * d;
        float sum_sq = 0.0f;
        for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
        float rms = sqrtf(sum_sq / d + eps);
        float scale = 1.0f / rms;
        for (int i = 0; i < d; i++) oup[i] = inp[i] * scale * weight[i];
    }
}

// ============================================================
// Utility: Matrix multiply (simple, non-optimized)
// ============================================================

static void matmul_nt(int M, int N, int K,
                      const float *A, const float *B,
                      float *C) {
    // C[M,N] = A[M,K] @ B[N,K]^T  (B is stored NxK but we use it as KxN internally)
    // Non-atomic: each thread writes to C[m, 0:N] for its assigned m range
    long long ops = (long long)M * N * K;
    #pragma omp parallel for if(ops > 500000)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];  // B[N,K] stored row-major
            }
            C[m * N + n] = sum;
        }
    }
}

// ============================================================
// Utility: 1D Convolution (depthwise, causal)
// ============================================================

void wubu_conv1d(int B, int T, int C, int k,
                 const float *input, const float *kernel,
                 float *output) {
    // input: [B, T+k-1, C] — already padded with k-1 zeros at start
    // kernel: [k, C]
    // output: [B, T, C]
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    int t_in = t + ki;  // input already padded with k-1 at start
                    sum += input[(b * (T + k - 1) + t_in) * C + c] *
                           kernel[ki * C + c];
                }
                output[(b * T + t) * C + c] = sum;
            }
        }
    }
}

// TGT (Toroidal Gradient Transformation) safe wrapping using π odometer
// Keeps values in [-π, π] range, preventing float32 overflow
#define TGT_PI      3.14159265358979323846f
#define TGT_BOUNDARY (2.0f * TGT_PI)

static inline float tgt_wrap(float x) {
    return fmodf(x + TGT_PI, TGT_BOUNDARY) - TGT_PI;
}

static inline float tgt_safe_expf(float x) {
    // Clamp to avoid float32 overflow: expf(89) ≈ 5e38 ≈ overflow
    if (x > 80.0f) x = 80.0f;
    if (x < -80.0f) return 0.0f;
    return expf(x);
}

// ============================================================
// SSM Layer Forward Pass (Euclidean)
// ============================================================

void wubu_ssm_forward(const float *x, int B, int T,
                      const ssm_layer_weights *w,
                      float *ssm_state,
                      float *conv_state,
                      float *output) {
    // x: [B, T, D_MODEL]
    // output: [B, T, D_MODEL]
    
    const int N = B * T;  // total tokens
    const int C = CONV_DIM;  // 8192
    
    // Allocate temporaries (in production: pre-allocate or use stack for small T)
    // For T up to 512, these fit on stack (~2-4MB)
    // For production: use heap
    float *qkv_all = (float *)malloc(N * (KEY_DIM * 2 + VALUE_DIM) * sizeof(float));
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
    float *delta_out = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
    
    if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
        !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
        !delta_out || !z_silu) {
        fprintf(stderr, "SSM forward: allocation failed\n");
        free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
        free(conv_input); free(conv_output);
        free(q_conv); free(k_conv); free(v_conv);
        free(q_norm); free(k_norm);
        free(delta_out); free(z_silu);
        return;
    }
    
    // Step 1: Fused QKV projection
    // x[B,T,2048] @ wqkv[2048,8192] -> qkv_all[B,T,8192]
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *qkv_s = qkv_all + s * C;
        for (int j = 0; j < C; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                sum += x_s[i] * w->attn_qkv_weight[i * C + j];
            }
            qkv_s[j] = sum;
        }
    }
    
    // Step 2: z gate projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *z_s = z_all + s * VALUE_DIM;
        for (int j = 0; j < VALUE_DIM; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                sum += x_s[i] * w->attn_gate_weight[i * VALUE_DIM + j];
            }
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
    
    // Step 4: Compute beta and gate (decay)
    // beta = sigmoid(beta_raw)
    // alpha_biased = alpha + ssm_dt_bias -> softplus -> * ssm_a
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    if (!beta_flat || !gate_flat) {
        fprintf(stderr, "SSM forward: beta/gate alloc failed\n");
        goto cleanup;
    }
    
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);
    
    float *alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));
    if (!alpha_biased || !alpha_softplus) goto cleanup;
    
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
    
    // Step 5: Convolution
    // Build conv_input: [B, T+CONV_KERNEL-1, C]
    // First CONV_KERNEL-1 elements are from conv_state, rest from qkv_all
    for (int b = 0; b < B; b++) {
        // Copy conv_state
        memcpy(conv_input + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        // Copy qkv_all
        memcpy(conv_input + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C,
               T * C * sizeof(float));
    }
    
    // Run convolution
    wubu_conv1d(B, T, C, CONV_KERNEL, conv_input, w->ssm_conv1d_weight, conv_output);
    
    // SiLU activation on conv output
    wubu_silu(N * C, conv_output, conv_output);
    
    // Update conv_state: last CONV_KERNEL-1 elements of conv_input
    for (int b = 0; b < B; b++) {
        float *ci = conv_input + (b * (T + CONV_KERNEL - 1) + T) * C;  // last k-1 elements
        memcpy(conv_state + b * (CONV_KERNEL - 1) * C, ci,
               (CONV_KERNEL - 1) * C * sizeof(float));
    }
    
    // Step 6: Split conv output into Q, K, V
    for (int s = 0; s < N; s++) {
        const float *cv = conv_output + s * C;
        memcpy(q_conv + s * KEY_DIM, cv, KEY_DIM * sizeof(float));
        memcpy(k_conv + s * KEY_DIM, cv + KEY_DIM, KEY_DIM * sizeof(float));
        memcpy(v_conv + s * VALUE_DIM, cv + 2 * KEY_DIM, VALUE_DIM * sizeof(float));
    }
    
    // Step 7: L2 normalize Q and K
    // q_conv: [N, SSM_K_HEADS, SSM_D_STATE]
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, 1e-12f, q_norm);
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, 1e-12f, k_norm);
    
    // Step 8: Repeat Q/K heads: 16 -> 32 (to match V heads)
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;  // 2
    
    // Step 9: Gated Delta Net recurrence per head
    // ssm_state: [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    // We process B batches and T timesteps, updating the state in-place
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;  // [DT_RANK=32]
            float *gate_s = gate_flat + s * DT_RANK;   // [DT_RANK=32]
            
            // For each V-head (32 heads):
            // For each V-head (32 heads) — fully parallel, each writes non-overlapping state
            #pragma omp parallel for if(T > 16 || B > 1)
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / repeat_factor;  // which K-head maps to this V-head
                
                float bg = beta_s[kh];
                float gg = tgt_safe_expf(gate_s[kh]);  // TGT: safe exp (clamped, no overflow)
                
                // Get Q, K, V for this head
                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                
                // Get state pointer for this V-head
                float *h = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                
                // Step 8a: State decay: h = h * exp(gate)
                for (int i = 0; i < SSM_D_STATE; i++) {
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        h[i * SSM_D_STATE + j] *= gg;
                    }
                }
                
                // Step 8b: Compute h @ k  -> [SSM_D_STATE]
                float hk[SSM_D_STATE];
                memset(hk, 0, sizeof(hk));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        hk[i] += h[i * SSM_D_STATE + j] * k_vh[j];
                    }
                }
                
                // Step 8c: diff = V - hk
                float diff[SSM_D_STATE];
                for (int i = 0; i < SSM_D_STATE; i++) {
                    diff[i] = v_vh[i] - hk[i];
                }
                
                // Step 8d: update = outer(k, diff) * beta + state
                for (int i = 0; i < SSM_D_STATE; i++) {
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        h[i * SSM_D_STATE + j] += k_vh[i] * diff[j] * bg;
                    }
                }
                
                // TGT: wrap state entries to prevent float32 overflow
                for (int i = 0; i < SSM_D_STATE; i++) {
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        h[i * SSM_D_STATE + j] = tgt_wrap(h[i * SSM_D_STATE + j]);
                    }
                }
                
                // Step 8e: output = h @ q  -> [SSM_D_STATE]
                // Store in delta_out
                float *out = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                memset(out, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        out[i] += h[i * SSM_D_STATE + j] * q_vh[j];
                    }
                }
            }
        }
    }
    
    // Step 10: Gated normalization
    // delta_out: [N, SSM_V_HEADS, SSM_D_STATE] = [N, 32, 128]
    // ssm_norm: [SSM_D_STATE] = [128]
    // z_silu: silu(z_all[VALUE_DIM])
    wubu_silu(N * VALUE_DIM, z_all, z_silu);
    
    // RMSNorm along SSM_D_STATE per head
    for (int s = 0; s < N; s++) {
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            float *out_vh = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float *z_vh = z_silu + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            
            // RMSNorm
            float sum_sq = 0.0f;
            for (int i = 0; i < SSM_D_STATE; i++) sum_sq += out_vh[i] * out_vh[i];
            float rms = sqrtf(sum_sq / SSM_D_STATE + 1e-6f);
            float scale = 1.0f / rms;
            
            // Apply norm weight and multiply by silu(z)
            for (int i = 0; i < SSM_D_STATE; i++) {
                out_vh[i] = (out_vh[i] * scale * w->ssm_norm_weight[i]) * z_vh[i];
            }
        }
    }
    
    // Step 11: Output projection
    // Python: final = gated_output @ weights['ssm_out.weight']
    // where weight is [VALUE_DIM, D_MODEL] = [4096, 2048]
    // result[j] = sum_i gated[i] * W[i][j] = sum_i gated[i] * data[j * VALUE_DIM + i] (see matmul pattern)
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
    
cleanup:
    free(qkv_all);
    free(z_all);
    free(beta_raw);
    free(alpha_raw);
    free(conv_input);
    free(conv_output);
    free(q_conv);
    free(k_conv);
    free(v_conv);
    free(q_norm);
    free(k_norm);
    free(delta_out);
    free(z_silu);
    free(beta_flat);
    free(gate_flat);
    free(alpha_biased);
    free(alpha_softplus);
}

// SSM forward with intermediate saving (for backward)
void wubu_ssm_forward_save(const float *x, int B, int T,
                           const ssm_layer_weights *w,
                           float *ssm_state,
                           float *conv_state,
                           float *output,
                           ssm_fwd_save_t *save)
{
    // Forward: run same computation, then save or free intermediates
    const int N = B * T;
    const int C = CONV_DIM;
    
    float *qkv_all = (float *)malloc(N * (KEY_DIM * 2 + VALUE_DIM) * sizeof(float));
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
    float *delta_out = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));
    
    if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
        !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
        !delta_out || !z_silu || !beta_flat || !gate_flat || !alpha_biased || !alpha_softplus) {
        fprintf(stderr, "SSM save: alloc failed\n");
        goto cleanup_save;
    }
    
    // === Steps 1-11: Same as wubu_ssm_forward ===
    // (Steps 1-10 are identical, just compute)
    
    // Step 1: QKV projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *qkv_s = qkv_all + s * C;
        for (int j = 0; j < C; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_qkv_weight[i * C + j];
            qkv_s[j] = (float)sum;
        }
    }
    
    // Step 2: z gate projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *z_s = z_all + s * VALUE_DIM;
        for (int j = 0; j < VALUE_DIM; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_gate_weight[i * VALUE_DIM + j];
            z_s[j] = (float)sum;
        }
    }
    
    // Step 3: beta/alpha projections
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *beta_s = beta_raw + s * DT_RANK;
        float *alpha_s = alpha_raw + s * DT_RANK;
        for (int j = 0; j < DT_RANK; j++) {
            double sum_b = 0.0, sum_a = 0.0;
            for (int i = 0; i < D_MODEL; i++) {
                sum_b += (double)x_s[i] * (double)w->ssm_beta_weight[i * DT_RANK + j];
                sum_a += (double)x_s[i] * (double)w->ssm_alpha_weight[i * DT_RANK + j];
            }
            beta_s[j] = (float)sum_b;
            alpha_s[j] = (float)sum_a;
        }
    }
    
    // Step 4: sigmoid(beta), softplus(alpha + bias) * ssm_a
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);
    for (int s = 0; s < N; s++)
        for (int j = 0; j < DT_RANK; j++)
            alpha_biased[s * DT_RANK + j] = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
    wubu_softplus(N * DT_RANK, alpha_biased, alpha_softplus);
    for (int s = 0; s < N; s++)
        for (int j = 0; j < DT_RANK; j++)
            gate_flat[s * DT_RANK + j] = alpha_softplus[s * DT_RANK + j] * w->ssm_a[j];
    
    // Step 5: Convolution + SiLU
    for (int b = 0; b < B; b++) {
        memcpy(conv_input + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        memcpy(conv_input + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C, T * C * sizeof(float));
    }
    wubu_conv1d(B, T, C, CONV_KERNEL, conv_input, w->ssm_conv1d_weight, conv_output);
    wubu_silu(N * C, conv_output, conv_output);
    
    // Update conv_state
    for (int b = 0; b < B; b++) {
        float *ci = conv_input + (b * (T + CONV_KERNEL - 1) + T) * C;
        memcpy(conv_state + b * (CONV_KERNEL - 1) * C, ci,
               (CONV_KERNEL - 1) * C * sizeof(float));
    }
    
    // Step 6: Split into Q, K, V
    for (int s = 0; s < N; s++) {
        const float *cv = conv_output + s * C;
        memcpy(q_conv + s * KEY_DIM, cv, KEY_DIM * sizeof(float));
        memcpy(k_conv + s * KEY_DIM, cv + KEY_DIM, KEY_DIM * sizeof(float));
        memcpy(v_conv + s * VALUE_DIM, cv + 2 * KEY_DIM, VALUE_DIM * sizeof(float));
    }
    
    // Step 7: L2 normalize Q and K
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, 1e-12f, q_norm);
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, 1e-12f, k_norm);
    
    // Step 8+9: Delta net recurrence with per-timestep state saving
    int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;
            float *gate_s = gate_flat + s * DT_RANK;
            
            // Save state before this timestep
            if (save && save->states_t) {
                memcpy(save->states_t + (b * (T+1) + t) * state_sz,
                       ssm_state, state_sz * sizeof(float));
            }
            
            #pragma omp parallel for
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / (SSM_V_HEADS / SSM_K_HEADS);
                float bg = beta_s[kh];
                float gg = tgt_safe_expf(gate_s[kh]);  // TGT: safe exp (clamped)
                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                float *h = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                
                for (int i = 0; i < SSM_D_STATE; i++)
                    for (int j = 0; j < SSM_D_STATE; j++)
                        h[i * SSM_D_STATE + j] *= gg;
                
                float hk[SSM_D_STATE];
                memset(hk, 0, sizeof(hk));
                for (int i = 0; i < SSM_D_STATE; i++)
                    for (int j = 0; j < SSM_D_STATE; j++)
                        hk[i] += h[i * SSM_D_STATE + j] * k_vh[j];
                
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float diff = v_vh[i] - hk[i];
                    for (int j = 0; j < SSM_D_STATE; j++)
                        h[i * SSM_D_STATE + j] += k_vh[i] * diff * bg;
                }
                
                float *out = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                memset(out, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++)
                    for (int j = 0; j < SSM_D_STATE; j++)
                        out[i] += h[i * SSM_D_STATE + j] * q_vh[j];
            }
        }
    }
    // Save final state
    if (save && save->states_t)
        memcpy(save->states_t + T * state_sz, ssm_state, state_sz * sizeof(float));
    
    // Step 10: Gated normalization
    wubu_silu(N * VALUE_DIM, z_all, z_silu);
    for (int s = 0; s < N; s++) {
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            float *out_vh = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float *z_vh = z_silu + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float sum_sq = 0.0f;
            for (int i = 0; i < SSM_D_STATE; i++) sum_sq += out_vh[i] * out_vh[i];
            float rms = sqrtf(sum_sq / SSM_D_STATE + 1e-6f);
            float scale = 1.0f / rms;
            for (int i = 0; i < SSM_D_STATE; i++)
                out_vh[i] = (out_vh[i] * scale * w->ssm_norm_weight[i]) * z_vh[i];
        }
    }
    
    // Step 11: Output projection
    for (int s = 0; s < N; s++) {
        const float *inp = delta_out + s * VALUE_DIM;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            double sum = 0.0;
            for (int i = 0; i < VALUE_DIM; i++)
                sum += (double)inp[i] * (double)w->ssm_out_weight[i * D_MODEL + j];
            out[j] = (float)sum;
        }
    }
    
    // === Save intermediates for backward ===
    if (save) {
        if (save->qkv_all) memcpy(save->qkv_all, qkv_all, N * C * sizeof(float));
        if (save->z_all) memcpy(save->z_all, z_all, N * VALUE_DIM * sizeof(float));
        if (save->beta_raw) memcpy(save->beta_raw, beta_raw, N * DT_RANK * sizeof(float));
        if (save->alpha_raw) memcpy(save->alpha_raw, alpha_raw, N * DT_RANK * sizeof(float));
        if (save->conv_post_silu) memcpy(save->conv_post_silu, conv_output, N * C * sizeof(float));
        if (save->q_conv) memcpy(save->q_conv, q_conv, N * KEY_DIM * sizeof(float));
        if (save->k_conv) memcpy(save->k_conv, k_conv, N * KEY_DIM * sizeof(float));
        if (save->v_conv) memcpy(save->v_conv, v_conv, N * VALUE_DIM * sizeof(float));
        if (save->q_norm) memcpy(save->q_norm, q_norm, N * KEY_DIM * sizeof(float));
        if (save->k_norm) memcpy(save->k_norm, k_norm, N * KEY_DIM * sizeof(float));
        if (save->delta_out) memcpy(save->delta_out, delta_out, N * VALUE_DIM * sizeof(float));
        if (save->z_silu) memcpy(save->z_silu, z_silu, N * VALUE_DIM * sizeof(float));
        if (save->beta_flat) memcpy(save->beta_flat, beta_flat, N * DT_RANK * sizeof(float));
        if (save->gate_flat) memcpy(save->gate_flat, gate_flat, N * DT_RANK * sizeof(float));
        if (save->conv_state_copy) memcpy(save->conv_state_copy, conv_state, B * (CONV_KERNEL-1) * C * sizeof(float));
    }
    
cleanup_save:
    free(qkv_all); free(z_all);
    free(beta_raw); free(alpha_raw);
    free(conv_input); free(conv_output);
    free(q_conv); free(k_conv); free(v_conv);
    free(q_norm); free(k_norm);
    free(delta_out); free(z_silu);
    free(beta_flat); free(gate_flat);
    free(alpha_biased); free(alpha_softplus);
}

// ============================================================
// Poincaré SSM Layer Forward Pass
// ============================================================

void wubu_poincare_ssm_forward(const float *x, int B, int T,
                               const ssm_layer_weights *w,
                               float *ssm_state,
                               float *conv_state,
                               float R,
                               float *output) {
    // Same as wubu_ssm_forward() except the recurrence step (Step 9)
    // uses Möbius operations instead of Euclidean linear algebra.
    //
    // Euclidean:  h[t] = h[t-1] * exp(gate) + (k_vh ⊗ (v - h[t-1] @ k_vh)) * beta
    // Poincaré:  h[t] = mobius_add(scalar_mul(exp(gate), h[t-1]),
    //                               exp_map(k_vh ⊗ (log_map(v, R) - log_map(h[t-1] @ k_vh)) * beta, R))
    //
    // Steps 1-8, 10-11 are IDENTICAL to Euclidean. Only Step 9 differs.
    // We reuse all preceding steps by copying the Euclidean forward code,
    // and modifying the recurrence section.
    
    // ... [Steps 1-8 are identical to Euclidean above] ...
    // For now, we call the Euclidean forward to get conv output and projections,
    // then rewrite the recurrence separately.
    
    // HACK: We copy most of the Euclidean code here. A cleaner design would
    // extract Steps 1-8 as shared helpers, but for Phase 2.2 this keeps things explicit.
    
    const int N = B * T;
    const int C = KEY_DIM * 2 + VALUE_DIM;  // = 8192
    
    // Allocate (same as Euclidean)
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
    float *delta_out = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
    
    if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
        !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
        !delta_out || !z_silu) {
        fprintf(stderr, "Poincaré SSM forward: allocation failed\n");
        free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
        free(conv_input); free(conv_output);
        free(q_conv); free(k_conv); free(v_conv);
        free(q_norm); free(k_norm);
        free(delta_out); free(z_silu);
        return;
    }
    
    // ========== Steps 1-8: IDENTICAL to Euclidean ==========
    
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
    
    // Step 4: Compute beta and gate (identical to Euclidean)
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    if (!beta_flat || !gate_flat) {
        fprintf(stderr, "Poincaré SSM forward: beta/gate alloc failed\n");
        goto cleanup_p;
    }
    
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);
    
    float *alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));
    if (!alpha_biased || !alpha_softplus) goto cleanup_p;
    
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
    
    // Steps 5-8: Conv, split, L2 norm (identical to Euclidean)
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
    
    // ========== Step 9: POINCARÉ RECURRENCE (differs from Euclidean) ==========
    
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;  // 2
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;
            float *gate_s = gate_flat + s * DT_RANK;
            
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / repeat_factor;
                
                float bg = beta_s[kh];
                float gg = tgt_safe_expf(gate_s[kh]);  // TGT: safe exp (clamped)  // scalar for Möbius multiplication
                
                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                
                float *h = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                
                // === Poincaré recurrence ===
                //
                // Instead of:
                //   h = h * gg  (Euclidean scalar multiply)
                //   hk = h @ k  (Euclidean matvec)
                //   diff = v - hk  (Euclidean subtract)
                //   h += k_vh ⊗ diff * bg  (Euclidean outer product update)
                //
                // We do:
                //   1. Decay state: h_decayed = scalar_mul(gg, h) in Poincaré ball
                //   2. Map to tangent: hk_tan = log_map(h, R) @ k  (inner in tangent space)
                //   3. Map V to tangent: v_tan = log_map(v, R)
                //   4. Diff in tangent: diff_tan = v_tan - hk_tan
                //   5. Update in tangent: update_tan = k ⊗ diff_tan * bg (outer product)
                //   6. Map back: update_ball = exp_map(update_tan, R)
                //   7. h[t] = mobius_add(h_decayed, update_ball)
                
                // Temporary buffers
                float temp_h[SSM_D_STATE];
                float temp_k[SSM_D_STATE];
                float temp_v[SSM_D_STATE];
                float hk_tan[SSM_D_STATE];
                float v_tan[SSM_D_STATE];
                float diff_tan[SSM_D_STATE];
                float update_ball[SSM_D_STATE];
                
                // Step 9a: Decay state in Poincaré ball
                // h_decayed = gg ⊗ h  (Möbius scalar multiplication)
                // h is a matrix [SSM_D_STATE x SSM_D_STATE]; we decay each row
                // For simplicity, treat h rows as independent ball vectors
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    wubu_mobius_scalar_mul(gg, h_row, SSM_D_STATE, R, temp_h);
                    memcpy(h + i * SSM_D_STATE, temp_h, SSM_D_STATE * sizeof(float));
                }
                
                // Step 9b: Predict v from h using tangent-space inner product
                // Map each row of h to tangent space, compute inner with k
                memset(hk_tan, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    wubu_log_map(h_row, SSM_D_STATE, R, temp_h);
                    // hk_tan[i] = log_map(h[i,:], R) · k  (tangent inner product)
                    hk_tan[i] = wubu_dot(temp_h, k_vh, SSM_D_STATE);
                }
                
                // Step 9c: Map V to tangent space
                wubu_log_map(v_vh, SSM_D_STATE, R, v_tan);
                
                // Step 9d: Diff in tangent space
                for (int i = 0; i < SSM_D_STATE; i++) {
                    diff_tan[i] = v_tan[i] - hk_tan[i];
                }
                
                // Step 9e: Outer product update in tangent space, then map to ball
                memset(update_ball, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        sum += k_vh[i] * diff_tan[j] * bg;
                    }
                    update_ball[i] = sum;
                }
                
                // Step 9f: Apply update — Möbius add instead of Euclidean add
                // update_ball is in tangent space, need to map it to ball
                float upd_ball[SSM_D_STATE];
                wubu_exp_map(update_ball, SSM_D_STATE, R, upd_ball);
                
                // Step 9g: h[t] = h[t-1] ⊕ update_ball  (Möbius addition)
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    wubu_mobius_add(h_row, upd_ball, SSM_D_STATE, R, temp_h);
                    memcpy(h + i * SSM_D_STATE, temp_h, SSM_D_STATE * sizeof(float));
                }
                
                // Step 9h: output = h @ q  (still Euclidean — Q is in tangent space)
                float *out = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                memset(out, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        out[i] += h_row[j] * q_vh[j];
                    }
                }
            }
        }
    }
    
    // ========== Steps 10-11: IDENTICAL to Euclidean ==========
    
    wubu_silu(N * VALUE_DIM, z_all, z_silu);
    
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
    
cleanup_p:
    free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
    free(conv_input); free(conv_output);
    free(q_conv); free(k_conv); free(v_conv);
    free(q_norm); free(k_norm);
    free(delta_out); free(z_silu);
    free(beta_flat); free(gate_flat);
    free(alpha_biased); free(alpha_softplus);
}

// ============================================================
// GQA Layer Forward Pass
// ============================================================

void wubu_gqa_forward(const float *x, int B, int T,
                      const gqa_layer_weights *w,
                      float *output) {
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;  // 4096
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;  // 512
    
    // Allocate
    // Q_full: [N, q_dim*2] = [N, 8192] — first q_dim=4096 Q, next 4096 gate
    float *Q_full = (float *)malloc(N * q_dim * 2 * sizeof(float));
    float *gate = (float *)malloc(N * q_dim * sizeof(float));
    float *K = (float *)malloc(N * kv_dim * sizeof(float));
    float *V = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));
    
    if (!Q_full || !gate || !K || !V || !Q_norm || !K_norm || !attn_out) {
        fprintf(stderr, "GQA forward: allocation failed\n");
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }
    
    // Step 1: Q + gate fused projection
    // wq: [D_MODEL, GQA_Q_HEADS*GQA_HEAD_DIM*2] = [2048, 8192]
    // Q_full: [N, q_dim*2] = [N, 8192] — first q_dim=4096 Q, next 4096 gate
    // (q_dim and kv_dim defined above)
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        int q_offset = s * q_dim * 2;  // FIXED: was s * q_dim
        // Q projection (first half of fused weight)
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + j];
            Q_full[q_offset + j] = (float)sum;
        }
        // Gate projection (second half of fused weight)
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + (j + q_dim)];
            Q_full[q_offset + q_dim + j] = (float)sum;
            gate[s * q_dim + j] = (float)sum;  // separate gate buffer for sigmoid
        }
        // NaN check on Q projection (first batch run only)
        static int gqa_nan_checked = 0;
        if (!gqa_nan_checked) {
            for (int j = 0; j < q_dim * 2; j++) {
                if (isnan(Q_full[s * q_dim * 2 + j]) || isinf(Q_full[s * q_dim * 2 + j])) {
                    printf("  GQA Q NaN at s=%d j=%d val=%e\n", s, j, (double)Q_full[s * q_dim * 2 + j]);
                    // Check ALL weight values at this column
                    int nan_w = 0, inf_w = 0;
                    for (int i = 0; i < D_MODEL; i++) {
                        float wv = w->attn_q_weight[i * (q_dim * 2) + j];
                        if (isnan(wv)) nan_w++;
                        if (isinf(wv)) inf_w++;
                    }
                    printf("    weight[*,%d]: NaN=%d Inf=%d\n", j, nan_w, inf_w);
                    // Check the dot product manually with printf per i
                    double test_sum = 0.0;
                    for (int i = 0; i < D_MODEL; i++) {
                        double prod = (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + j];
                        if (isnan(prod) || isinf(prod)) {
                            printf("    NaN/Inf at i=%d: x=%.2e w=%.2e prod=%.2e\n",
                                   i, (double)x_s[i], (double)w->attn_q_weight[i * (q_dim * 2) + j], prod);
                        }
                        test_sum += prod;
                    }
                    printf("    double sum=%e nan=%d inf=%d\n", test_sum, isnan(test_sum)?1:0, isinf(test_sum)?1:0);
                    printf("    cast to float: %e\n", (float)test_sum);
                    gqa_nan_checked = 1;
                    break;
                }
            }
        }
    }
    
    // Step 2: K projection [2048, 512]
    // (kv_dim defined above)
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_k_weight[i * kv_dim + j];
            K[s * kv_dim + j] = (float)sum;
        }
        for (int j = 0; j < kv_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_v_weight[i * kv_dim + j];
            V[s * kv_dim + j] = sum;
        }
    }
    
    // NaN guard: replace any NaN/Inf in Q_full, K, V with 0
    for (int i = 0; i < N * q_dim * 2; i++)
        if (isnan(Q_full[i]) || isinf(Q_full[i])) Q_full[i] = 0.0f;
    for (int i = 0; i < N * q_dim; i++)
        if (isnan(gate[i]) || isinf(gate[i])) gate[i] = 0.0f;
    for (int i = 0; i < N * kv_dim; i++) {
        if (isnan(K[i]) || isinf(K[i])) K[i] = 0.0f;
        if (isnan(V[i]) || isinf(V[i])) V[i] = 0.0f;
    }
    
    // Step 4: Q/K RMSNorm
    // Q_full has [N, q_dim*2] — split Q (first q_dim) from gate (second q_dim)
    // Q_norm should only normalize the Q portion, not the gate
    // Strategy: copy Q from Q_full to temp buffer, RMSNorm it, then copy back
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    if (!Q_only) { free(Q_full); free(gate); free(K); free(V); free(Q_norm); free(K_norm); free(attn_out); return; }
    for (int s = 0; s < N; s++)
        memcpy(Q_only + s * q_dim, Q_full + s * q_dim * 2, q_dim * sizeof(float));
    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM, Q_only, w->attn_q_norm_weight, 1e-6f, Q_norm);
    free(Q_only);
    
    // K RMSNorm — K has [N, kv_dim] = [N, KV_HEADS * HEAD_DIM]
    // Data layout: K[(b*T+t)*kv_dim + h*HEAD_DIM + i]
    // RMSNorm sees [B, T*KV_HEADS, HEAD_DIM] same layout
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM, K, w->attn_k_norm_weight, 1e-6f, K_norm);
    
    // Step 4: (RoPE skipped here - will be implemented separately)
    
    // Step 5: GQA Attention
    // 16 Q heads, 2 KV heads. Each KV head serves 8 Q heads.
    float scale = 1.0f / sqrtf(GQA_HEAD_DIM);
    
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);  // which KV head serves this Q
                
                const float *q_vec = Q_norm + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                float *out_vec = attn_out + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                
                memset(out_vec, 0, GQA_HEAD_DIM * sizeof(float));
                
                // Compute attention over all previous timesteps
                float attn_weights[4096];  // max T
                float max_score = -1e30f;
                
                // Q @ K^T * scale
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *k_vec = K_norm + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float score = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++)
                        score += q_vec[i] * k_vec[i];
                    score *= scale;
                    // TGT: wrap attention score to prevent overflow
                    score = tgt_wrap(score);
                    attn_weights[t_k] = score;
                    if (score > max_score) max_score = score;
                }
                
                // If all scores are valid (at least one exists), do softmax
                // max_score from -1e30f means at least one score exists
                
                // Softmax
                float sum_exp = 0.0f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                    sum_exp += attn_weights[t_k];
                }
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] /= sum_exp;
                }
                
                // Weighted sum of V
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *v_vec = V + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float a = attn_weights[t_k];
                    for (int i = 0; i < GQA_HEAD_DIM; i++) {
                        out_vec[i] += a * v_vec[i];
                    }
                }
                
                // NaN debug for h_q loop
                for (int _i = 0; _i < GQA_HEAD_DIM; _i++) {
                    if (isnan(out_vec[_i])) {
                        // Compute Q norm and KV norms
                        double q_norm_v = 0, kv_norms[4096];
                        for (int _k = 0; _k < GQA_HEAD_DIM; _k++) q_norm_v += (double)q_vec[_k] * q_vec[_k];
                        for (int _tk = 0; _tk <= t_q; _tk++) {
                            const float *k_v = K_norm + ((b * T + _tk) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                            kv_norms[_tk] = 0;
                            for (int _k = 0; _k < GQA_HEAD_DIM; _k++) kv_norms[_tk] += (double)k_v[_k] * k_v[_k];
                        }
                        printf("  GQA NaN at h%d t%d: q_norm=%.2f kv_norms=[", h_q, t_q, sqrt(q_norm_v));
                        for (int _k = 0; _k <= t_q && _k < 4; _k++) printf("%.2f ", sqrt(kv_norms[_k]));
                        printf("] q[0:3]=%.2e %.2e %.2e score=%e\n",
                               (double)q_vec[0], (double)q_vec[1], (double)q_vec[2], (double)attn_weights[0]);
                        break;
                    }
                }
            }
        }
    }
    
    // Step 6: Gate (sigmoid)
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    if (!gate_sig) { free(gate_sig); free(Q_full); free(gate); free(K); free(V); free(Q_norm); free(K_norm); free(attn_out); return; }
    wubu_sigmoid(N * q_dim, gate, gate_sig);
    
    for (int i = 0; i < N * q_dim; i++) {
        attn_out[i] *= gate_sig[i];
    }
    
    // Step 7: Output projection
    // Python: final = attn_out @ w['attn_output.weight']
    // weight shape: [q_dim, D_MODEL] = [4096, 2048]
    // result[j] = sum_i attn_out[i] * W[i][j] = sum_i attn_out[i] * data[j * q_dim + i]
    for (int s = 0; s < N; s++) {
        const float *inp = attn_out + s * q_dim;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int i = 0; i < q_dim; i++)
                sum += inp[i] * w->attn_output_weight[i * D_MODEL + j];
            out[j] = sum;
        }
    }
    
    free(Q_full);
    free(gate);
    free(K);
    free(V);
    free(Q_norm);
    free(K_norm);
    free(attn_out);
    free(gate_sig);
}

// GQA forward with intermediate saving (for backward)
void wubu_gqa_forward_save(const float *x, int B, int T,
                           const gqa_layer_weights *w,
                           float *output,
                           gqa_fwd_save_t *save)
{
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    
    float *Q_full = (float *)malloc(N * q_dim * 2 * sizeof(float));
    float *gate = (float *)malloc(N * q_dim * sizeof(float));
    float *K = (float *)malloc(N * kv_dim * sizeof(float));
    float *V = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));
    
    if (!Q_full || !gate || !K || !V || !Q_norm || !K_norm || !attn_out) {
        fprintf(stderr, "GQA save: alloc failed\n");
        goto cleanup_gqa_save;
    }
    
    // === Steps 1-7: Same as wubu_gqa_forward ===
    
    // Step 1: Q + gate fused projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        int q_offset = s * q_dim * 2;
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + j];
            Q_full[q_offset + j] = (float)sum;
        }
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + (j + q_dim)];
            Q_full[q_offset + q_dim + j] = (float)sum;
            gate[s * q_dim + j] = (float)sum;
        }
    }
    
    // Step 2: K projection (kv_dim defined above)
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_k_weight[i * kv_dim + j];
            K[s * kv_dim + j] = (float)sum;
        }
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_v_weight[i * kv_dim + j];
            V[s * kv_dim + j] = (float)sum;
        }
    }
    
    // Step 3: Q/K RMSNorm
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    for (int s = 0; s < N; s++)
        memcpy(Q_only + s * q_dim, Q_full + s * q_dim * 2, q_dim * sizeof(float));
    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM, Q_only, w->attn_q_norm_weight, 1e-6f, Q_norm);
    free(Q_only);
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM, K, w->attn_k_norm_weight, 1e-6f, K_norm);
    
    // Step 4: (RoPE skipped)
    
    // Step 5: GQA Attention
    float scale = 1.0f / sqrtf(GQA_HEAD_DIM);
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                const float *q_vec = Q_norm + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                float *out_vec = attn_out + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                memset(out_vec, 0, GQA_HEAD_DIM * sizeof(float));
                
                float attn_weights[4096];
                float max_score = -1e30f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *k_vec = K_norm + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float score = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++)
                        score += q_vec[i] * k_vec[i];
                    score *= scale;
                    // TGT: wrap attention score to prevent overflow
                    score = tgt_wrap(score);
                    attn_weights[t_k] = score;
                    if (score > max_score) max_score = score;
                }
                
                float sum_exp = 0.0f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                    sum_exp += attn_weights[t_k];
                }
                for (int t_k = 0; t_k <= t_q; t_k++)
                    attn_weights[t_k] /= sum_exp;
                
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *v_vec = V + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float a = attn_weights[t_k];
                    for (int i = 0; i < GQA_HEAD_DIM; i++)
                        out_vec[i] += a * v_vec[i];
                }
            }
        }
    }
    
    // Step 6: Gate (sigmoid)
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    wubu_sigmoid(N * q_dim, gate, gate_sig);
    
    // Save attn_out BEFORE gate multiply
    if (save && save->attn_out_pre_gate)
        memcpy(save->attn_out_pre_gate, attn_out, N * q_dim * sizeof(float));
    
    for (int i = 0; i < N * q_dim; i++)
        attn_out[i] *= gate_sig[i];
    
    // Step 7: Output projection
    for (int s = 0; s < N; s++) {
        const float *inp = attn_out + s * q_dim;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            double sum = 0.0;
            for (int i = 0; i < q_dim; i++)
                sum += (double)inp[i] * (double)w->attn_output_weight[i * D_MODEL + j];
            out[j] = (float)sum;
        }
    }
    
    // Save intermediates
    if (save) {
        if (save->Q_raw) memcpy(save->Q_raw, Q_full, N * q_dim * sizeof(float));  // first half
        if (save->Q_norm) memcpy(save->Q_norm, Q_norm, N * q_dim * sizeof(float));
        if (save->K_raw) memcpy(save->K_raw, K, N * kv_dim * sizeof(float));
        if (save->K_norm) memcpy(save->K_norm, K_norm, N * kv_dim * sizeof(float));
        if (save->V) memcpy(save->V, V, N * kv_dim * sizeof(float));
        if (save->gate) memcpy(save->gate, gate, N * q_dim * sizeof(float));
        if (save->gate_sig) memcpy(save->gate_sig, gate_sig, N * q_dim * sizeof(float));
    }
    
cleanup_gqa_save:
    free(Q_full); free(gate); free(K); free(V);
    free(Q_norm); free(K_norm); free(attn_out); free(gate_sig);
}

void wubu_poincare_gqa_forward(const float *x, int B, int T,
                               const gqa_layer_weights *w,
                               float R,
                               float *output) {
    // Same as wubu_gqa_forward() but uses Poincaré distance for attention scores.
    // Euclidean GQA: score = Q·K / sqrt(head_dim)
    // Poincaré GQA:   score = -d(Q,K)² / sqrt(head_dim)  where d is Poincaré distance
    //
    // Steps 1-4 (projections, RMSNorm), Step 6 (gate sigmoid), Step 7 (output proj)
    // are IDENTICAL to Euclidean GQA. Only Step 5 (attention) differs.
    
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    
    // Allocate
    float *Q_full = (float *)malloc(N * q_dim * sizeof(float));
    float *gate = (float *)malloc(N * q_dim * sizeof(float));
    float *K = (float *)malloc(N * kv_dim * sizeof(float));
    float *V = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));
    
    // ========== Steps 1-3: IDENTICAL to Euclidean GQA ==========
    
    // Step 1: Q + gate fused projection [2048, 8192]
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        for (int j = 0; j < q_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_q_weight[i * (q_dim * 2) + j];
            Q_full[s * q_dim + j] = sum;
        }
        for (int j = 0; j < q_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_q_weight[i * (q_dim * 2) + (j + q_dim)];
            gate[s * q_dim + j] = sum;
        }
    }
    
    // Step 2: K projection [2048, 512]
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        for (int j = 0; j < kv_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_k_weight[i * kv_dim + j];
            K[s * kv_dim + j] = sum;
        }
        for (int j = 0; j < kv_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++)
                sum += x_s[i] * w->attn_v_weight[i * kv_dim + j];
            V[s * kv_dim + j] = sum;
        }
    }
    
    // Step 3: RMSNorm on Q and K
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    memcpy(Q_only, Q_full, N * q_dim * sizeof(float));
    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM, Q_only, w->attn_q_norm_weight, 1e-6f, Q_norm);
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM, K, w->attn_k_norm_weight, 1e-6f, K_norm);
    free(Q_only);
    
    // ========== Step 5: Poincaré Distance Attention ==========
    // Instead of Q·K / sqrt(d), we compute:
    //   score[t_k] = -d(q, k_vh[t_k])² / sqrt(head_dim)
    // where d is Poincaré geodesic distance at curvature R.
    
    float scale = 1.0f / sqrtf(GQA_HEAD_DIM);
    
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            #pragma omp parallel for if(T > 32 || (B > 1 && T > 4))
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                
                const float *q_vec = Q_norm + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                float *out_vec = attn_out + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                
                memset(out_vec, 0, GQA_HEAD_DIM * sizeof(float));
                
                // Compute hyperbolic attention over all previous timesteps
                // Q,K are Euclidean after RMSNorm (norm ~64). Scale into ball:
                //   q_ball = q * (R / max_norm), k_ball = k * (R / max_norm)
                // Then compute distance in the ball.
                // Use tangent-space approximation to avoid boundary issues:
                //   d² ≈ ||log_map(q_ball,R) - log_map(k_ball,R)||²
                //
                // Simpler: just use log_map of the raw vectors (treating them as
                // tangent vectors, not ball points), since the norms are much
                // larger than R and full Mobius addition would be unstable.
                // This is valid as: dist² ≈ ||log_map(q,R) - log_map(k,R)||²
                // which is approximately the Euclidean distance weighted by
                // the conformal factor.
                
                float attn_weights[4096];
                float max_score = -1e30f;
                
                float *log_q = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
                float *log_k = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
                
                // Scale factor to map Q/K into the ball
                float scale_in = 0.99f * R / 64.0f;  // norm ~64, scale to 0.99R
                
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *k_vec = K_norm + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    
                    // Compute log_map of vectors scaled into ball
                    // log_map(x) = R * artanh(||x||/R) * x/||x||
                    // But x_ball = x * scale_in, so ||x_ball|| = ||x|| * scale_in
                    
                    // For q:
                    float nq = 0.0f; for (int i = 0; i < GQA_HEAD_DIM; i++) nq += q_vec[i]*q_vec[i];
                    nq = sqrtf(nq) * scale_in;  // scaled norm
                    // For k:
                    float nk = 0.0f; for (int i = 0; i < GQA_HEAD_DIM; i++) nk += k_vec[i]*k_vec[i];
                    nk = sqrtf(nk) * scale_in;  // scaled norm
                    
                    // log_map with clamp for artanh
                    float rq = nq / R; if (rq >= 0.9999f) rq = 0.9999f;
                    float rk = nk / R; if (rk >= 0.9999f) rk = 0.9999f;
                    float artanh_q = nq > 1e-10f ? 0.5f * logf((1.0f + rq) / (1.0f - rq)) : 0.0f;
                    float artanh_k = nk > 1e-10f ? 0.5f * logf((1.0f + rk) / (1.0f - rk)) : 0.0f;
                    
                    float scale_q = nq > 1e-10f ? R * artanh_q / nq : 0.0f;
                    float scale_k = nk > 1e-10f ? R * artanh_k / nk : 0.0f;
                    
                    for (int i = 0; i < GQA_HEAD_DIM; i++) {
                        log_q[i] = q_vec[i] * scale_q * scale_in;
                        log_k[i] = k_vec[i] * scale_k * scale_in;
                    }
                    
                    // Euclidean distance in tangent space
                    float dist_sq = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++) {
                        float d = log_q[i] - log_k[i];
                        dist_sq += d * d;
                    }
                    
                    // score = -dist² / sqrt(head_dim)
                    float score = -(dist_sq) * scale;
                    attn_weights[t_k] = score;
                    if (score > max_score) max_score = score;
                }
                
                free(log_q);
                free(log_k);
                
                // Softmax
                float sum_exp = 0.0f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                    sum_exp += attn_weights[t_k];
                }
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] /= sum_exp;
                }
                
                // Weighted sum of V (identical to Euclidean)
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *v_vec = V + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float a = attn_weights[t_k];
                    for (int i = 0; i < GQA_HEAD_DIM; i++) {
                        out_vec[i] += a * v_vec[i];
                    }
                }
            }
        }
    }
    
    // ========== Steps 6-7: IDENTICAL to Euclidean GQA ==========
    
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    wubu_sigmoid(N * q_dim, gate, gate_sig);
    for (int i = 0; i < N * q_dim; i++) {
        attn_out[i] *= gate_sig[i];
    }
    
    for (int s = 0; s < N; s++) {
        const float *inp = attn_out + s * q_dim;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int i = 0; i < q_dim; i++)
                sum += inp[i] * w->attn_output_weight[i * D_MODEL + j];
            out[j] = sum;
        }
    }
    
    free(Q_full); free(gate); free(K); free(V);
    free(Q_norm); free(K_norm); free(attn_out);
    free(gate_sig);
}

// ============================================================
// Layer Type Helpers
// ============================================================

int wubu_is_ssm_layer(int layer_idx) {
    // Every 4th layer (index 3, 7, 11, ...) is GQA (full attention)
    // Rest are SSM layers
    return (layer_idx + 1) % 4 != 0;
}

// ============================================================
// Backward Pass — SSM Output Projection (Step 11)
// ============================================================
// Forward: output[s,j] = sum_i delta_out[s,i] * W[i,j]
//          where W = [VALUE_DIM, D_MODEL]
// Backward:
//   d_delta_out += d_output @ W^T   [N,V] = [N,D] @ [D,V]^T
//   dW += delta_out^T @ d_output   [V,D] = [V,N] @ [N,D]
void wubu_ssm_backward_output_proj(
    const float *delta_out,      // [N, VALUE_DIM] forward input (for dW)
    const float *d_output,       // [N, D_MODEL] gradient from upstream
    const float *ssm_out_weight, // [VALUE_DIM, D_MODEL] forward weight
    float *d_delta_out,          // [N, VALUE_DIM] gradient to propagate
    float *d_ssm_out_weight,     // [VALUE_DIM, D_MODEL] weight grad accum (or NULL)
    int N)
{
    // d_delta_out = d_output @ W^T
    for (int s = 0; s < N; s++) {
        for (int i = 0; i < VALUE_DIM; i++) {
            double sum = 0.0;
            for (int j = 0; j < D_MODEL; j++)
                sum += (double)d_output[s * D_MODEL + j] * (double)ssm_out_weight[i * D_MODEL + j];
            d_delta_out[s * VALUE_DIM + i] += (float)sum;
        }
    }
    // dW = delta_out^T @ d_output  (only if weight grad is requested)
    if (d_ssm_out_weight) {
        for (int i = 0; i < VALUE_DIM; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                double sum = 0.0;
                for (int s = 0; s < N; s++)
                    sum += (double)delta_out[s * VALUE_DIM + i] * (double)d_output[s * D_MODEL + j];
                d_ssm_out_weight[i * D_MODEL + j] += (float)sum;
            }
        }
    }
}

// ============================================================
// Backward Pass — Gated Normalization (Step 10)
// ============================================================
// Forward: out[i] = x[i] * scale * w[i] * z[i]
//   where scale = 1/sqrt(mean(x²)+eps), w = norm_weight, z = silu(z_raw)
// Backward: see derivation below
void wubu_ssm_backward_gated_norm(
    const float *x,          // [N, VALUE_DIM] pre-norm delta_out
    const float *z_silu,     // [N, VALUE_DIM] silu(z_raw) from forward
    const float *d_out,      // [N, VALUE_DIM] upstream grad (dL/dout)
    const float *norm_w,     // [SSM_D_STATE] norm weight (broadcast over V_HEADS)
    float *d_x,              // [N, VALUE_DIM] grad to propagate
    float *d_z_silu,         // [N, VALUE_DIM] grad for z_silu
    int B, int T)
{
    const int N = B * T;
    const int d = SSM_D_STATE;  // 128
    const int n_heads = SSM_V_HEADS;  // 32
    
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *x_h = x + (s * n_heads + h) * d;
            const float *z_h = z_silu + (s * n_heads + h) * d;
            const float *do_h = d_out + (s * n_heads + h) * d;
            float *dx_h = d_x + (s * n_heads + h) * d;
            float *dz_h = d_z_silu + (s * n_heads + h) * d;
            
            // Compute mean(x²) and rms
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
            float rms = sqrtf((float)(sum_sq / d) + 1e-6f);
            float s = 1.0f / rms;  // scale
            float s3 = s * s * s;  // ds/dm = -s³/2
            
            // Compute inner = sum_j d_out[j] * x[j] * w[j] * z[j]
            double inner = 0.0;
            for (int j = 0; j < d; j++)
                inner += (double)do_h[j] * (double)x_h[j] * (double)norm_w[j] * (double)z_h[j];
            
            // dL/dx[i] = do[i] * w[i] * s * z[i] - (s³/d) * x[i] * inner
            for (int i = 0; i < d; i++) {
                float grad = do_h[i] * norm_w[i] * s * z_h[i];
                grad -= (s3 / d) * x_h[i] * (float)inner;
                dx_h[i] += grad;
            }
            
            // dL/dz_silu[i] = do[i] * x[i] * w[i] * s
            for (int i = 0; i < d; i++) {
                dz_h[i] += do_h[i] * x_h[i] * norm_w[i] * s;
            }
        }
    }
}

// ============================================================
// Backward Pass — Gated Norm Weight Gradient
// ============================================================
// dL/dw[i] = sum_{s,h} dL/dy[s,h,i] * x[s,h,i] * s[s,h] * z[s,h,i]
void wubu_ssm_backward_gated_norm_weight(
    const float *x, const float *z_silu,
    const float *d_out,
    float *d_norm_weight, int B, int T)
{
    if (!d_norm_weight) return;
    const int N = B * T;
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_vh; h++) {
            const float *x_h = x + (s * n_vh + h) * d;
            const float *z_h = z_silu + (s * n_vh + h) * d;
            const float *do_h = d_out + (s * n_vh + h) * d;
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
            float s_val = 1.0f / sqrtf((float)(sum_sq / d) + 1e-6f);
            for (int i = 0; i < d; i++)
                d_norm_weight[i] += do_h[i] * x_h[i] * s_val * z_h[i];
        }
    }
}

// ============================================================
// Backward Pass — SiLU activation
// ============================================================
// silu(x) = x * sigmoid(x)
// silu'(x) = silu(x) + sigmoid(x) * (1 - silu(x))
void wubu_silu_backward(int n, const float *x, const float *y,
                        const float *dy, float *dx) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-v));
        float silu = y[i];
        float silu_grad = silu + sig * (1.0f - silu);
        dx[i] += dy[i] * silu_grad;
    }
}

// ============================================================
// Backward Pass — L2 Normalization
// ============================================================
// Forward: out[i] = x[i] / sqrt(sum(x²) + eps)
// Backward: see derivation in header comment
void wubu_l2_norm_backward(int B, int T, int n_heads, int d,
                           const float *x, float eps,
                           const float *d_out, float *d_x) {
    const int N = B * T;
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *inp = x + (s * n_heads + h) * d;
            const float *do_h = d_out + (s * n_heads + h) * d;
            float *dx = d_x + (s * n_heads + h) * d;
            
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)inp[i] * (double)inp[i];
            float norm = sqrtf(sum_sq + eps);
            float n3 = norm * norm * norm;
            
            // d_i = (do_i / norm) - (x_i / n³) * sum_j (do_j * x_j)
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += (double)do_h[j] * (double)inp[j];
            
            for (int i = 0; i < d; i++) {
                dx[i] += (float)((double)do_h[i] / norm - (double)inp[i] * dot / n3);
            }
        }
    }
}

// ============================================================
// Backward Pass — SSM Delta Net Recurrence (Step 9)
// ============================================================
// Forward (per head):
//   h_new = h_old * gg + k * (v - (h_old * gg) @ k) * bg
//   output = h_new @ q
// where gg = exp(gate), bg = beta
//
// Backward processes timesteps in reverse (BPTT).
// Requires per-timestep saved states from forward pass.
void wubu_ssm_backward_recurrence(
    int B, int T,
    const float *saved_states,      // [T+1, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
                                    // saved_states[t] = state after timestep t
                                    // saved_states[0] = initial state (before any timestep)
    const float *q_norm,            // [N, SSM_K_HEADS, SSM_D_STATE]
    const float *k_norm,            // [N, SSM_K_HEADS, SSM_D_STATE]
    const float *v_conv,            // [N, SSM_V_HEADS, SSM_D_STATE]
    const float *beta_flat,         // [N, DT_RANK]
    const float *gate_flat,         // [N, DT_RANK]
    const float *d_output,          // [N, VALUE_DIM] grad from gated norm (dL/d(delta_out))
    float *d_q_norm,                // [N, SSM_K_HEADS, SSM_D_STATE] output grad
    float *d_k_norm,                // [N, SSM_K_HEADS, SSM_D_STATE] output grad
    float *d_v_conv,                // [N, SSM_V_HEADS, SSM_D_STATE] output grad
    float *d_beta_flat,             // [N, DT_RANK] output grad
    float *d_gate_flat,             // [N, DT_RANK] output grad
    float *d_state_init)            // [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] BPTT to initial state
{
    const int d = SSM_D_STATE;     // 128
    const int n_vh = SSM_V_HEADS;  // 32
    const int n_kh = SSM_K_HEADS;  // 16
    const int repeat = n_vh / n_kh;
    const int N = B * T;
    const int state_sz = n_vh * d * d;
    
    // BPTT: process timesteps in reverse
    // d_h_next accumulates the gradient w.r.t. h_new from future timesteps
    float *d_h_next = (float *)calloc(state_sz, sizeof(float));
    if (!d_h_next) {
        fprintf(stderr, "backward_recurrence: d_h_next alloc failed\n");
        return;
    }
    
    for (int t = T - 1; t >= 0; t--) {
        for (int b = 0; b < B; b++) {
            int s = b * T + t;  // flat token index
            
            // Get saved states: h_old = state before this timestep, h_new = state after
            const float *h_old = saved_states + t * state_sz;
            const float *h_new = saved_states + (t + 1) * state_sz;
            
            // d_h_new = d_output[t] @ q^T + d_h_next (BPTT from t+1)
            // d_output[t]: [VALUE_DIM] = [n_vh * d]
            // For each vh: output[i] = sum_j h_new[i,j] * q[j]
            // dL/dh_new[i,j] += d_output[t][i] * q[j]
            
            float *d_h_new = (float *)calloc(state_sz, sizeof(float));
            if (!d_h_new) { free(d_h_next); return; }
            
            // dL/dh_new += d_output @ q^T
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *q_vh = q_norm + (s * n_kh + kh) * d;
                const float *do_vh = d_output + (s * n_vh + vh) * d;
                
                for (int i = 0; i < d; i++) {
                    float *dh = d_h_new + vh * d * d + i * d;
                    float do_i = do_vh[i];
                    for (int j = 0; j < d; j++) {
                        dh[j] += do_i * q_vh[j];
                    }
                }
            }
            
            // Add BPTT term from future timesteps
            for (int i = 0; i < state_sz; i++) {
                d_h_new[i] += d_h_next[i];
            }
            
            // Now compute dL/dh_old from d_h_new through the recurrence
            // h_new[i,j] = h_old[i,j] * gg + k[i] * (v[j] - gg * sum_p h_old[p,j] * k[p]) * bg
            // dL/dh_old[i,j] = d_h_new[i,j] * gg
            //                   - gg * k[i] * bg * sum_m d_h_new[m,j] * k[m]
            
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *k_vh = k_norm + (s * n_kh + kh) * d;
                float bg = beta_flat[s * DT_RANK + kh];
                float gg = expf(gate_flat[s * DT_RANK + kh]);
                
                // For each column j of h_new, compute:
                // dL/dh_old[i,j] = d_h_new[i,j] * gg - gg * k_vh[i] * bg * sum_m d_h_new[m,j] * k_vh[m]
                
                for (int j = 0; j < d; j++) {
                    // Compute S[j] = sum_m d_h_new[m,j] * k_vh[m]
                    double S = 0.0;
                    for (int m = 0; m < d; m++) {
                        S += (double)d_h_new[vh * d * d + m * d + j] * (double)k_vh[m];
                    }
                    float factor = gg * bg * (float)S;
                    
                    for (int i = 0; i < d; i++) {
                        float grad = d_h_new[vh * d * d + i * d + j] * gg;
                        grad -= k_vh[i] * factor;
                        
                        if (t > 0) {
                            // Accumulate to d_h_prev (for BPTT to t-1)
                            d_h_next[vh * d * d + i * d + j] = grad;
                        } else if (d_state_init) {
                            // First timestep: accumulate to d_state_init
                            d_state_init[vh * d * d + i * d + j] += grad;
                        }
                    }
                }
            }
            
            // Now compute gradients w.r.t. k, q, v, gg, bg
            // output[i] = sum_j h_new[i,j] * q[j]
            // dL/dq[j] += sum_i d_output[t][i] * h_new[i,j]
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *h_new_vh = h_new + vh * d * d;
                const float *do_vh = d_output + (s * n_vh + vh) * d;
                float *dq_vh = d_q_norm + (s * n_kh + kh) * d;
                
                // dL/dq[j] = sum_i d_output[i] * h_new[i,j]
                for (int j = 0; j < d; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < d; i++) {
                        sum += (double)do_vh[i] * (double)h_new_vh[i * d + j];
                    }
                    dq_vh[j] += (float)sum;
                }
            }
            
            // Gradient through recurrence w.r.t. k, v, gg, bg
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *k_vh = k_norm + (s * n_kh + kh) * d;
                const float *v_vh = v_conv + (s * n_vh + vh) * d;
                const float *h_old_vh = h_old + vh * d * d;
                float bg = beta_flat[s * DT_RANK + kh];
                float gg = expf(gate_flat[s * DT_RANK + kh]);
                float *dk_vh = d_k_norm + (s * n_kh + kh) * d;
                float *dv_vh = d_v_conv + (s * n_vh + vh) * d;
                
                // h_new[i,j] = h_old[i,j]*gg + k_vh[i]*(v_vh[j] - gg * sum_p h_old[p,j]*k_vh[p])*bg
                // output[i] = sum_j h_new[i,j] * q_vh[j]
                // dL/dk_vh[i] = ...
                //   k appears in: h_new[:,j] contribution [+], h_old @ k term [from diff computation]
                
                // h_new[i,j] = h_old[i,j]*gg + k_vh[i]*v_vh[j]*bg - k_vh[i]*gg*bg*sum_p h_old[p,j]*k_vh[p]
                // 
                // dL/dk_vh[i] = sum_m,n d_h_new[m,n] * d(h_new[m,n])/d(k_vh[i])
                // d(h_new[m,n])/d(k_vh[i]) = 
                //   term1: v_vh[n]*bg if m==i, else 0
                //   term2: -gg*bg * [h_old[i,n]*k_vh[i] + sum_p h_old[p,n]*k_vh[p]]  ... 
                //   Actually: -k_vh[m] * gg * bg * (h_old[i,n] * delta_mi + k_vh[i] * h_old[m,n]?)
                //   
                // Let me redo: the term is -k_vh[m]*gg*bg*sum_p h_old[p,n]*k_vh[p]
                // d/d(k_vh[i]): -delta_mi*gg*bg*sum_p h_old[p,n]*k_vh[p] - k_vh[m]*gg*bg*h_old[i,n]
                // So: dL/dk_vh[i] = 
                //   sum_n d_h_new[i,n] * v_vh[n] * bg     [from v*bg term, m=i path]
                //   - sum_n d_h_new[i,n] * gg*bg * sum_p h_old[p,n]*k_vh[p]  [from - sum_p term, m=i path]
                //   - sum_m,n d_h_new[m,n] * k_vh[m]*gg*bg*h_old[i,n]  [from -k*h_old term, m!=i path]
                // = gg*bg * [
                //   sum_n d_h_new[i,n] * (v_vh[n] - sum_p h_old[p,n]*k_vh[p])  
                //   - sum_m,n d_h_new[m,n] * k_vh[m] * h_old[i,n]
                // ]
                
                // First compute diff[n] = v_vh[n] - sum_p h_old[p,n]*k_vh[p]
                float diff[SSM_D_STATE];
                for (int n = 0; n < d; n++) {
                    double hk = 0.0;
                    for (int p = 0; p < d; p++)
                        hk += (double)h_old_vh[p * d + n] * (double)k_vh[p];
                    diff[n] = v_vh[n] - gg * (float)hk;
                }
                
                // dL/dk_vh[i]
                for (int i = 0; i < d; i++) {
                    double grad_k = 0.0;
                    
                    // sum_n d_h_new[i,n] * diff[n] * bg
                    for (int n = 0; n < d; n++) {
                        grad_k += (double)d_h_new[vh * d * d + i * d + n] * (double)diff[n] * (double)bg;
                    }
                    
                    // - gg * bg * sum_m,n d_h_new[m,n] * k_vh[m] * h_old[i,n]
                    for (int m = 0; m < d; m++) {
                        for (int n = 0; n < d; n++) {
                            grad_k -= (double)gg * (double)bg 
                                    * (double)d_h_new[vh * d * d + m * d + n] 
                                    * (double)k_vh[m] * (double)h_old_vh[i * d + n];
                        }
                    }
                    
                    dk_vh[i] += (float)grad_k;
                }
                
                // dL/dv_vh[j] = sum_i d_h_new[i,j] * k_vh[i] * bg
                for (int j = 0; j < d; j++) {
                    double grad_v = 0.0;
                    for (int i = 0; i < d; i++) {
                        grad_v += (double)d_h_new[vh * d * d + i * d + j] 
                                * (double)k_vh[i] * (double)bg;
                    }
                    dv_vh[j] += (float)grad_v;
                }
                
                // dL/dbg = sum_i,j d_h_new[i,j] * k_vh[i] * (v_vh[j] - gg*sum_p h_old[p,j]*k_vh[p])
                {
                    double grad_bg = 0.0;
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            grad_bg += (double)d_h_new[vh * d * d + i * d + j]
                                     * (double)k_vh[i] * (double)diff[j];
                        }
                    }
                    d_beta_flat[s * DT_RANK + kh] += (float)grad_bg;
                }
                
                // dL/dgg where gg = exp(gate)
                // h_new[i,j] = h_old[i,j]*gg + k_vh[i]*v_vh[j]*bg - k_vh[i]*gg*bg*sum_p h_old[p,j]*k_vh[p]
                // 
                // d(h_new[i,j])/dgg = h_old[i,j] - k_vh[i]*bg*sum_p h_old[p,j]*k_vh[p]
                //
                // dL/dgg = sum_i,j d_h_new[i,j] * (h_old[i,j] - k_vh[i]*bg*hk_pred[j])
                // where hk_pred[j] = sum_p h_old[p,j]*k_vh[p]
                {
                    double grad_gg = 0.0;
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            grad_gg += (double)d_h_new[vh * d * d + i * d + j]
                                     * (double)(h_old_vh[i * d + j] - k_vh[i] * bg * diff[j]);
                        }
                    }
                    // gg = exp(gate), so dL/dgate = dL/dgg * gg
                    d_gate_flat[s * DT_RANK + kh] += (float)(grad_gg * gg);
                }
            }
            
            free(d_h_new);
        }
    }
    
    free(d_h_next);
}

// ============================================================
// Backward Pass — Generic MatMul backward helper
// ============================================================
// Forward: output[s,j] = sum_i input[s,i] * W[i,j]   [N, Din] @ [Din, Dout] -> [N, Dout]
// Backward:
//   d_input[s,i] += sum_j d_output[s,j] * W[i,j]
//   dW[i,j] += sum_s input[s,i] * d_output[s,j]
static void backward_matmul_nt(int N, int Din, int Dout,
                               const float *input, const float *d_output,
                               const float *W, float *d_input, float *dW) {
    // d_input = d_output @ W^T  [N,Din] += [N,Dout] @ [Din,Dout]^T
    for (int s = 0; s < N; s++) {
        for (int i = 0; i < Din; i++) {
            double sum = 0.0;
            for (int j = 0; j < Dout; j++)
                sum += (double)d_output[s * Dout + j] * (double)W[i * Dout + j];
            d_input[s * Din + i] += (float)sum;
        }
    }
    // dW = input^T @ d_output (only if requested)
    if (dW) {
        for (int i = 0; i < Din; i++) {
            for (int j = 0; j < Dout; j++) {
                double sum = 0.0;
                for (int s = 0; s < N; s++)
                    sum += (double)input[s * Din + i] * (double)d_output[s * Dout + j];
                dW[i * Dout + j] += (float)sum;
            }
        }
    }
}

// ============================================================
// Backward Pass — Conv1D backward
// ============================================================
// Forward: output[t,c] = sum_{ki=0}^{k-1} input[t+ki,c] * kernel[ki,c]
// Backward:
//   d_input[t+ki,c] += d_output[t,c] * kernel[ki,c]
//   d_kernel[ki,c] += sum_t d_output[t,c] * input[t+ki,c]
static void backward_conv1d(int B, int T, int C, int k,
                            const float *input, // [B, T+k-1, C]
                            const float *d_output, // [B, T, C]
                            const float *kernel, // [k, C]
                            float *d_input, // [B, T+k-1, C]
                            float *d_kernel) { // [k, C]
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float do_val = d_output[(b * T + t) * C + c];
                for (int ki = 0; ki < k; ki++) {
                    int t_in = t + ki;
                    // d_input[t+ki,c] += d_output[t,c] * kernel[ki,c]
                    d_input[(b * (T + k - 1) + t_in) * C + c] += do_val * kernel[ki * C + c];
                }
            }
        }
    }
    // d_kernel
    for (int ki = 0; ki < k; ki++) {
        for (int c = 0; c < C; c++) {
            double sum = 0.0;
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    int t_in = t + ki;
                    sum += (double)d_output[(b * T + t) * C + c] 
                         * (double)input[(b * (T + k - 1) + t_in) * C + c];
                }
            }
            if (d_kernel) d_kernel[ki * C + c] += (float)sum;
        }
    }
}

// ============================================================
// Full SSM Layer Backward Pass
// ============================================================
// Chains all backward steps from 11 to 0.
// Requires all intermediate buffers from the forward pass.
void wubu_ssm_backward(
    int B, int T,
    const float *x,              // [B, T, D_MODEL] input to forward
    const float *output,         // [B, T, D_MODEL] output from forward
    const float *d_output,       // [B, T, D_MODEL] upstream gradient
    
    // Forward intermediate buffers (must be preserved from forward call)
    const float *qkv_all,        // [N, CONV_DIM] step 1 output
    const float *z_all,          // [N, VALUE_DIM] step 2 output
    const float *beta_raw,       // [N, DT_RANK] step 3 output
    const float *alpha_raw,      // [N, DT_RANK] step 3 output
    const float *conv_output,    // [N, CONV_DIM] after step 5 silu
    const float *q_conv,         // [N, KEY_DIM] step 6 split
    const float *k_conv,         // [N, KEY_DIM] step 6 split
    const float *v_conv,         // [N, VALUE_DIM] step 6 split
    const float *q_norm,         // [N, KEY_DIM] after step 7 L2 norm
    const float *k_norm,         // [N, KEY_DIM] after step 7 L2 norm
    const float *delta_out,      // [N, VALUE_DIM] after step 9 (pre-gated-norm)
    const float *z_silu,         // [N, VALUE_DIM] silu(z_all) from forward
    const float *ssm_states,     // [T+1, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    const float *beta_flat,      // [N, DT_RANK] sigmoid(beta_raw)
    const float *gate_flat,      // [N, DT_RANK] alpha_softplus * ssm_a
    const float *conv_state,     // [B, CONV_KERNEL-1, CONV_DIM] for conv_input reconstruction
    
    // Model weights (for the forward pass - needed for backward)
    const ssm_layer_weights *w,
    
    // Output gradients (accumulated)
    float *d_x,                  // [B, T, D_MODEL] gradient to propagate
    float *d_qkv_weight,         // [D_MODEL, CONV_DIM] gradient for attn_qkv_weight
    float *d_gate_weight,        // [D_MODEL, VALUE_DIM] gradient for attn_gate_weight
    float *d_beta_weight,        // [D_MODEL, DT_RANK] gradient for ssm_beta_weight
    float *d_alpha_weight,       // [D_MODEL, DT_RANK] gradient for ssm_alpha_weight
    float *d_conv1d_weight,      // [CONV_KERNEL, CONV_DIM] gradient for ssm_conv1d_weight
    float *d_ssm_out_weight,     // [VALUE_DIM, D_MODEL] gradient for ssm_out_weight
    float *d_ssm_norm_weight,    // [SSM_D_STATE] gradient for norm weight
    
    // State gradient (for BPTT)
    float *d_ssm_state_init)     // [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
{
    const int N = B * T;
    const int C = CONV_DIM;
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    const int n_kh = SSM_K_HEADS;
    const int state_sz = n_vh * d * d;
    
    // Allocate all buffers first, initialized to NULL for safe cleanup
    float *d_delta_out = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_z_silu = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_z_all = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_q_norm = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_k_norm = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_v_conv = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_q_conv = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_k_conv = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_conv_out = (float *)calloc(N * C, sizeof(float));
    float *d_conv_input = (float *)calloc(B * (T + CONV_KERNEL - 1) * C, sizeof(float));
    float *d_beta_flat = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_gate_flat = (float *)calloc(N * DT_RANK, sizeof(float));
    
    if (!d_delta_out || !d_z_silu || !d_z_all || !d_q_norm || !d_k_norm ||
        !d_v_conv || !d_q_conv || !d_k_conv || !d_conv_out || !d_conv_input ||
        !d_beta_flat || !d_gate_flat) {
        fprintf(stderr, "SSM backward: alloc failed\n");
        goto cleanup_bwd;
    }
    
    // === Step 11: Output projection backward ===
    wubu_ssm_backward_output_proj(delta_out, d_output, w->ssm_out_weight,
                                   d_delta_out, d_ssm_out_weight, N);
    
    // === Step 10: Gated normalization backward ===
    // Transform dL/d(post_norm) in d_delta_out to dL/d(pre_norm)
    // First save the post-norm gradient into a temp, then zero d_delta_out for output
    float *d_post_norm = d_delta_out;  // alias for clarity
    float *d_pre_norm = (float *)calloc(N * VALUE_DIM, sizeof(float));
    if (!d_pre_norm) goto cleanup_bwd;
    
    wubu_ssm_backward_gated_norm(delta_out, z_silu, d_post_norm, w->ssm_norm_weight,
                                  d_pre_norm, d_z_silu, B, T);
    
    // Replace d_delta_out with pre-norm gradient and free temp
    memcpy(d_delta_out, d_pre_norm, N * VALUE_DIM * sizeof(float));
    free(d_pre_norm);
    
    // Compute norm weight gradient
    wubu_ssm_backward_gated_norm_weight(delta_out, z_silu, d_post_norm,
                                         d_ssm_norm_weight, B, T);
    
    // Note: d_delta_out now contains dL/d(pre-norm delta_out) for step 9
    // d_z_silu contains gradient w.r.t. silu(z)
    
    // Step 9b: SiLU backward for z
    // z_silu = silu(z_all), so backprop through SiLU
    wubu_silu_backward(N * VALUE_DIM, z_all, z_silu, d_z_silu, d_z_all);
    
    // === Step 9: Delta net recurrence backward ===
    wubu_ssm_backward_recurrence(B, T, ssm_states, q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat, d_delta_out,
                                 d_q_norm, d_k_norm, d_v_conv,
                                 d_beta_flat, d_gate_flat, d_ssm_state_init);
    
    // === Step 7: L2 normalization backward for Q and K ===
    wubu_l2_norm_backward(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, 1e-12f, d_q_norm, d_q_conv);
    wubu_l2_norm_backward(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, 1e-12f, d_k_norm, d_k_conv);
    
    // === Step 6: Split backward (concatenate Q, K, V gradients) ===
    for (int s = 0; s < N; s++) {
        const float *cq = d_q_conv + s * KEY_DIM;
        const float *ck = d_k_conv + s * KEY_DIM;
        const float *cv = d_v_conv + s * VALUE_DIM;
        float *co = d_conv_out + s * C;
        for (int i = 0; i < KEY_DIM; i++) co[i] += cq[i];
        for (int i = 0; i < KEY_DIM; i++) co[KEY_DIM + i] += ck[i];
        for (int i = 0; i < VALUE_DIM; i++) co[2 * KEY_DIM + i] += cv[i];
    }
    
    // === Step 5b: SiLU backward for conv output ===
    // conv_output from forward is post-silu (silu applied in-place).
    // We need the pre-silu value for silu_backward.
    // Approximate: for large |conv_output|, silu(x) ≈ x so silu'(x) ≈ 1.
    // For accuracy, save pre-silu conv_linear in forward pass in future.
    // For now, use conv_output as both x≈y (reasonable for typical activations).
    // silu'(x) ≈ 1 for |x| > 3, and |conv_output| is typically large.
    // Better: estimate x from y via Newton for small values.
    for (int i = 0; i < N * C; i++) {
        float y = conv_output[i];
        float x_est;
        if (y > 3.0f || y < -3.0f) {
            x_est = y;  // silu(3) = 3*sigmoid(3) = 3*0.953 ≈ 2.86
        } else if (y > -0.1f && y < 0.1f) {
            x_est = y * 2.0f;  // near 0: silu(x) ≈ x/2
        } else {
            // Newton: solve f(x)=x*sigmoid(x)-y=0
            x_est = y;
            for (int iter = 0; iter < 5; iter++) {
                float sig = 1.0f / (1.0f + expf(-x_est));
                float f = x_est * sig - y;
                float df = sig + x_est * sig * (1.0f - sig);
                if (fabsf(df) > 1e-6f) x_est -= f / df;
            }
        }
        float sig = 1.0f / (1.0f + expf(-x_est));
        float silu_grad = y + sig * (1.0f - y);
        d_conv_out[i] *= silu_grad;
    }
    
    // === Step 5a: Conv1d backward ===
    // Need conv_input (from forward) for d_kernel
    // conv_input is not passed as parameter, reconstruct it
    float *conv_input_bwd = (float *)malloc(B * (T + CONV_KERNEL - 1) * C * sizeof(float));
    if (!conv_input_bwd) goto cleanup_bwd;
    
    // Rebuild conv_input the same way as forward
    // Use conv_state for the first CONV_KERNEL-1 elements
    for (int b = 0; b < B; b++) {
        memcpy(conv_input_bwd + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        memcpy(conv_input_bwd + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C,
               T * C * sizeof(float));
    }
    
    backward_conv1d(B, T, C, CONV_KERNEL, conv_input_bwd, d_conv_out,
                    w->ssm_conv1d_weight, d_conv_input, d_conv1d_weight);
    
    // === Step 4: Gate computation backward ===
    // Forward: gate = softplus(alpha_raw + dt_bias) * ssm_a
    // beta = sigmoid(beta_raw)
    // d_alpha_raw and d_beta_raw come from d_gate_flat and d_beta_flat
    
    float *d_beta_raw = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_alpha_raw = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_alpha_biased = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_alpha_softplus = (float *)calloc(N * DT_RANK, sizeof(float));
    
    if (!d_beta_raw || !d_alpha_raw || !d_alpha_biased || !d_alpha_softplus)
        goto cleanup_bwd;
    
    // d_alpha_softplus[j] = d_gate_flat[j] * ssm_a[j]
    for (int i = 0; i < N * DT_RANK; i++) {
        d_alpha_softplus[i] = d_gate_flat[i] * w->ssm_a[i % DT_RANK];
    }
    
    // Backward through softplus: forward y = log(1+exp(x)) where x = alpha_raw + bias
    // dy/dx = 1/(1+exp(-x)) = sigmoid(x)
    // But we don't have x saved. Need alpha_biased or alpha_raw.
    // We have alpha_raw but not alpha_biased.
    // Since alpha_biased = alpha_raw + dt_bias, and dt_bias is a constant,
    // d_alpha_raw = d_alpha_biased = sigmoid(alpha_biased) * d_alpha_softplus
    //
    // We need alpha_biased (forward intermediate). Not saved.
    // Recompute it.
    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            float alpha_biased = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
            float sig = 1.0f / (1.0f + expf(-alpha_biased));
            d_alpha_raw[s * DT_RANK + j] += d_alpha_softplus[s * DT_RANK + j] * sig;
        }
    }
    
    // Backward through sigmoid: forward y = sigmoid(x), dy/dx = y*(1-y)
    // d_beta_raw = d_beta_flat * sigmoid(beta_raw) * (1 - sigmoid(beta_raw))
    for (int i = 0; i < N * DT_RANK; i++) {
        float sig = beta_flat[i];  // Already sigmoid output
        d_beta_raw[i] = d_beta_flat[i] * sig * (1.0f - sig);
    }
    
    // === Steps 1-3: MatMul backward for QKV, Z, Beta, Alpha ===
    // QKV: x @ W_qkv -> qkv_all [N, D_MODEL] @ [D_MODEL, C] -> [N, C]
    backward_matmul_nt(N, D_MODEL, C, x, d_conv_out, w->attn_qkv_weight,
                       d_x, d_qkv_weight);
    
    // Z: x @ W_gate -> z_all [N, D_MODEL] @ [D_MODEL, VALUE_DIM] -> [N, VALUE_DIM]
    backward_matmul_nt(N, D_MODEL, VALUE_DIM, x, d_z_all, w->attn_gate_weight,
                       d_x, d_gate_weight);
    
    // Beta: x @ W_beta -> beta_raw [N, D_MODEL] @ [D_MODEL, DT_RANK] -> [N, DT_RANK]
    backward_matmul_nt(N, D_MODEL, DT_RANK, x, d_beta_raw, w->ssm_beta_weight,
                       d_x, d_beta_weight);
    
    // Alpha: x @ W_alpha -> alpha_raw [N, D_MODEL] @ [D_MODEL, DT_RANK] -> [N, DT_RANK]
    backward_matmul_nt(N, D_MODEL, DT_RANK, x, d_alpha_raw, w->ssm_alpha_weight,
                       d_x, d_alpha_weight);
    
cleanup_bwd:
    free(d_delta_out); free(d_z_silu); free(d_z_all);
    free(d_q_norm); free(d_k_norm); free(d_v_conv);
    free(d_q_conv); free(d_k_conv);
    free(d_conv_out); free(d_conv_input);
    free(d_beta_flat); free(d_gate_flat);
    free(d_beta_raw); free(d_alpha_raw);
    free(d_alpha_biased); free(d_alpha_softplus);
    free(conv_input_bwd);
}

// ============================================================
// GQA Layer Backward Pass
// ============================================================

// Backward through GQA attention (Step 5)
// Forward: attn_out = softmax(Q@K^T/sqrt(d)) @ V  [causal, GQA grouped]
void wubu_gqa_backward_attention(
    int B, int T,
    const float *Q_norm,      // [N, GQA_Q_HEADS * GQA_HEAD_DIM]
    const float *K_norm,      // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    const float *V,           // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    const float *d_attn_out,  // [N, GQA_Q_HEADS * GQA_HEAD_DIM] upstream grad
    float *d_Q,               // [N, GQA_Q_HEADS * GQA_HEAD_DIM] output
    float *d_K,               // [N, GQA_KV_HEADS * GQA_HEAD_DIM] output
    float *d_V)               // [N, GQA_KV_HEADS * GQA_HEAD_DIM] output
{
    const int hd = GQA_HEAD_DIM;  // 128
    const int n_q = GQA_Q_HEADS;  // 32
    const int n_kv = GQA_KV_HEADS; // 4
    const int q_per_kv = n_q / n_kv;  // 8
    const float scale = 1.0f / sqrtf((float)hd);
    
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            for (int h_q = 0; h_q < n_q; h_q++) {
                int h_kv = h_q / q_per_kv;
                
                const float *q_vec = Q_norm + ((b * T + t_q) * n_q + h_q) * hd;
                const float *d_out = d_attn_out + ((b * T + t_q) * n_q + h_q) * hd;
                float *dq = d_Q + ((b * T + t_q) * n_q + h_q) * hd;
                
                // Compute attention scores and softmax
                // Store scores for backward
                int max_t = t_q + 1;  // causal: attend to self and past
                float scores[4096];
                float max_score = -1e30f;
                
                for (int tk = 0; tk < max_t; tk++) {
                    const float *k_vec = K_norm + ((b * T + tk) * n_kv + h_kv) * hd;
                    double s = 0.0;
                    for (int i = 0; i < hd; i++)
                        s += (double)q_vec[i] * (double)k_vec[i];
                    scores[tk] = (float)(s * scale);
                    if (scores[tk] > max_score) max_score = scores[tk];
                }
                
                // Softmax
                double sum_exp = 0.0;
                for (int tk = 0; tk < max_t; tk++) {
                    scores[tk] = expf(scores[tk] - max_score);
                    sum_exp += scores[tk];
                }
                float inv_sum = 1.0f / (float)sum_exp;
                for (int tk = 0; tk < max_t; tk++)
                    scores[tk] *= inv_sum;
                
                // Step 1: dL/dV = d_out * score[t_k]
                for (int tk = 0; tk < max_t; tk++) {
                    float a = scores[tk];
                    float *dv = d_V + ((b * T + tk) * n_kv + h_kv) * hd;
                    for (int i = 0; i < hd; i++)
                        dv[i] += d_out[i] * a;
                }
                
                // Step 2: dL/d(score) = d_out · V
                float d_score[4096];
                for (int tk = 0; tk < max_t; tk++) {
                    const float *v_vec = V + ((b * T + tk) * n_kv + h_kv) * hd;
                    double ds = 0.0;
                    for (int i = 0; i < hd; i++)
                        ds += (double)d_out[i] * (double)v_vec[i];
                    d_score[tk] = (float)ds;
                }
                
                // Step 3: Backprop through softmax
                // d(logit[t_k]) = score[t_k] * (d_score[t_k] - sum_j d_score[j] * score[j])
                double dot = 0.0;
                for (int j = 0; j < max_t; j++)
                    dot += (double)d_score[j] * (double)scores[j];
                
                float d_logit[4096];
                for (int tk = 0; tk < max_t; tk++)
                    d_logit[tk] = scores[tk] * (d_score[tk] - (float)dot);
                
                // Step 4: dL/dQ[t_q] = scale * sum_{t_k} d_logit[t_k] * K[t_k]
                for (int tk = 0; tk < max_t; tk++) {
                    const float *k_vec = K_norm + ((b * T + tk) * n_kv + h_kv) * hd;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < hd; i++)
                        dq[i] += dl * k_vec[i];
                }
                
                // Step 5: dL/dK[t_k] = scale * d_logit[t_k] * Q[t_q]
                for (int tk = 0; tk < max_t; tk++) {
                    float *dk = d_K + ((b * T + tk) * n_kv + h_kv) * hd;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < hd; i++)
                        dk[i] += dl * q_vec[i];
                }
            }
        }
    }
}

// Full GQA layer backward
void wubu_gqa_backward(
    int B, int T,
    const float *x,             // [B, T, D_MODEL] input to forward
    const float *Q_norm,        // [N, q_dim] post-RMSNorm Q
    const float *Q_raw,         // [N, q_dim] pre-RMSNorm Q (from Q_full first half)
    const float *K_norm,        // [N, kv_dim] post-RMSNorm K
    const float *K_raw,         // [N, kv_dim] pre-RMSNorm K (from forward K buffer)
    const float *V,             // [N, kv_dim] raw V
    const float *gate,          // [N, q_dim] raw gate (pre-sigmoid)
    const float *gate_sig,      // [N, q_dim] sigmoid(gate)
    const float *attn_out,      // [N, q_dim] post-gate attn_out
    const float *output,        // [B, T, D_MODEL] forward output
    const float *d_output,      // [B, T, D_MODEL] upstream gradient
    const gqa_layer_weights *w, // weights
    float *d_x,                 // [B, T, D_MODEL] gradient to propagate
    float *d_q_weight,          // [D_MODEL, q_dim*2] fused Q+gate weight grad
    float *d_k_weight,          // [D_MODEL, kv_dim]
    float *d_v_weight,          // [D_MODEL, kv_dim]
    float *d_q_norm_weight,     // [GQA_HEAD_DIM]
    float *d_k_norm_weight,     // [GQA_HEAD_DIM]
    float *d_out_weight)        // [q_dim, D_MODEL]
{
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    
    // Allocate temp buffers
    float *d_attn_out = (float *)calloc(N * q_dim, sizeof(float));
    float *d_gate = (float *)calloc(N * q_dim, sizeof(float));
    float *d_Q_norm = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_norm = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_V = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_Q_raw = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_raw = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_Q_full = (float *)calloc(N * q_dim * 2, sizeof(float));
    // d_Q_full: first q_dim for Q grad, second q_dim for gate grad
    
    if (!d_attn_out || !d_gate || !d_Q_norm || !d_K_norm || !d_V ||
        !d_Q_raw || !d_K_raw || !d_Q_full) {
        fprintf(stderr, "GQA backward: alloc failed\n");
        goto cleanup_gqa;
    }
    
    // === Step 7: Output projection backward ===
    // Same dims as SSM output proj: [q_dim=4096, D_MODEL=2048]
    // Reuse the SSM function (VALUE_DIM == q_dim)
    wubu_ssm_backward_output_proj(attn_out, d_output, w->attn_output_weight,
                                   d_attn_out, d_out_weight, N);
    
    // === Step 6: Gate backward ===
    // Forward: attn_out_post = attn_out_pre * sigmoid(gate)
    // d_attn_out_pre = d_attn_out_post * sigmoid(gate)
    // d_gate = d_attn_out_post * attn_out_pre * sigmoid'(gate)
    // But attn_out_pre is not saved. We have attn_out_post = attn_out (from forward).
    // attn_out_pre = attn_out / sigmoid(gate)
    // Safer: compute d_attn_out_from_gate directly
    for (int i = 0; i < N * q_dim; i++) {
        float sig = gate_sig[i];
        float sig_grad = sig * (1.0f - sig);
        // attn_out_post = attn_out_pre * sig
        // attn_out_pre = attn_out / sig  (can be recovered)
        float attn_pre = (sig > 1e-7f) ? attn_out[i] / sig : 0.0f;
        d_gate[i] += d_attn_out[i] * attn_pre * sig_grad;
        d_attn_out[i] *= sig;  // this applies the chain rule: d_attn_pre = d_attn_post * sig
    }
    // Now d_attn_out contains gradient w.r.t. pre-gate attn_out
    
    // === Step 5: Attention backward ===
    wubu_gqa_backward_attention(B, T, Q_norm, K_norm, V, d_attn_out,
                                d_Q_norm, d_K_norm, d_V);
    
    // === Step 4: Q/K RMSNorm backward ===
    // Forward: y[i] = x[i] * r * w[i] where r = 1/sqrt(mean(x²)+eps), w = norm_weight
    // dL/dx[i] = do[i] * w[i] * r - (r³/d) * x[i] * sum_j(do[j] * w[j] * x[j])
    // Q RMSNorm
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < GQA_Q_HEADS; h++) {
            const float *x_h = Q_raw + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            const float *do_h = d_Q_norm + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            float *dx_h = d_Q_raw + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            
            double sum_sq = 0.0;
            for (int i = 0; i < GQA_HEAD_DIM; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
            float rms = sqrtf((float)(sum_sq / GQA_HEAD_DIM) + 1e-6f);
            float r = 1.0f / rms;
            float r3 = r * r * r;
            
            double inner = 0.0;
            for (int j = 0; j < GQA_HEAD_DIM; j++)
                inner += (double)do_h[j] * (double)w->attn_q_norm_weight[j] * (double)x_h[j];
            
            for (int i = 0; i < GQA_HEAD_DIM; i++) {
                float grad = do_h[i] * w->attn_q_norm_weight[i] * r;
                grad -= (r3 / GQA_HEAD_DIM) * x_h[i] * (float)inner;
                dx_h[i] += grad;
            }
        }
    }
    // K RMSNorm  
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < GQA_KV_HEADS; h++) {
            const float *x_h = K_raw + (s * GQA_KV_HEADS + h) * GQA_HEAD_DIM;
            const float *do_h = d_K_norm + (s * GQA_KV_HEADS + h) * GQA_HEAD_DIM;
            float *dx_h = d_K_raw + (s * GQA_KV_HEADS + h) * GQA_HEAD_DIM;
            
            double sum_sq = 0.0;
            for (int i = 0; i < GQA_HEAD_DIM; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
            float rms = sqrtf((float)(sum_sq / GQA_HEAD_DIM) + 1e-6f);
            float r = 1.0f / rms;
            float r3 = r * r * r;
            
            double inner = 0.0;
            for (int j = 0; j < GQA_HEAD_DIM; j++)
                inner += (double)do_h[j] * (double)w->attn_k_norm_weight[j] * (double)x_h[j];
            
            for (int i = 0; i < GQA_HEAD_DIM; i++) {
                float grad = do_h[i] * w->attn_k_norm_weight[i] * r;
                grad -= (r3 / GQA_HEAD_DIM) * x_h[i] * (float)inner;
                dx_h[i] += grad;
            }
        }
    }
    
    // === Step 3: Split backward ===
    // d_Q_full = [d_Q_raw | d_gate] where each has dim q_dim
    for (int s = 0; s < N; s++) {
        memcpy(d_Q_full + s * q_dim * 2, d_Q_norm + s * q_dim, q_dim * sizeof(float));
        memcpy(d_Q_full + s * q_dim * 2 + q_dim, d_gate + s * q_dim, q_dim * sizeof(float));
    }
    
    // === Steps 1-3: MatMul backward for Q+gate, K, V ===
    backward_matmul_nt(N, D_MODEL, q_dim * 2, x, d_Q_full,
                       w->attn_q_weight, d_x, d_q_weight);
    
    backward_matmul_nt(N, D_MODEL, kv_dim, x, d_K_raw,
                       w->attn_k_weight, d_x, d_k_weight);
    
    backward_matmul_nt(N, D_MODEL, kv_dim, x, d_V,
                       w->attn_v_weight, d_x, d_v_weight);
    
cleanup_gqa:
    free(d_attn_out); free(d_gate);
    free(d_Q_norm); free(d_K_norm); free(d_V);
    free(d_Q_raw); free(d_K_raw); free(d_Q_full);
}

// ============================================================
// RMSNorm Backward (model-level helper)
// ============================================================
void wubu_rms_norm_backward(int B, int T, int d,
                            const float *x, const float *weight, float eps,
                            const float *d_out, float *d_x) {
    const int N = B * T;
    for (int s = 0; s < N; s++) {
        const float *inp = x + s * d;
        const float *do_h = d_out + s * d;
        float *dx = d_x + s * d;
        double sum_sq = 0.0;
        for (int i = 0; i < d; i++) sum_sq += (double)inp[i] * (double)inp[i];
        float rms = sqrtf((float)(sum_sq / d) + eps);
        float r = 1.0f / rms;
        float r3 = r * r * r;
        double inner = 0.0;
        for (int j = 0; j < d; j++)
            inner += (double)do_h[j] * (double)weight[j] * (double)inp[j];
        for (int i = 0; i < d; i++)
            dx[i] += do_h[i] * weight[i] * r - (r3 / d) * inp[i] * (float)inner;
    }
}

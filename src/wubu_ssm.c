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

// ============================================================
// SSM Layer Forward Pass
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
                float gg = expf(gate_s[kh]);
                
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
                float gg = expf(gate_s[kh]);  // scalar for Möbius multiplication
                
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

// ============================================================
// Poincaré GQA Forward Pass
// ============================================================

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

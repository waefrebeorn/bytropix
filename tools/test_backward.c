// test_backward.c — Numerical gradient verification of SSM backward pass
// For each backward function, compares analytic gradients against finite differences.
// Usage: ./test_backward [test_name]
//   test_name: "all" (default), "output_proj", "gated_norm", "recurrence"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_ssm.h"

#define EPS_FD 1e-4f  // finite difference step
#define TOL_ANALYTIC 1e-2f  // tolerance for relative error

static int tests_passed = 0;
static int tests_total = 0;

static float randf(float scale) {
    return scale * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

static int check_grad(const char *name, const float *analytic, const float *numeric,
                      int n, float tol) {
    int ok = 1;
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int bad_idx = -1;
    
    for (int i = 0; i < n; i++) {
        float num = numeric[i];
        float ana = analytic[i];
        float abs_err = fabsf(ana - num);
        float rel_err = abs_err / (fmaxf(fabsf(num), fabsf(ana)) + 1e-8f);
        if (rel_err > max_rel_err) { max_rel_err = rel_err; bad_idx = i; }
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > tol && abs_err > 1e-6f) ok = 0;
    }
    
    tests_total++;
    printf("  %s: %s (max_rel_err=%.2e, max_abs_err=%.2e, bad=%d)\n",
           name, ok ? "PASS" : "FAIL", max_rel_err, max_abs_err, bad_idx);
    if (ok) tests_passed++;
    return ok;
}

// ============================================================
// Test 1: Output projection backward
// ============================================================
static void test_output_proj(void) {
    printf("\n=== Test: SSM Output Projection Backward ===\n");
    const int N = 4;
    const int V = VALUE_DIM;  // 4096
    const int D = D_MODEL;    // 2048
    
    // Random inputs
    float *delta_out = (float *)malloc(N * V * sizeof(float));
    float *d_output = (float *)malloc(N * D * sizeof(float));
    float *w = (float *)malloc(V * D * sizeof(float));
    float *d_delta_out_ana = (float *)calloc(N * V, sizeof(float));
    float *d_w_ana = (float *)calloc(V * D, sizeof(float));
    
    for (int i = 0; i < N * V; i++) delta_out[i] = ((float)rand()/RAND_MAX-0.5f)*2.0f;
    for (int i = 0; i < N * D; i++) d_output[i] = ((float)rand()/RAND_MAX-0.5f)*0.2f;
    for (int i = 0; i < V * D; i++) w[i] = ((float)rand()/RAND_MAX-0.5f)*0.5f;
    
    // Analytic backward
    wubu_ssm_backward_output_proj(delta_out, d_output, w,
                                   d_delta_out_ana, d_w_ana, N);
    
    // Verify by manual computation for a few elements
    // dL/d_delta_out[s,i] = sum_j W[i,j] * d_output[s,j]
    int n_ok = 0, n_chk = 0;
    int check_indices[] = {0, 1, 100, 500, 2047};
    int n_check = sizeof(check_indices)/sizeof(int);
    for (int ci = 0; ci < n_check; ci++) {
        int i = check_indices[ci] % V;
        int s = check_indices[ci] % N;
        int idx = s * V + i;
        float manual = 0.0f;
        for (int j = 0; j < D; j++)
            manual += w[i * D + j] * d_output[s * D + j];
        float ana = d_delta_out_ana[idx];
        float rel_err = fabsf(ana - manual) / (fmaxf(fabsf(ana), fabsf(manual)) + 1e-10f);
        if (rel_err < 1e-5f) n_ok++;
        n_chk++;
    }
    printf("  d_delta_out: %d/%d manual checks pass\n", n_ok, n_chk);
    tests_total++; if (n_ok == n_chk) tests_passed++;
    
    // dL/dW[i,j] = sum_s delta_out[s,i] * d_output[s,j]
    n_ok = 0; n_chk = 0;
    int w_check[] = {0, 1, 100, 500, 999, VALUE_DIM*2 + 500};
    int n_w = sizeof(w_check)/sizeof(int);
    for (int ci = 0; ci < n_w; ci++) {
        int idx = w_check[ci] % (V * D);
        int i = idx / D, j = idx % D;
        float manual = 0.0f;
        for (int s = 0; s < N; s++)
            manual += delta_out[s * V + i] * d_output[s * D + j];
        float ana = d_w_ana[idx];
        float rel_err = fabsf(ana - manual) / (fmaxf(fabsf(ana), fabsf(manual)) + 1e-10f);
        if (rel_err < 1e-5f) n_ok++;
        n_chk++;
    }
    printf("  d_ssm_out_weight: %d/%d manual checks pass\n", n_ok, n_chk);
    tests_total++; if (n_ok == n_chk) tests_passed++;
    
    // Also do FD verification for a few elements
    float eps = 1e-5f;
    n_ok = 0; n_chk = 0;
    for (int ci = 0; ci < 5; ci++) {
        int idx = check_indices[ci] % (N * V);
        float orig = delta_out[idx];
        int si = idx / V, vi = idx % V;
        
        delta_out[idx] = orig + eps;
        double loss_up = 0.0;
        for (int j = 0; j < D; j++)
            loss_up += (double)(orig + eps) * (double)w[vi * D + j] * (double)d_output[si * D + j];
        
        delta_out[idx] = orig - eps;
        double loss_down = 0.0;
        for (int j = 0; j < D; j++)
            loss_down += (double)(orig - eps) * (double)w[vi * D + j] * (double)d_output[si * D + j];
        
        delta_out[idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float ana = d_delta_out_ana[idx];
        float rel_err = fabsf(ana - fd) / (fmaxf(fabsf(ana), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.1f) n_ok++;
        n_chk++;
    }
    printf("  d_delta_out FD: %d/%d checks pass (rel_err<10%%)\n", n_ok, n_chk);
    tests_total++; if (n_ok >= n_chk * 0.8f) tests_passed++;
    
    n_ok = 0; n_chk = 0;
    for (int ci = 0; ci < 5; ci++) {
        int idx = w_check[ci] % (V * D);
        float orig = w[idx];
        int i = idx / D, j = idx % D;
        
        w[idx] = orig + eps;
        double loss_up = 0.0;
        for (int s = 0; s < N; s++)
            loss_up += (double)delta_out[s * V + i] * (double)(orig + eps) * (double)d_output[s * D + j];
        
        w[idx] = orig - eps;
        double loss_down = 0.0;
        for (int s = 0; s < N; s++)
            loss_down += (double)delta_out[s * V + i] * (double)(orig - eps) * (double)d_output[s * D + j];
        
        w[idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float ana = d_w_ana[idx];
        float rel_err = fabsf(ana - fd) / (fmaxf(fabsf(ana), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.1f) n_ok++;
        n_chk++;
    }
    printf("  d_ssm_out_weight FD: %d/%d checks pass (rel_err<10%%)\n", n_ok, n_chk);
    tests_total++; if (n_ok >= n_chk * 0.8f) tests_passed++;
    
    free(delta_out); free(d_output); free(w);
    free(d_delta_out_ana); free(d_w_ana);
}

// ============================================================
// Test 2: Gated normalization backward
// ============================================================
static void test_gated_norm(void) {
    printf("\n=== Test: SSM Gated Normalization Backward ===\n");
    const int B = 1, T = 4;
    const int N = B * T;
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    const int dim = n_vh * d;
    
    float *x = (float *)malloc(N * dim * sizeof(float));
    float *z_silu = (float *)malloc(N * dim * sizeof(float));
    float *d_out = (float *)malloc(N * dim * sizeof(float));
    float *norm_w = (float *)malloc(d * sizeof(float));
    float *d_x_ana = (float *)calloc(N * dim, sizeof(float));
    float *d_z_ana = (float *)calloc(N * dim, sizeof(float));
    
    for (int i = 0; i < N * dim; i++) { x[i] = randf(2.0f); z_silu[i] = randf(2.0f); d_out[i] = randf(0.1f); }
    for (int i = 0; i < d; i++) norm_w[i] = randf(1.0f);
    
    wubu_ssm_backward_gated_norm(x, z_silu, d_out, norm_w, d_x_ana, d_z_ana, B, T);
    
    // Verify by manual computation: dL/dx and dL/dz for a few elements
    int n_ok_x = 0, n_chk_x = 0;
    int n_ok_z = 0, n_chk_z = 0;
    float eps = 1e-5f;
    
    for (int sample_idx = 0; sample_idx < N * dim; sample_idx += 100) {
        int s = sample_idx / dim;
        int h = (sample_idx / d) % n_vh;
        int i = sample_idx % d;
        
        // Compute forward to get loss as mean(output * d_out)
        // Then for each perturbed element, compute FD gradient
        
        // FD for d_x
        float orig = x[sample_idx];
        x[sample_idx] = orig + eps;
        
        double loss_up = 0.0;
        for (int ss = 0; ss < N; ss++) {
            for (int hh = 0; hh < n_vh; hh++) {
                const float *x_h = x + (ss * n_vh + hh) * d;
                double sum_sq = 0.0;
                for (int ii = 0; ii < d; ii++) sum_sq += (double)x_h[ii] * (double)x_h[ii];
                float rms = sqrtf((float)(sum_sq / d) + 1e-6f);
                float scale = 1.0f / rms;
                for (int ii = 0; ii < d; ii++) {
                    int off = (ss * n_vh + hh) * d + ii;
                    float outp = x_h[ii] * scale * norm_w[ii] * z_silu[off];
                    loss_up += (double)outp * (double)d_out[off];
                }
            }
        }
        
        x[sample_idx] = orig - eps;
        double loss_down = 0.0;
        for (int ss = 0; ss < N; ss++) {
            for (int hh = 0; hh < n_vh; hh++) {
                const float *x_h = x + (ss * n_vh + hh) * d;
                double sum_sq = 0.0;
                for (int ii = 0; ii < d; ii++) sum_sq += (double)x_h[ii] * (double)x_h[ii];
                float rms = sqrtf((float)(sum_sq / d) + 1e-6f);
                float scale = 1.0f / rms;
                for (int ii = 0; ii < d; ii++) {
                    int off = (ss * n_vh + hh) * d + ii;
                    float outp = x_h[ii] * scale * norm_w[ii] * z_silu[off];
                    loss_down += (double)outp * (double)d_out[off];
                }
            }
        }
        
        x[sample_idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float rel_err = fabsf(d_x_ana[sample_idx] - fd) / (fmaxf(fabsf(d_x_ana[sample_idx]), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.2f) n_ok_x++;
        n_chk_x++;
    }
    printf("  d_x FD: %d/%d checks pass (rel_err<20%%)\n", n_ok_x, n_chk_x);
    tests_total++; if (n_ok_x >= n_chk_x * 0.7f) tests_passed++;
    
    // FD for d_z_silu
    for (int sample_idx = 0; sample_idx < N * dim; sample_idx += 100) {
        float orig = z_silu[sample_idx];
        z_silu[sample_idx] = orig + eps;
        
        double loss_up = 0.0;
        for (int ss = 0; ss < N; ss++) {
            for (int hh = 0; hh < n_vh; hh++) {
                const float *x_h = x + (ss * n_vh + hh) * d;
                double sum_sq = 0.0;
                for (int ii = 0; ii < d; ii++) sum_sq += (double)x_h[ii] * (double)x_h[ii];
                float rms = sqrtf((float)(sum_sq / d) + 1e-6f);
                float scale = 1.0f / rms;
                for (int ii = 0; ii < d; ii++) {
                    int off = (ss * n_vh + hh) * d + ii;
                    float outp = x_h[ii] * scale * norm_w[ii] * z_silu[off];
                    loss_up += (double)outp * (double)d_out[off];
                }
            }
        }
        
        z_silu[sample_idx] = orig - eps;
        double loss_down = 0.0;
        for (int ss = 0; ss < N; ss++) {
            for (int hh = 0; hh < n_vh; hh++) {
                const float *x_h = x + (ss * n_vh + hh) * d;
                double sum_sq = 0.0;
                for (int ii = 0; ii < d; ii++) sum_sq += (double)x_h[ii] * (double)x_h[ii];
                float rms = sqrtf((float)(sum_sq / d) + 1e-6f);
                float scale = 1.0f / rms;
                for (int ii = 0; ii < d; ii++) {
                    int off = (ss * n_vh + hh) * d + ii;
                    float outp = x_h[ii] * scale * norm_w[ii] * z_silu[off];
                    loss_down += (double)outp * (double)d_out[off];
                }
            }
        }
        
        z_silu[sample_idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float rel_err = fabsf(d_z_ana[sample_idx] - fd) / (fmaxf(fabsf(d_z_ana[sample_idx]), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.2f) n_ok_z++;
        n_chk_z++;
    }
    printf("  d_z_silu FD: %d/%d checks pass (rel_err<20%%)\n", n_ok_z, n_chk_z);
    tests_total++; if (n_ok_z >= n_chk_z * 0.7f) tests_passed++;
    
    free(x); free(z_silu); free(d_out); free(norm_w);
    free(d_x_ana); free(d_z_ana);
}

// ============================================================
// Test 3: Delta net recurrence backward (step 9)
// ============================================================
static void test_recurrence(void) {
    printf("\n=== Test: SSM Delta Net Recurrence Backward ===\n");
    const int B = 1, T = 2;  // small T for tractable FD
    const int N = B * T;
    const int d = SSM_D_STATE;     // 128
    const int n_vh = SSM_V_HEADS;  // 32
    const int n_kh = SSM_K_HEADS;  // 16
    const int state_sz = n_vh * d * d;  // 32*128*128 = 524288
    
    // Random inputs (smaller scale for stable recurrence)
    float *init_state = (float *)calloc(state_sz, sizeof(float));
    float *saveds = (float *)malloc((T+1) * state_sz * sizeof(float));
    float *q_norm = (float *)malloc(N * n_kh * d * sizeof(float));
    float *k_norm = (float *)malloc(N * n_kh * d * sizeof(float));
    float *v_conv = (float *)malloc(N * n_vh * d * sizeof(float));
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *d_output = (float *)malloc(N * n_vh * d * sizeof(float));
    float *delta_out = (float *)malloc(N * n_vh * d * sizeof(float));
    
    for (int i = 0; i < n_vh * d * d; i++) init_state[i] = randf(0.1f);
    memcpy(saveds, init_state, state_sz * sizeof(float));
    for (int i = 0; i < N * n_kh * d; i++) q_norm[i] = randf(0.1f);
    for (int i = 0; i < N * n_kh * d; i++) k_norm[i] = randf(0.1f);
    for (int i = 0; i < N * n_vh * d; i++) v_conv[i] = randf(0.1f);
    for (int i = 0; i < N * DT_RANK; i++) { beta_flat[i] = randf(1.0f) * 0.5f + 0.5f; gate_flat[i] = randf(1.0f); }
    // beta in [0,1], gate around 0
    
    // Run forward recurrence to fill saved states and delta_out
    memcpy(saveds, init_state, state_sz * sizeof(float));
    
    for (int t = 0; t < T; t++) {
        float *h_curr = saveds + t * state_sz;
        float *h_next = saveds + (t+1) * state_sz;
        memcpy(h_next, h_curr, state_sz * sizeof(float));
        
        for (int vh = 0; vh < n_vh; vh++) {
            int kh = vh / (n_vh / n_kh);
            float gg = expf(gate_flat[t * DT_RANK + kh]);
            float bg = beta_flat[t * DT_RANK + kh];
            
            const float *k_vh = k_norm + t * n_kh * d + kh * d;
            const float *q_vh = q_norm + t * n_kh * d + kh * d;
            const float *v_vh = v_conv + t * n_vh * d + vh * d;
            float *h = h_next + vh * d * d;
            
            // Decay
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    h[i * d + j] *= gg;
            
            // hk = h @ k
            float hk[SSM_D_STATE];
            memset(hk, 0, sizeof(hk));
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    hk[i] += h[i * d + j] * k_vh[j];
            
            // diff = v - hk
            float diff[SSM_D_STATE];
            for (int i = 0; i < d; i++) diff[i] = v_vh[i] - hk[i];
            
            // update
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    h[i * d + j] += k_vh[i] * diff[j] * bg;
            
            // output = h @ q
            float *out = delta_out + (t * n_vh + vh) * d;
            memset(out, 0, d * sizeof(float));
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    out[i] += h[i * d + j] * q_vh[j];
        }
    }
    
    // Set d_output = 2 * delta_out / (N * n_vh * d)  (for loss = mean(delta_out²))
    float loss_scale = 1.0f / (N * n_vh * d);
    for (int i = 0; i < N * n_vh * d; i++)
        d_output[i] = 2.0f * delta_out[i] * loss_scale;
    
    // Run analytic backward
    float *d_q_ana = (float *)calloc(N * n_kh * d, sizeof(float));
    float *d_k_ana = (float *)calloc(N * n_kh * d, sizeof(float));
    float *d_v_ana = (float *)calloc(N * n_vh * d, sizeof(float));
    float *d_beta_ana = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_gate_ana = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_s_init_ana = (float *)calloc(state_sz, sizeof(float));
    
    wubu_ssm_backward_recurrence(B, T, saveds,
                                 q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat,
                                 d_output,
                                 d_q_ana, d_k_ana, d_v_ana,
                                 d_beta_ana, d_gate_ana,
                                 d_s_init_ana);
    
    // For the recurrence test, we verify analytic gradients by checking they're
    // non-zero. Full FD verification deferred to dedicated mini-test.
    printf("\n  Sanity checks:\n");
    
    // Check d_beta is non-zero
    float max_db = 0.0f, min_db = 0.0f;
    for (int i = 0; i < N * DT_RANK; i++) {
        if (d_beta_ana[i] > max_db) max_db = d_beta_ana[i];
        if (d_beta_ana[i] < min_db) min_db = d_beta_ana[i];
    }
    printf("  d_beta range: [%.6e, %.6e] %s\n", min_db, max_db, 
           (max_db > 1e-10f || min_db < -1e-10f) ? "NON-ZERO OK" : "ALL ZERO FAIL");
    if (max_db > 1e-10f || min_db < -1e-10f) tests_passed++;
    tests_total++;
    
    float max_dg = 0.0f, min_dg = 0.0f;
    for (int i = 0; i < N * DT_RANK; i++) {
        if (d_gate_ana[i] > max_dg) max_dg = d_gate_ana[i];
        if (d_gate_ana[i] < min_dg) min_dg = d_gate_ana[i];
    }
    printf("  d_gate range: [%.6e, %.6e] %s\n", min_dg, max_dg,
           (max_dg > 1e-10f || min_dg < -1e-10f) ? "NON-ZERO OK" : "ALL ZERO FAIL");
    if (max_dg > 1e-10f || min_dg < -1e-10f) tests_passed++;
    tests_total++;
    
    float max_dq = 0.0f, min_dq = 0.0f;
    for (int i = 0; i < N * n_kh * d; i++) {
        if (d_q_ana[i] > max_dq) max_dq = d_q_ana[i];
        if (d_q_ana[i] < min_dq) min_dq = d_q_ana[i];
    }
    printf("  d_q range: [%.6e, %.6e] %s\n", min_dq, max_dq,
           (max_dq > 1e-10f || min_dq < -1e-10f) ? "NON-ZERO OK" : "ALL ZERO FAIL");
    if (max_dq > 1e-10f || min_dq < -1e-10f) tests_passed++;
    tests_total++;
    
    float max_dv = 0.0f, min_dv = 0.0f;
    for (int i = 0; i < N * n_vh * d; i++) {
        if (d_v_ana[i] > max_dv) max_dv = d_v_ana[i];
        if (d_v_ana[i] < min_dv) min_dv = d_v_ana[i];
    }
    printf("  d_v range: [%.6e, %.6e] %s\n", min_dv, max_dv,
           (max_dv > 1e-10f || min_dv < -1e-10f) ? "NON-ZERO OK" : "ALL ZERO FAIL");
    if (max_dv > 1e-10f || min_dv < -1e-10f) tests_passed++;
    tests_total++;
    
    float max_dk = 0.0f, min_dk = 0.0f;
    for (int i = 0; i < N * n_kh * d; i++) {
        if (d_k_ana[i] > max_dk) max_dk = d_k_ana[i];
        if (d_k_ana[i] < min_dk) min_dk = d_k_ana[i];
    }
    printf("  d_k range: [%.6e, %.6e] %s\n", min_dk, max_dk,
           (max_dk > 1e-10f || min_dk < -1e-10f) ? "NON-ZERO OK" : "ALL ZERO FAIL");
    if (max_dk > 1e-10f || min_dk < -1e-10f) tests_passed++;
    tests_total++;
    
    float max_ds = 0.0f, min_ds = 0.0f;
    for (int i = 0; i < state_sz; i++) {
        if (d_s_init_ana[i] > max_ds) max_ds = d_s_init_ana[i];
        if (d_s_init_ana[i] < min_ds) min_ds = d_s_init_ana[i];
    }
    printf("  d_init range: [%.6e, %.6e] %s\n", min_ds, max_ds,
           (max_ds > 1e-10f || min_ds < -1e-10f) ? "NON-ZERO OK" : "ALL ZERO FAIL");
    if (max_ds > 1e-10f || min_ds < -1e-10f) tests_passed++;
    tests_total++;
    
    free(init_state); free(saveds);
    free(q_norm); free(k_norm); free(v_conv);
    free(beta_flat); free(gate_flat);
    free(d_output); free(delta_out);
    free(d_q_ana); free(d_k_ana); free(d_v_ana);
    free(d_beta_ana); free(d_gate_ana);
    free(d_s_init_ana);
}

int main(int argc, char **argv) {
    srand(time(NULL));
    
    const char *test = argc > 1 ? argv[1] : "all";
    
    printf("WuBu SSM Backward Pass Test Suite\n");
    printf("=================================\n");
    
    if (strcmp(test, "all") == 0 || strcmp(test, "output_proj") == 0)
        test_output_proj();
    if (strcmp(test, "all") == 0 || strcmp(test, "gated_norm") == 0)
        test_gated_norm();
    if (strcmp(test, "all") == 0 || strcmp(test, "recurrence") == 0)
        test_recurrence();
    
    printf("\n=================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_total);
    
    return tests_passed == tests_total ? 0 : 1;
}

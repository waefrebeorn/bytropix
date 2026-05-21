/* test_chunked_ssm.c — verify chunked SSM matches sequential SSM */
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define d 128
#define hk 16
#define hv 32
#define rf 2

int main() {
    int T = 128;  // > 64 to trigger chunked path
    int B = 1;
    int N = B * T;

    // Allocate inputs
    float *q_norm = (float *)malloc(N * hk * d * sizeof(float));
    float *k_norm = (float *)malloc(N * hk * d * sizeof(float));
    float *v_conv = (float *)malloc(N * hv * d * sizeof(float));
    float *beta_flat = (float *)malloc(N * hv * sizeof(float));
    float *gate_flat = (float *)malloc(N * hv * sizeof(float));
    float *state_seq = (float *)calloc(hv * d * d, sizeof(float));
    float *state_chk = (float *)calloc(hv * d * d, sizeof(float));
    float *out_seq = (float *)calloc(N * hv * d, sizeof(float));
    float *out_chk = (float *)calloc(N * hv * d, sizeof(float));

    if (!q_norm || !k_norm || !v_conv || !beta_flat || !gate_flat ||
        !state_seq || !state_chk || !out_seq || !out_chk) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    // Fill with random-ish data
    srand(42);
    for (int i = 0; i < N * hk * d; i++) q_norm[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * hk * d; i++) k_norm[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * hv * d; i++) v_conv[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * hv; i++) {
        beta_flat[i] = (float)rand() / RAND_MAX * 0.1f;  // small beta
        gate_flat[i] = (float)rand() / RAND_MAX * 0.01f; // small gate (stable recurrence)
    }
    // Initialize state with small values
    for (int i = 0; i < hv * d * d; i++) {
        state_seq[i] = (float)rand() / RAND_MAX * 0.001f;
        state_chk[i] = state_seq[i];
    }

    // Run sequential recurrence
    // Copy the sequential code inline:
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * hv;
            float *gate_s = gate_flat + s * hv;
            for (int vh = 0; vh < hv; vh++) {
                int kh = vh % hk;  // cyclic repeat
                float bg = beta_s[vh];
                float gg = expf(fminf(gate_s[vh], 80.0f));
                float q_scaled[d];
                const float q_scale = 1.0f / sqrtf((float)d);
                for (int i = 0; i < d; i++)
                    q_scaled[i] = q_norm[(s * hk + kh) * d + i] * q_scale;
                float *h = state_seq + vh * d * d;
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        h[i * d + j] *= gg;
                float hk_v[d]; memset(hk_v, 0, sizeof(hk_v));
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        hk_v[i] += h[i * d + j] * k_norm[(s * hk + kh) * d + j];
                for (int i = 0; i < d; i++) {
                    float diff = v_conv[(s * hv + vh) * d + i] - hk_v[i];
                    for (int j = 0; j < d; j++)
                        h[i * d + j] += k_norm[(s * hk + kh) * d + j] * diff * bg;
                }
                float *out = out_seq + (s * hv + vh) * d;
                memset(out, 0, d * sizeof(float));
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        out[i] += h[i * d + j] * q_scaled[j];
            }
        }
    }

    // Run chunked
    wubu_ssm_chunked_recurrence(B, T, q_norm, k_norm, v_conv,
                                beta_flat, gate_flat, state_chk, out_chk);

    // Compare
    int max_errs = 10, errs = 0;
    double max_diff_out = 0, max_diff_state = 0;
    for (int i = 0; i < N * hv * d; i++) {
        double diff = fabs(out_seq[i] - out_chk[i]);
        if (diff > max_diff_out) max_diff_out = diff;
        if (diff > 1e-4 && errs < max_errs) {
            printf("  out diff at %d: seq=%f chk=%f diff=%e\n", i, out_seq[i], out_chk[i], diff);
            errs++;
        }
    }
    for (int i = 0; i < hv * d * d; i++) {
        double diff = fabs(state_seq[i] - state_chk[i]);
        if (diff > max_diff_state) max_diff_state = diff;
        if (diff > 1e-4 && errs < max_errs) {
            printf("  state diff at %d: seq=%f chk=%f diff=%e\n", i, state_seq[i], state_chk[i], diff);
            errs++;
        }
    }

    printf("Output max diff: %e\n", max_diff_out);
    printf("State max diff:  %e\n", max_diff_state);
    printf("State seq[0:3]: %.8f %.8f %.8f\n", state_seq[0], state_seq[1], state_seq[2]);
    printf("State chk[0:3]: %.8f %.8f %.8f\n", state_chk[0], state_chk[1], state_chk[2]);

    int pass = (max_diff_out < 1e-3 && max_diff_state < 1e-3);
    printf("%s\n", pass ? "PASS" : "FAIL");

    free(q_norm); free(k_norm); free(v_conv);
    free(beta_flat); free(gate_flat);
    free(state_seq); free(state_chk);
    free(out_seq); free(out_chk);
    return pass ? 0 : 1;
}

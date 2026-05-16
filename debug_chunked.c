#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void init_test_data(int T, int seed,
                           float *q_norm, float *k_norm, float *v_conv,
                           float *beta_flat, float *gate_flat) {
    srand(seed);
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    const int n_kh = SSM_K_HEADS;
    const int N = T;
    
    for (int i = 0; i < N * n_kh * d; i++) q_norm[i] = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < N * n_kh * d; i++) k_norm[i] = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < N * n_vh * d; i++) v_conv[i] = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < N * DT_RANK; i++) {
        beta_flat[i] = ((float)rand() / RAND_MAX) * 1.0f;
        gate_flat[i] = 2.0f * ((float)rand() / RAND_MAX - 0.5f) * 3.0f;  // [-3, 3]
    }
}

int main() {
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    const int n_kh = SSM_K_HEADS;
    const int B = 1;
    const int state_sz = n_vh * d * d;
    const int T = 65;
    const int N = T;
    const int seed = 43;

    printf("=== Debug: T=%d, seed=%d ===\n", T, seed);
    
    float *q_norm = (float *)malloc(N * n_kh * d * sizeof(float));
    float *k_norm = (float *)malloc(N * n_kh * d * sizeof(float));
    float *v_conv = (float *)malloc(N * n_vh * d * sizeof(float));
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    
    float *state_seq = (float *)calloc(state_sz, sizeof(float));
    float *state_chk = (float *)calloc(state_sz, sizeof(float));
    float *out_seq = (float *)calloc(N * n_vh * d, sizeof(float));
    float *out_chk = (float *)calloc(N * n_vh * d, sizeof(float));

    init_test_data(T, seed, q_norm, k_norm, v_conv, beta_flat, gate_flat);

    // Same state init
    srand(seed + 1000);
    for (int i = 0; i < state_sz; i++) {
        float v = 0.2f * ((float)rand() / RAND_MAX - 0.5f);
        state_seq[i] = v;
        state_chk[i] = v;
    }

    printf("State init: state[0]=%.10f state[100]=%.10f\n", state_seq[0], state_seq[100]);

    // Run sequential
    wubu_ssm_sequential_recurrence(B, T, q_norm, k_norm, v_conv,
                                    beta_flat, gate_flat, state_seq, out_seq);
    printf("After sequential state[0]=%.10f state[100]=%.10f\n", state_seq[0], state_seq[100]);
    printf("Output seq[0*4096+0]=%.10f seq[64*4096+0]=%.10f\n",
           out_seq[0], out_seq[64 * n_vh * d]);

    // Run chunked
    wubu_ssm_chunked_recurrence(B, T, q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat, state_chk, out_chk);
    printf("After chunked state[0]=%.10f state[100]=%.10f\n", state_chk[0], state_chk[100]);
    printf("Output chk[0*4096+0]=%.10f chk[64*4096+0]=%.10f\n",
           out_chk[0], out_chk[64 * n_vh * d]);

    // Compare all outputs
    float max_err = 0;
    int max_idx = 0;
    for (int i = 0; i < N * n_vh * d; i++) {
        float err = fabsf(out_seq[i] - out_chk[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("Output max abs diff: %e at idx %d (token=%d, head=%d, dim=%d)\n",
           max_err, max_idx, max_idx / (n_vh * d), (max_idx / d) % n_vh, max_idx % d);

    float max_state_err = 0;
    int max_state_idx = 0;
    for (int i = 0; i < state_sz; i++) {
        float err = fabsf(state_seq[i] - state_chk[i]);
        if (err > max_state_err) { max_state_err = err; max_state_idx = i; }
    }
    printf("State max abs diff: %e at idx %d\n", max_state_err, max_state_idx);

    // Print per-token max diff
    printf("\nPer-token output max diff:\n");
    for (int t = 0; t < T; t++) {
        float mt = 0;
        for (int i = 0; i < n_vh * d; i++) {
            float err = fabsf(out_seq[t * n_vh * d + i] - out_chk[t * n_vh * d + i]);
            if (err > mt) mt = err;
        }
        printf("  t=%d: max_diff=%e", t, mt);
        if (mt > 1e-5f) printf(" ***");
        printf("\n");
    }

    // Compare first 5 output values for seq vs chunked at each token
    printf("\nFirst 5 output values for tokens 0, 63, 64:\n");
    int debug_tokens[] = {0, 63, 64};
    for (int ti = 0; ti < 3; ti++) {
        int t = debug_tokens[ti];
        printf("  Token %d:\n", t);
        for (int i = 0; i < 5; i++) {
            printf("    [%d] seq=%.10f chk=%.10f\n", 
                   i, out_seq[t * n_vh * d + i], out_chk[t * n_vh * d + i]);
        }
    }

    free(q_norm); free(k_norm); free(v_conv);
    free(beta_flat); free(gate_flat);
    free(state_seq); free(state_chk);
    free(out_seq); free(out_chk);

    return 0;
}

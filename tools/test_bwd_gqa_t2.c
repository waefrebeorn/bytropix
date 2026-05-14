// test_bwd_gqa_t2.c — T=2 GQA backward vs T=1 baseline
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_ssm.h"

static double compute_loss(int B, int T, const float *Q, const float *K,
                           const float *V, const float *d_out) {
    const int hd = GQA_HEAD_DIM;
    const int n_q = GQA_Q_HEADS;
    const int n_kv = GQA_KV_HEADS;
    int q_per_kv = n_q / n_kv;
    const float scale = 1.0f / sqrtf((float)hd);
    double loss = 0.0;
    for (int b = 0; b < B; b++)
        for (int tq = 0; tq < T; tq++)
            for (int hq = 0; hq < n_q; hq++) {
                int hkv = hq / q_per_kv;
                const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                float maxs = -1e30f;
                int max_t = tq + 1;
                float score[4];  // T <= 4 for this test
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                    double ss = 0.0;
                    for (int i = 0; i < hd; i++) ss += (double)qv[i] * (double)kv[i];
                    score[tk] = (float)(ss / sqrtf((float)hd));
                    if (score[tk] > maxs) maxs = score[tk];
                }
                double sumexp = 0.0;
                for (int tk = 0; tk < max_t; tk++) { score[tk] = expf(score[tk]-maxs); sumexp += score[tk]; }
                float inv = 1.0f/(float)sumexp;
                for (int tk = 0; tk < max_t; tk++) score[tk] *= inv;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                    float a = score[tk];
                    for (int i = 0; i < hd; i++)
                        loss += (double)(a * vv[i]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + i];
                }
            }
    return loss;
}

static void test_T(int T, int seed) {
    const int B = 1;
    const int hd = GQA_HEAD_DIM;
    const int n_q = GQA_Q_HEADS;
    const int n_kv = GQA_KV_HEADS;
    const int N = B * T;
    
    srand(seed);
    float *Q = (float *)malloc(N * n_q * hd * sizeof(float));
    float *K = (float *)malloc(N * n_kv * hd * sizeof(float));
    float *V = (float *)malloc(N * n_kv * hd * sizeof(float));
    float *d_out = (float *)malloc(N * n_q * hd * sizeof(float));
    
    for (int i = 0; i < N * n_q * hd; i++) Q[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i = 0; i < N * n_kv * hd; i++) { K[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; V[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; }
    for (int i = 0; i < N * n_q * hd; i++) d_out[i] = ((float)rand()/RAND_MAX-0.5f)*0.01f;
    
    float *dQ_ana = (float *)calloc(N * n_q * hd, sizeof(float));
    float *dK_ana = (float *)calloc(N * n_kv * hd, sizeof(float));
    float *dV_ana = (float *)calloc(N * n_kv * hd, sizeof(float));
    wubu_gqa_backward_attention(B, T, Q, K, V, d_out, dQ_ana, dK_ana, dV_ana);
    
    float eps = 1e-5f;
    
    int q_ok = 0, q_chk = 60;
    for (int ci = 0; ci < q_chk; ci++) {
        int idx = rand() % (N * n_q * hd);
        float orig = Q[idx];
        Q[idx] = orig + eps; double lup = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig - eps; double ldn = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dQ_ana[idx] - fd) / (fmaxf(fabsf(dQ_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) q_ok++;
    }
    
    int k_ok = 0, k_chk = 60;
    for (int ci = 0; ci < k_chk; ci++) {
        int idx = rand() % (N * n_kv * hd);
        float orig = K[idx];
        K[idx] = orig + eps; double lup = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig - eps; double ldn = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dK_ana[idx] - fd) / (fmaxf(fabsf(dK_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) k_ok++;
    }
    
    int v_ok = 0, v_chk = 60;
    for (int ci = 0; ci < v_chk; ci++) {
        int idx = rand() % (N * n_kv * hd);
        float orig = V[idx];
        V[idx] = orig + eps; double lup = compute_loss(B, T, Q, K, V, d_out);
        V[idx] = orig - eps; double ldn = compute_loss(B, T, Q, K, V, d_out);
        V[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dV_ana[idx] - fd) / (fmaxf(fabsf(dV_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) v_ok++;
    }
    
    printf("T=%d: dQ %d/%d  dK %d/%d  dV %d/%d\n", T, q_ok, q_chk, k_ok, k_chk, v_ok, v_chk);
    
    free(Q); free(K); free(V); free(d_out);
    free(dQ_ana); free(dK_ana); free(dV_ana);
}

int main(void) {
    printf("=== GQA Backward: Causal accuracy vs T ===\n");
    test_T(1, 42);
    test_T(2, 42);
    test_T(3, 42);
    test_T(4, 42);
    return 0;
}

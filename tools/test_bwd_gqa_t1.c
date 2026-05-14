// test_bwd_gqa_t1.c — GQA backward with T=1 (no causal, single step)
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
                float scores[4096];
                int max_t = tq + 1;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                    double ss = 0.0;
                    for (int i = 0; i < hd; i++) ss += (double)qv[i] * (double)kv[i];
                    scores[tk] = (float)(ss * scale);
                    if (scores[tk] > maxs) maxs = scores[tk];
                }
                double sumexp = 0.0;
                for (int tk = 0; tk < max_t; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                float inv = 1.0f/(float)sumexp;
                for (int tk = 0; tk < max_t; tk++) scores[tk] *= inv;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                    float a = scores[tk];
                    for (int i = 0; i < hd; i++)
                        loss += (double)(a * vv[i]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + i];
                }
            }
    return loss;
}

int main(void) {
    srand(42);
    const int B = 1, T = 1;  // <--- SINGLE TIMESTEP
    const int hd = GQA_HEAD_DIM;
    const int n_q = GQA_Q_HEADS;
    const int n_kv = GQA_KV_HEADS;
    const int N = B * T;
    
    float *Q = (float *)malloc(N * n_q * hd * sizeof(float));
    float *K = (float *)malloc(N * n_kv * hd * sizeof(float));
    float *V = (float *)malloc(N * n_kv * hd * sizeof(float));
    float *d_out = (float *)malloc(N * n_q * hd * sizeof(float));
    
    // Small deterministic values
    for (int i = 0; i < N * n_q * hd; i++) Q[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i = 0; i < N * n_kv * hd; i++) { K[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; V[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; }
    for (int i = 0; i < N * n_q * hd; i++) d_out[i] = ((float)rand()/RAND_MAX-0.5f)*0.01f;
    
    // Analytic
    float *dQ_ana = (float *)calloc(N * n_q * hd, sizeof(float));
    float *dK_ana = (float *)calloc(N * n_kv * hd, sizeof(float));
    float *dV_ana = (float *)calloc(N * n_kv * hd, sizeof(float));
    wubu_gqa_backward_attention(B, T, Q, K, V, d_out, dQ_ana, dK_ana, dV_ana);
    
    printf("=== T=1 GQA Attention Backward ===\n");
    printf("B=%d T=%d N=%d n_q=%d n_kv=%d hd=%d\n\n", B, T, N, n_q, n_kv, hd);
    
    float eps = 1e-5f;
    
    // ===== dQ =====
    printf("dQ FD check:\n");
    int q_ok = 0;
    for (int ci = 0; ci < 20; ci++) {
        int idx = rand() % (N * n_q * hd);
        float orig = Q[idx];
        Q[idx] = orig + eps;
        double lup = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig - eps;
        double ldn = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dQ_ana[idx] - fd) / (fmaxf(fabsf(dQ_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) q_ok++;
        else printf("  FAIL idx=%d: ana=%.4e fd=%.4e rel=%.2f\n", idx, dQ_ana[idx], fd, rel);
    }
    printf("dQ: %d/20 pass\n\n", q_ok);
    
    // ===== dK =====
    printf("dK FD check:\n");
    int k_ok = 0;
    for (int ci = 0; ci < 20; ci++) {
        int idx = rand() % (N * n_kv * hd);
        float orig = K[idx];
        K[idx] = orig + eps;
        double lup = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig - eps;
        double ldn = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dK_ana[idx] - fd) / (fmaxf(fabsf(dK_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) k_ok++;
        else printf("  FAIL idx=%d: ana=%.4e fd=%.4e rel=%.2f\n", idx, dK_ana[idx], fd, rel);
    }
    printf("dK: %d/20 pass\n\n", k_ok);
    
    // ===== dV =====
    printf("dV FD check:\n");
    int v_ok = 0;
    for (int ci = 0; ci < 20; ci++) {
        int idx = rand() % (N * n_kv * hd);
        float orig = V[idx];
        V[idx] = orig + eps;
        double lup = compute_loss(B, T, Q, K, V, d_out);
        V[idx] = orig - eps;
        double ldn = compute_loss(B, T, Q, K, V, d_out);
        V[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dV_ana[idx] - fd) / (fmaxf(fabsf(dV_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) v_ok++;
        else printf("  FAIL idx=%d: ana=%.4e fd=%.4e rel=%.2f\n", idx, dV_ana[idx], fd, rel);
    }
    printf("dV: %d/20 pass\n", v_ok);
    
    free(Q); free(K); free(V); free(d_out);
    free(dQ_ana); free(dK_ana); free(dV_ana);
    return 0;
}

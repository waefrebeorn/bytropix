// test_bwd_gqa_standalone.c — Standalone GQA backward to find the bug
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HD 256
#define N_Q 16
#define N_KV 2
#define Q_PER_KV (N_Q / N_KV)

static double compute_loss(int B, int T, const float *Q, const float *K,
                           const float *V, const float *d_out) {
    const float scale = 1.0f / sqrtf((float)HD);
    double loss = 0.0;
    for (int b = 0; b < B; b++)
        for (int tq = 0; tq < T; tq++)
            for (int hq = 0; hq < N_Q; hq++) {
                int hkv = hq / Q_PER_KV;
                const float *qv = Q + ((b*T+tq)*N_Q + hq) * HD;
                float maxs = -1e30f;
                float score[8];
                int max_t = tq + 1;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*N_KV + hkv) * HD;
                    double ss = 0.0;
                    for (int i = 0; i < HD; i++) ss += (double)qv[i] * (double)kv[i];
                    score[tk] = (float)(ss * scale);
                    if (score[tk] > maxs) maxs = score[tk];
                }
                double sumexp = 0.0;
                for (int tk = 0; tk < max_t; tk++) { score[tk] = expf(score[tk]-maxs); sumexp += score[tk]; }
                float inv = 1.0f/(float)sumexp;
                for (int tk = 0; tk < max_t; tk++) score[tk] *= inv;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*N_KV + hkv) * HD;
                    float a = score[tk];
                    for (int i = 0; i < HD; i++)
                        loss += (double)(a * vv[i]) * (double)d_out[((b*T+tq)*N_Q + hq)*HD + i];
                }
            }
    return loss;
}

// Standalone backward, same logic as wubu_gqa_backward_attention
static void my_gqa_bwd(int B, int T, const float *Q, const float *K, const float *V,
                       const float *d_out, float *dQ, float *dK, float *dV) {
    const float scale = 1.0f / sqrtf((float)HD);
    for (int b = 0; b < B; b++) {
        for (int tq = 0; tq < T; tq++) {
            for (int hq = 0; hq < N_Q; hq++) {
                int hkv = hq / Q_PER_KV;
                const float *qv = Q + ((b*T+tq)*N_Q + hq) * HD;
                const float *do_ = d_out + ((b*T+tq)*N_Q + hq) * HD;
                float *dq = dQ + ((b*T+tq)*N_Q + hq) * HD;
                int max_t = tq + 1;
                float score[8], d_score[8];
                float maxv = -1e30f;
                
                // Scores
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*N_KV + hkv) * HD;
                    double s = 0.0;
                    for (int i = 0; i < HD; i++) s += (double)qv[i] * (double)kv[i];
                    score[tk] = (float)(s * scale);
                    if (score[tk] > maxv) maxv = score[tk];
                }
                // Softmax
                double sumexp = 0.0;
                for (int tk = 0; tk < max_t; tk++) { score[tk] = expf(score[tk] - maxv); sumexp += score[tk]; }
                float inv = 1.0f / (float)sumexp;
                for (int tk = 0; tk < max_t; tk++) score[tk] *= inv;
                
                // dV
                for (int tk = 0; tk < max_t; tk++) {
                    float *dv = dV + ((b*T+tk)*N_KV + hkv) * HD;
                    float a = score[tk];
                    for (int i = 0; i < HD; i++) dv[i] += do_[i] * a;
                }
                // d_score
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*N_KV + hkv) * HD;
                    double ds = 0.0;
                    for (int i = 0; i < HD; i++) ds += (double)do_[i] * (double)vv[i];
                    d_score[tk] = (float)ds;
                }
                // Softmax backward
                double dot = 0.0;
                for (int j = 0; j < max_t; j++) dot += (double)d_score[j] * (double)score[j];
                float d_logit[8];
                for (int tk = 0; tk < max_t; tk++) d_logit[tk] = score[tk] * (d_score[tk] - (float)dot);
                
                // dQ
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*N_KV + hkv) * HD;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < HD; i++) dq[i] += dl * kv[i];
                }
                // dK
                for (int tk = 0; tk < max_t; tk++) {
                    float *dk = dK + ((b*T+tk)*N_KV + hkv) * HD;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < HD; i++) dk[i] += dl * qv[i];
                }
            }
        }
    }
}

static int test_standalone(int T, int seed) {
    const int B = 1, N = B * T;
    srand(seed);
    float *Q = (float *)calloc(N * N_Q * HD, 4);
    float *K = (float *)calloc(N * N_KV * HD, 4);
    float *V = (float *)calloc(N * N_KV * HD, 4);
    float *d_out = (float *)calloc(N * N_Q * HD, 4);
    for (int i = 0; i < N * N_Q * HD; i++) Q[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i = 0; i < N * N_KV * HD; i++) { K[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; V[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; }
    for (int i = 0; i < N * N_Q * HD; i++) d_out[i] = ((float)rand()/RAND_MAX-0.5f)*0.01f;
    
    float *dQ = (float *)calloc(N * N_Q * HD, 4);
    float *dK = (float *)calloc(N * N_KV * HD, 4);
    float *dV = (float *)calloc(N * N_KV * HD, 4);
    my_gqa_bwd(B, T, Q, K, V, d_out, dQ, dK, dV);
    
    float eps = 1e-5f;
    int q_ok = 0, q_chk = 60;
    for (int ci = 0; ci < q_chk; ci++) {
        int idx = rand() % (N * N_Q * HD);
        float orig = Q[idx]; Q[idx] = orig + eps; double lup = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig - eps; double ldn = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dQ[idx] - fd) / (fmaxf(fabsf(dQ[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) q_ok++;
    }
    int k_ok = 0, k_chk = 60;
    for (int ci = 0; ci < k_chk; ci++) {
        int idx = rand() % (N * N_KV * HD);
        float orig = K[idx]; K[idx] = orig + eps; double lup = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig - eps; double ldn = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dK[idx] - fd) / (fmaxf(fabsf(dK[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) k_ok++;
    }
    int v_ok = 0, v_chk = 60;
    for (int ci = 0; ci < v_chk; ci++) {
        int idx = rand() % (N * N_KV * HD);
        float orig = V[idx]; V[idx] = orig + eps; double lup = compute_loss(B, T, Q, K, V, d_out);
        V[idx] = orig - eps; double ldn = compute_loss(B, T, Q, K, V, d_out);
        V[idx] = orig;
        float fd = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dV[idx] - fd) / (fmaxf(fabsf(dV[idx]), fabsf(fd)) + 1e-10f);
        if (rel < 0.3f) v_ok++;
    }
    
    printf("T=%d (standalone): dQ %d/%d  dK %d/%d  dV %d/%d\n", T, q_ok, q_chk, k_ok, k_chk, v_ok, v_chk);
    free(Q); free(K); free(V); free(d_out);
    free(dQ); free(dK); free(dV);
    return (q_ok == q_chk && k_ok == k_chk) ? 1 : 0;
}

int main(void) {
    printf("=== Standalone GQA Backward (no wubu_ssm.o) ===\n");
    test_standalone(1, 42);
    test_standalone(2, 42);
    test_standalone(3, 42);
    return 0;
}

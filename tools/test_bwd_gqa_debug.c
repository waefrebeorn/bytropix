// test_bwd_gqa_debug.c — Diagnose dQ/dK mismatch with tiny single-head test
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_ssm.h"

// Simplified forward: loss = sum(softmax(Q@K^T/sqrt(d)) @ V * d_out)
static double compute_loss(int B, int T, const float *Q, const float *K, 
                           const float *V, const float *d_out) {
    const int hd = GQA_HEAD_DIM;
    const int n_q = GQA_Q_HEADS;
    const int n_kv = GQA_KV_HEADS;
    int q_per_kv = n_q / n_kv;
    const float scale = 1.0f / sqrtf((float)hd);
    
    double loss = 0.0;
    for (int b = 0; b < B; b++) {
        for (int tq = 0; tq < T; tq++) {
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
        }
    }
    return loss;
}

int main(void) {
    srand(42);
    const int B = 1, T = 3;
    const int hd = GQA_HEAD_DIM;  // 128
    const int n_q = GQA_Q_HEADS;  // 32
    const int n_kv = GQA_KV_HEADS; // 4
    const int q_per_kv = n_q / n_kv; // 8
    const int N = B * T;
    const float scale = 1.0f / sqrtf((float)hd);
    
    float *Q = (float *)malloc(N * n_q * hd * sizeof(float));
    float *K = (float *)malloc(N * n_kv * hd * sizeof(float));
    float *V = (float *)malloc(N * n_kv * hd * sizeof(float));
    float *d_out = (float *)malloc(N * n_q * hd * sizeof(float));
    
    for (int i = 0; i < N * n_q * hd; i++) Q[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i = 0; i < N * n_kv * hd; i++) { K[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; V[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; }
    for (int i = 0; i < N * n_q * hd; i++) d_out[i] = ((float)rand()/RAND_MAX-0.5f)*0.01f;
    
    // Analytic backward
    float *dQ_ana = (float *)calloc(N * n_q * hd, sizeof(float));
    float *dK_ana = (float *)calloc(N * n_kv * hd, sizeof(float));
    float *dV_ana = (float *)calloc(N * n_kv * hd, sizeof(float));
    
    wubu_gqa_backward_attention(B, T, Q, K, V, d_out, dQ_ana, dK_ana, dV_ana);
    
    // Pick a specific Q element to debug: s=2, hq=4, i=49 (from known failure)
    int test_q_idx = 2 * n_q * hd + 4 * hd + 49;  // s=2, hq=4, i=49
    if (test_q_idx >= N * n_q * hd) test_q_idx = 1 * n_q * hd + 4 * hd + 49;
    
    // Debug: print all intermediate values for this query head
    int s_test = test_q_idx / (n_q * hd);
    int hq_test = (test_q_idx / hd) % n_q;
    int i_test = test_q_idx % hd;
    int hkv_test = hq_test / q_per_kv;
    
    printf("=== Debug single Q element ===\n");
    printf("s=%d hq=%d hkv=%d i=%d scale=%.6f\n", s_test, hq_test, hkv_test, i_test, scale);
    printf("N_q=%d N_kv=%d q_per_kv=%d hd=%d\n\n", n_q, n_kv, q_per_kv, hd);
    
    // Manual forward for this query head
    const float *qv = Q + (s_test * n_q + hq_test) * hd;
    int max_t = s_test + 1;  // t_q = s_test
    printf("q_vec[0..4]: %.6f %.6f %.6f %.6f %.6f\n", qv[0], qv[1], qv[2], qv[3], qv[4]);
    
    float scores[4096], scores_raw[4096];
    float max_score = -1e30f;
    for (int tk = 0; tk < max_t; tk++) {
        const float *kv = K + (s_test * n_kv + hkv_test) * hd + tk * n_kv * hd;
        double s = 0.0;
        for (int i = 0; i < hd; i++) s += (double)qv[i] * (double)kv[i];
        scores_raw[tk] = (float)(s * scale);
        scores[tk] = scores_raw[tk];
        if (scores[tk] > max_score) max_score = scores[tk];
        printf("  score_raw[tk=%d]=%.6f (k[0..2]=%.4f %.4f %.4f)\n", 
               tk, scores_raw[tk], kv[0], kv[1], kv[2]);
    }
    
    double sum_exp = 0.0;
    for (int tk = 0; tk < max_t; tk++) { scores[tk] = expf(scores[tk] - max_score); sum_exp += scores[tk]; }
    float inv_sum = 1.0f / (float)sum_exp;
    for (int tk = 0; tk < max_t; tk++) scores[tk] *= inv_sum;
    
    printf("softmax: ");
    for (int tk = 0; tk < max_t; tk++) printf("%.6f ", scores[tk]);
    printf("\n");
    
    // d_score
    float d_score[4096];
    for (int tk = 0; tk < max_t; tk++) {
        const float *vv = V + (s_test * n_kv + hkv_test) * hd + tk * n_kv * hd;
        const float *do_ = d_out + (s_test * n_q + hq_test) * hd;
        double ds = 0.0;
        for (int i = 0; i < hd; i++) ds += (double)do_[i] * (double)vv[i];
        d_score[tk] = (float)ds;
        printf("  d_score[tk=%d] = %.10f\n", tk, d_score[tk]);
    }
    
    // Softmax backward
    double dot = 0.0;
    for (int j = 0; j < max_t; j++) dot += (double)d_score[j] * (double)scores[j];
    printf("  dot(sum d_score*score) = %.10f\n", dot);
    
    float d_logit[4096];
    for (int tk = 0; tk < max_t; tk++) {
        d_logit[tk] = scores[tk] * (d_score[tk] - (float)dot);
        printf("  d_logit[tk=%d] = %.10f\n", tk, d_logit[tk]);
    }
    
    // dQ manual
    float dq_manual[hd];
    memset(dq_manual, 0, sizeof(dq_manual));
    for (int tk = 0; tk < max_t; tk++) {
        const float *kv = K + (s_test * n_kv + hkv_test) * hd + tk * n_kv * hd;
        float dl = d_logit[tk] * scale;
        for (int i = 0; i < hd; i++) dq_manual[i] += dl * kv[i];
    }
    
    printf("\ndQ[%d] (s=%d hq=%d i=%d): ana=%.10f manual=%.10f\n", 
           test_q_idx, s_test, hq_test, i_test, 
           dQ_ana[test_q_idx], dq_manual[i_test]);
    
    // FD
    float eps = 1e-5f;
    float orig = Q[test_q_idx];
    Q[test_q_idx] = orig + eps;
    double loss_up = compute_loss(B, T, Q, K, V, d_out);
    Q[test_q_idx] = orig - eps;
    double loss_down = compute_loss(B, T, Q, K, V, d_out);
    Q[test_q_idx] = orig;
    float fd = (float)((loss_up - loss_down) / (2.0 * eps));
    
    printf("FD = %.10f (loss_up=%.10f, loss_down=%.10f, eps=%g)\n", fd, loss_up, loss_down, eps);
    printf("dQ_ana/FD ratio: %.4f\n", dQ_ana[test_q_idx] / (fd + 1e-30f));
    
    // COMPARE ANA VS MANUAL for this head
    float max_diff = 0.0f, max_ana = 0.0f;
    int max_diff_i = 0;
    for (int i = 0; i < hd; i++) {
        int idx = (s_test * n_q + hq_test) * hd + i;
        float diff = fabsf(dQ_ana[idx] - dq_manual[i]);
        if (diff > max_diff) { max_diff = diff; max_diff_i = i; max_ana = dQ_ana[idx]; }
    }
    printf("ANA vs manual max diff at i=%d: diff=%.10f ana=%.10f manual=%.10f\n",
           max_diff_i, max_diff, max_ana, dq_manual[max_diff_i]);
    
    // Check: does d_logit match between ANA and manual computation?
    printf("\n=== Extended FD check ===\n");
    int n_ok_q = 0, n_ok_k = 0, n_ok_v = 0;
    for (int ci = 0; ci < 10; ci++) {
        int idx = rand() % (N * n_q * hd);
        float orig_q = Q[idx];
        Q[idx] = orig_q + eps;
        double lup = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig_q - eps;
        double ldn = compute_loss(B, T, Q, K, V, d_out);
        Q[idx] = orig_q;
        float fd_q = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dQ_ana[idx] - fd_q) / (fmaxf(fabsf(dQ_ana[idx]), fabsf(fd_q)) + 1e-10f);
        if (rel < 0.3f) n_ok_q++; else printf("  Q FAIL idx=%d: ana=%.2e fd=%.2e rel=%.2f\n", idx, dQ_ana[idx], fd_q, rel);
    }
    printf("dQ: %d/10 pass\n", n_ok_q);
    
    // Repeat dK check with more detail
    printf("\n=== dK detail ===\n");
    for (int ci = 0; ci < 5; ci++) {
        int idx = rand() % (N * n_kv * hd);
        float orig_k = K[idx];
        K[idx] = orig_k + eps;
        double lup = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig_k - eps;
        double ldn = compute_loss(B, T, Q, K, V, d_out);
        K[idx] = orig_k;
        float fd_k = (float)((lup - ldn) / (2.0 * eps));
        float rel = fabsf(dK_ana[idx] - fd_k) / (fmaxf(fabsf(dK_ana[idx]), fabsf(fd_k)) + 1e-10f);
        if (rel < 0.3f) n_ok_k++; else printf("  K FAIL idx=%d: ana=%.2e fd=%.2e rel=%.2f\n", idx, dK_ana[idx], fd_k, rel);
    }
    printf("dK: %d/5 pass\n", n_ok_k);
    
    free(Q); free(K); free(V); free(d_out);
    free(dQ_ana); free(dK_ana); free(dV_ana);
    return 0;
}

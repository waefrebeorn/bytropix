// test_bwd_gqa.c — Verify GQA attention backward with finite differences
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "wubu_ssm.h"

int main(void) {
    srand(42);
    const int B = 1, T = 3;
    const int hd = GQA_HEAD_DIM;  // 128
    const int n_q = GQA_Q_HEADS;  // 32
    const int n_kv = GQA_KV_HEADS; // 4
    const int N = B * T;
    
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
    
    // Forward: attn_out = softmax(Q@K^T/sqrt(d)) @ V
    // Loss = sum(attn_out * d_out)  (linear loss = gradient check)
    const float scale = 1.0f / sqrtf((float)hd);
    float eps = 1e-5f;
    
    printf("=== GQA Attention Backward Verification ===\n\n");
    
    // Check a few Q gradients
    int n_ok = 0, n_chk = 0;
    printf("dQ finite-difference check (eps=%g):\n", eps);
    for (int ci = 0; ci < 20; ci++) {
        int idx = rand() % (N * n_q * hd);
        int s = idx / (n_q * hd);
        int h_q = (idx / hd) % n_q;
        int i = idx % hd;
        int h_kv = h_q / (n_q / n_kv);
        
        float orig = Q[idx];
        Q[idx] = orig + eps;
        
        // Recompute forward for this change
        double loss_up = 0.0;
        for (int b = 0; b < B; b++) {
            for (int tq = 0; tq < T; tq++) {
                for (int hq = 0; hq < n_q; hq++) {
                    int hkv = hq / (n_q / n_kv);
                    const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                    
                    // Compute softmax scores
                    float maxs = -1e30f;
                    float scores[4096];
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                        double ss = 0.0;
                        for (int ii = 0; ii < hd; ii++) ss += (double)qv[ii] * (double)kv[ii];
                        scores[tk] = (float)(ss * scale);
                        if (scores[tk] > maxs) maxs = scores[tk];
                    }
                    double sumexp = 0.0;
                    for (int tk = 0; tk <= tq; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                    float inv = 1.0f/(float)sumexp;
                    for (int tk = 0; tk <= tq; tk++) scores[tk] *= inv;
                    
                    // Weighted sum
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                        float a = scores[tk];
                        for (int ii = 0; ii < hd; ii++)
                            loss_up += (double)(a * vv[ii]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + ii];
                    }
                }
            }
        }
        
        Q[idx] = orig - eps;
        double loss_down = 0.0;
        // ... (same forward recomputation)
        for (int b = 0; b < B; b++) {
            for (int tq = 0; tq < T; tq++) {
                for (int hq = 0; hq < n_q; hq++) {
                    int hkv = hq / (n_q / n_kv);
                    const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                    
                    float maxs = -1e30f;
                    float scores[4096];
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                        double ss = 0.0;
                        for (int ii = 0; ii < hd; ii++) ss += (double)qv[ii] * (double)kv[ii];
                        scores[tk] = (float)(ss * scale);
                        if (scores[tk] > maxs) maxs = scores[tk];
                    }
                    double sumexp = 0.0;
                    for (int tk = 0; tk <= tq; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                    float inv = 1.0f/(float)sumexp;
                    for (int tk = 0; tk <= tq; tk++) scores[tk] *= inv;
                    
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                        float a = scores[tk];
                        for (int ii = 0; ii < hd; ii++)
                            loss_down += (double)(a * vv[ii]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + ii];
                    }
                }
            }
        }
        
        Q[idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float rel_err = fabsf(dQ_ana[idx] - fd) / (fmaxf(fabsf(dQ_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.3f) n_ok++;
        else printf("  dQ[%d] s=%d hq=%d i=%d: ana=%g fd=%g rel_err=%.2f\n",
                    idx, s, h_q, i, dQ_ana[idx], fd, rel_err);
        n_chk++;
    }
    printf("dQ FD: %d/%d pass (rel_err<30%%)\n", n_ok, n_chk);
    
    n_ok = 0; n_chk = 0;
    for (int ci = 0; ci < 20; ci++) {
        int idx = rand() % (N * n_kv * hd);
        int s = idx / (n_kv * hd);
        int h_kv = (idx / hd) % n_kv;
        int i = idx % hd;
        
        float orig = K[idx];
        K[idx] = orig + eps;
        
        double loss_up = 0.0;
        for (int b = 0; b < B; b++) {
            for (int tq = 0; tq < T; tq++) {
                for (int hq = 0; hq < n_q; hq++) {
                    int hkv = hq / (n_q / n_kv);
                    const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                    
                    float maxs = -1e30f;
                    float scores[4096];
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                        double ss = 0.0;
                        for (int ii = 0; ii < hd; ii++) ss += (double)qv[ii] * (double)kv[ii];
                        scores[tk] = (float)(ss * scale);
                        if (scores[tk] > maxs) maxs = scores[tk];
                    }
                    double sumexp = 0.0;
                    for (int tk = 0; tk <= tq; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                    float inv = 1.0f/(float)sumexp;
                    for (int tk = 0; tk <= tq; tk++) scores[tk] *= inv;
                    
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                        float a = scores[tk];
                        for (int ii = 0; ii < hd; ii++)
                            loss_up += (double)(a * vv[ii]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + ii];
                    }
                }
            }
        }
        
        K[idx] = orig - eps;
        double loss_down = 0.0;
        for (int b = 0; b < B; b++) {
            for (int tq = 0; tq < T; tq++) {
                for (int hq = 0; hq < n_q; hq++) {
                    int hkv = hq / (n_q / n_kv);
                    const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                    
                    float maxs = -1e30f;
                    float scores[4096];
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                        double ss = 0.0;
                        for (int ii = 0; ii < hd; ii++) ss += (double)qv[ii] * (double)kv[ii];
                        scores[tk] = (float)(ss * scale);
                        if (scores[tk] > maxs) maxs = scores[tk];
                    }
                    double sumexp = 0.0;
                    for (int tk = 0; tk <= tq; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                    float inv = 1.0f/(float)sumexp;
                    for (int tk = 0; tk <= tq; tk++) scores[tk] *= inv;
                    
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                        float a = scores[tk];
                        for (int ii = 0; ii < hd; ii++)
                            loss_down += (double)(a * vv[ii]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + ii];
                    }
                }
            }
        }
        
        K[idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float rel_err = fabsf(dK_ana[idx] - fd) / (fmaxf(fabsf(dK_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.3f) n_ok++;
        n_chk++;
    }
    printf("dK FD: %d/%d pass (rel_err<30%%)\n", n_ok, n_chk);
    
    n_ok = 0; n_chk = 0;
    for (int ci = 0; ci < 20; ci++) {
        int idx = rand() % (N * n_kv * hd);
        float orig = V[idx];
        V[idx] = orig + eps;
        
        double loss_up = 0.0;
        for (int b = 0; b < B; b++) {
            for (int tq = 0; tq < T; tq++) {
                for (int hq = 0; hq < n_q; hq++) {
                    int hkv = hq / (n_q / n_kv);
                    const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                    
                    float maxs = -1e30f;
                    float scores[4096];
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                        double ss = 0.0;
                        for (int ii = 0; ii < hd; ii++) ss += (double)qv[ii] * (double)kv[ii];
                        scores[tk] = (float)(ss * scale);
                        if (scores[tk] > maxs) maxs = scores[tk];
                    }
                    double sumexp = 0.0;
                    for (int tk = 0; tk <= tq; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                    float inv = 1.0f/(float)sumexp;
                    for (int tk = 0; tk <= tq; tk++) scores[tk] *= inv;
                    
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                        float a = scores[tk];
                        for (int ii = 0; ii < hd; ii++)
                            loss_up += (double)(a * vv[ii]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + ii];
                    }
                }
            }
        }
        
        V[idx] = orig - eps;
        double loss_down = 0.0;
        for (int b = 0; b < B; b++) {
            for (int tq = 0; tq < T; tq++) {
                for (int hq = 0; hq < n_q; hq++) {
                    int hkv = hq / (n_q / n_kv);
                    const float *qv = Q + ((b*T+tq)*n_q + hq) * hd;
                    
                    float maxs = -1e30f;
                    float scores[4096];
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *kv = K + ((b*T+tk)*n_kv + hkv) * hd;
                        double ss = 0.0;
                        for (int ii = 0; ii < hd; ii++) ss += (double)qv[ii] * (double)kv[ii];
                        scores[tk] = (float)(ss * scale);
                        if (scores[tk] > maxs) maxs = scores[tk];
                    }
                    double sumexp = 0.0;
                    for (int tk = 0; tk <= tq; tk++) { scores[tk] = expf(scores[tk]-maxs); sumexp += scores[tk]; }
                    float inv = 1.0f/(float)sumexp;
                    for (int tk = 0; tk <= tq; tk++) scores[tk] *= inv;
                    
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *vv = V + ((b*T+tk)*n_kv + hkv) * hd;
                        float a = scores[tk];
                        for (int ii = 0; ii < hd; ii++)
                            loss_down += (double)(a * vv[ii]) * (double)d_out[((b*T+tq)*n_q + hq)*hd + ii];
                    }
                }
            }
        }
        
        V[idx] = orig;
        float fd = (float)((loss_up - loss_down) / (2.0 * eps));
        float rel_err = fabsf(dV_ana[idx] - fd) / (fmaxf(fabsf(dV_ana[idx]), fabsf(fd)) + 1e-10f);
        if (rel_err < 0.3f) n_ok++;
        n_chk++;
    }
    printf("dV FD: %d/%d pass (rel_err<30%%)\n", n_ok, n_chk);
    
    free(Q); free(K); free(V); free(d_out);
    free(dQ_ana); free(dK_ana); free(dV_ana);
    printf("\nDone.\n");
    return 0;
}

// test_bwd_gqa_trace.c — Trace every intermediate for one dK element
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HD 256
#define N_Q 16
#define N_KV 2
#define Q_PER_KV (N_Q / N_KV)

int main(void) {
    srand(42);
    const int B = 1, T = 2;
    const int N = B * T;
    const float scale = 1.0f / sqrtf((float)HD);
    
    float *Q = (float *)calloc(N * N_Q * HD, 4);
    float *K = (float *)calloc(N * N_KV * HD, 4);
    float *V = (float *)calloc(N * N_KV * HD, 4);
    float *d_out = (float *)calloc(N * N_Q * HD, 4);
    for (int i = 0; i < N * N_Q * HD; i++) Q[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i = 0; i < N * N_KV * HD; i++) { K[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; V[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; }
    for (int i = 0; i < N * N_Q * HD; i++) d_out[i] = ((float)rand()/RAND_MAX-0.5f)*0.01f;
    
    // Pick a specific dK element to trace: K[0, 0, 49] (tk=0, hkv=0, dim=49)
    int tk_trace = 0, hkv_trace = 0, i_trace = 49;
    int dk_idx = (tk_trace * N_KV + hkv_trace) * HD + i_trace;
    
    printf("=== Tracing dK[tk=%d, hkv=%d, i=%d] (idx=%d) ===\n\n", tk_trace, hkv_trace, i_trace, dk_idx);
    
    float eps = 1e-5f;
    
    // FD total gradient
    float orig = K[dk_idx];
    K[dk_idx] = orig + eps;
    double loss_up = 0.0, loss_down;
    // Must recompute total loss
    for (int b = 0; b < B; b++) {
        for (int tq = 0; tq < T; tq++) {
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
                        loss_up += (double)(a * vv[i]) * (double)d_out[((b*T+tq)*N_Q + hq)*HD + i];
                }
            }
        }
    }
    K[dk_idx] = orig - eps;
    loss_down = 0.0;
    for (int b = 0; b < B; b++) { /* same as above */
        for (int tq = 0; tq < T; tq++) {
            for (int hq = 0; hq < N_Q; hq++) {
                int hkv = hq / Q_PER_KV;
                const float *qv = Q + ((b*T+tq)*N_Q + hq) * HD;
                float maxs = -1e30f;
                float score[8]; int max_t = tq + 1;
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
                        loss_down += (double)(a * vv[i]) * (double)d_out[((b*T+tq)*N_Q + hq)*HD + i];
                }
            }
        }
    }
    K[dk_idx] = orig;
    float fd_total = (float)((loss_up - loss_down) / (2.0 * eps));
    
    printf("FD total gradient: %.10f\n\n", fd_total);
    
    // Now compute contribution PER QUERY POSITION
    // Store dK from each (t_q, h_q) pair
    float dK_manual[N][N_Q];  // dK contribution from each (t_q, h_q), for our specific dim
    memset(dK_manual, 0, sizeof(dK_manual));
    
    float ana_dK = 0.0f;
    
    for (int b = 0; b < B; b++) {
        for (int tq = 0; tq < T; tq++) {
            for (int hq = 0; hq < N_Q; hq++) {
                int hkv = hq / Q_PER_KV;
                if (hkv != hkv_trace) continue;  // only interested in our hkv
                
                const float *qv = Q + ((b*T+tq)*N_Q + hq) * HD;
                const float *do_ = d_out + ((b*T+tq)*N_Q + hq) * HD;
                int max_t = tq + 1;
                float score[8];
                float maxv = -1e30f;
                
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*N_KV + hkv) * HD;
                    double s = 0.0;
                    for (int i = 0; i < HD; i++) s += (double)qv[i] * (double)kv[i];
                    score[tk] = (float)(s * scale);
                    if (score[tk] > maxv) maxv = score[tk];
                }
                double sumexp = 0.0;
                for (int tk = 0; tk < max_t; tk++) { score[tk] = expf(score[tk]-maxv); sumexp += score[tk]; }
                float inv = 1.0f/(float)sumexp;
                for (int tk = 0; tk < max_t; tk++) score[tk] *= inv;
                
                // d_score
                float d_score[8];
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*N_KV + hkv) * HD;
                    double ds = 0.0;
                    for (int i = 0; i < HD; i++) ds += (double)do_[i] * (double)vv[i];
                    d_score[tk] = (float)ds;
                }
                
                double dot = 0.0;
                for (int j = 0; j < max_t; j++) dot += (double)d_score[j] * (double)score[j];
                
                float d_logit[8];
                for (int tk = 0; tk < max_t; tk++)
                    d_logit[tk] = score[tk] * (d_score[tk] - (float)dot);
                
                // dK contribution: d_logit[tk_trace] * scale * qv[i_trace]
                // ONLY if tk_trace < max_t (within causal mask)
                if (tk_trace < max_t) {
                    float contrib = d_logit[tk_trace] * scale * qv[i_trace];
                    dK_manual[tq][hq] = contrib;
                    ana_dK += contrib;
                    
                    if (hq < 2) {  // just print first couple heads for sanity
                        printf("  (tq=%d, hq=%d): score=[", tq, hq);
                        for (int tk = 0; tk < max_t; tk++) printf("%.4e", score[tk]);
                        printf("] d_score=[");
                        for (int tk = 0; tk < max_t; tk++) printf("%.4e", d_score[tk]);
                        printf("] d_logit[%d]=%.4e q[%d]=%.4e contrib=%.4e\n",
                               tk_trace, d_logit[tk_trace], i_trace, qv[i_trace], contrib);
                    }
                }
            }
        }
    }
    
    printf("\nSummed contributions per (tq, hq):\n");
    for (int tq = 0; tq < T; tq++)
        for (int hq = 0; hq < N_Q; hq++)
            if (dK_manual[tq][hq] != 0.0f)
                printf("  tq=%d hq=%d: contrib=%.6e\n", tq, hq, dK_manual[tq][hq]);
    
    printf("\nTotal ana dK = %.10f\n", ana_dK);
    printf("Total FD     = %.10f\n", fd_total);
    printf("Ratio ana/FD = %.4f\n", ana_dK / (fd_total + 1e-30f));
    
    free(Q); free(K); free(V); free(d_out);
    return 0;
}

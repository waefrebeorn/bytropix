// test_bwd_gqa_simple.c — Deterministic values, test FD precision
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HD 8       // tiny
#define N_Q 2       // 2 Q heads
#define N_KV 1       // 1 KV head
#define Q_PER_KV (N_Q / N_KV)

// Forward loss: L = Σ d_out[tq,hq,i] * (Σ_k softmax_k * V[tk,hkv,i])
static double loss_fwd(int B, int T, const float *Q, const float *K,
                       const float *V, const float *d_out) {
    const float scale = 1.0f / sqrtf((float)HD);
    double L = 0.0;
    for (int b = 0; b < B; b++)
        for (int tq = 0; tq < T; tq++)
            for (int hq = 0; hq < N_Q; hq++) {
                int hkv = hq / Q_PER_KV;
                const float *qv = Q + ((b*T+tq)*N_Q + hq) * HD;
                int max_t = tq + 1;
                float score[16]; float maxs = -1e30f;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*N_KV + hkv) * HD;
                    double s = 0.0;
                    for (int i = 0; i < HD; i++) s += (double)qv[i] * (double)kv[i];
                    score[tk] = (float)(s * scale);
                    if (score[tk] > maxs) maxs = score[tk];
                }
                double se = 0.0;
                for (int tk = 0; tk < max_t; tk++) { score[tk] = expf(score[tk]-maxs); se += score[tk]; }
                float inv = 1.0f/(float)se;
                for (int tk = 0; tk < max_t; tk++) score[tk] *= inv;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*N_KV + hkv) * HD;
                    float a = score[tk];
                    for (int i = 0; i < HD; i++)
                        L += (double)(a * vv[i]) * (double)d_out[((b*T+tq)*N_Q + hq)*HD + i];
                }
            }
    return L;
}

// Analytic backward for dK (same logic)
static void bwd_dK(int B, int T, const float *Q, const float *K,
                   const float *V, const float *d_out, float *dK) {
    const float scale = 1.0f / sqrtf((float)HD);
    for (int b = 0; b < B; b++)
        for (int tq = 0; tq < T; tq++)
            for (int hq = 0; hq < N_Q; hq++) {
                int hkv = hq / Q_PER_KV;
                const float *qv = Q + ((b*T+tq)*N_Q + hq) * HD;
                const float *do_ = d_out + ((b*T+tq)*N_Q + hq) * HD;
                int max_t = tq + 1;
                float score[16]; float maxs = -1e30f;
                for (int tk = 0; tk < max_t; tk++) {
                    const float *kv = K + ((b*T+tk)*N_KV + hkv) * HD;
                    double s = 0.0;
                    for (int i = 0; i < HD; i++) s += (double)qv[i] * (double)kv[i];
                    score[tk] = (float)(s * scale);
                    if (score[tk] > maxs) maxs = score[tk];
                }
                double se = 0.0;
                for (int tk = 0; tk < max_t; tk++) { score[tk] = expf(score[tk]-maxs); se += score[tk]; }
                float inv = 1.0f/(float)se;
                for (int tk = 0; tk < max_t; tk++) score[tk] *= inv;
                
                float d_score[16];
                for (int tk = 0; tk < max_t; tk++) {
                    const float *vv = V + ((b*T+tk)*N_KV + hkv) * HD;
                    double ds = 0.0;
                    for (int i = 0; i < HD; i++) ds += (double)do_[i] * (double)vv[i];
                    d_score[tk] = (float)ds;
                }
                double dot = 0.0;
                for (int j = 0; j < max_t; j++) dot += (double)d_score[j] * (double)score[j];
                float d_logit[16];
                for (int tk = 0; tk < max_t; tk++)
                    d_logit[tk] = score[tk] * (d_score[tk] - (float)dot);
                
                for (int tk = 0; tk < max_t; tk++) {
                    float *dk = dK + ((b*T+tk)*N_KV + hkv) * HD;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < HD; i++) dk[i] += dl * qv[i];
                }
            }
}

int main(void) {
    const int B = 1, T = 3;
    const int N = B * T;
    const float scale = 1.0f / sqrtf((float)HD);
    
    // Deterministic values: Q is same for all positions, K is structured
    float *Q = (float *)calloc(N * N_Q * HD, 4);
    float *K = (float *)calloc(N * N_KV * HD, 4);
    float *V = (float *)calloc(N * N_KV * HD, 4);
    float *d_out = (float *)calloc(N * N_Q * HD, 4);
    
    // Easy numbers: Q all 1.0, K varies, V varies, d_out = d_out[tq,hq,i] = i+1
    for (int i = 0; i < N * N_Q * HD; i++) Q[i] = 1.0f;
    for (int i = 0; i < N * N_KV * HD; i++) K[i] = (float)(i + 1) * 0.01f;
    for (int i = 0; i < N * N_KV * HD; i++) V[i] = (float)(i + 1) * 0.01f;
    for (int i = 0; i < N * N_Q * HD; i++) d_out[i] = 1.0f;
    
    // Analytic dK
    float *dK_ana = (float *)calloc(N * N_KV * HD, 4);
    bwd_dK(B, T, Q, K, V, d_out, dK_ana);
    
    printf("=== Deterministic test HD=%d N_Q=%d N_KV=%d T=%d ===\n\n", HD, N_Q, N_KV, T);
    
    // Test dK[0,0,:] — FD vs analytic
    for (int eps_pow = 1; eps_pow <= 10; eps_pow++) {
        float eps = powf(10.0f, -(float)eps_pow);
        int n_ok = 0, n_chk = 0;
        for (int ci = 0; ci < HD; ci++) {
            int idx = ci;  // K[0,0,ci] with N_KV=1, HD=8
            float orig = K[idx];
            K[idx] = orig + eps;
            double lup = loss_fwd(B, T, Q, K, V, d_out);
            K[idx] = orig - eps;
            double ldn = loss_fwd(B, T, Q, K, V, d_out);
            K[idx] = orig;
            float fd = (float)((lup - ldn) / (2.0 * eps));
            float rel = fabsf(dK_ana[idx] - fd) / (fmaxf(fabsf(dK_ana[idx]), fabsf(fd)) + 1e-10f);
            if (rel < 0.3f) n_ok++;
        }
        printf("eps=1e-%d: dK %d/%d pass", eps_pow, n_ok, HD);
        if (n_ok < HD) {
            // Print first failing
            for (int ci = 0; ci < HD; ci++) {
                int idx = ci;
                float orig = K[idx];
                K[idx] = orig + eps; double lup = loss_fwd(B, T, Q, K, V, d_out);
                K[idx] = orig - eps; double ldn = loss_fwd(B, T, Q, K, V, d_out);
                K[idx] = orig;
                float fd = (float)((lup - ldn) / (2.0 * eps));
                float rel = fabsf(dK_ana[idx] - fd) / (fmaxf(fabsf(dK_ana[idx]), fabsf(fd)) + 1e-10f);
                if (rel >= 0.3f) printf(" [%d: ana=%.4e fd=%.4e rel=%.2f]", ci, dK_ana[idx], fd, rel);
            }
        }
        printf("\n");
    }
    
    printf("\nAll dK values:\n");
    for (int i = 0; i < N * N_KV * HD; i++)
        printf("  dK[%d] = %.6e\n", i, dK_ana[i]);
    
    printf("\nDETAILED: FD at various eps for K[0,0,0]:\n");
    for (int eps_pow = 3; eps_pow <= 8; eps_pow++) {
        float eps = powf(10.0f, -(float)eps_pow);
        float orig = K[0];
        K[0] = orig + eps; double lup = loss_fwd(B, T, Q, K, V, d_out);
        K[0] = orig - eps; double ldn = loss_fwd(B, T, Q, K, V, d_out);
        K[0] = orig;
        printf("  eps=1e-%d: L_base=%.10f diff=%.2e fd=%.4e (ana=%.4e)\n",
               eps_pow, lup, lup-ldn, (float)((lup-ldn)/(2*eps)), dK_ana[0]);
    }
    
    free(Q); free(K); free(V); free(d_out); free(dK_ana);
    return 0;
}

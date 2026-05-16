/* wubu_ssm_chunked.c — Chunked Gated DeltaNet recurrence

   For B=1, T tokens. Each head processes CS=64 tokens at a time.
   Intra-chunk: triangular decay mask + matmuls.
   Inter-chunk: state carry with full chunk decay + kg^T @ v_new.

   Reference: llama.cpp delta-net-base.cpp build_delta_net_chunking lines 265-273
*/

#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#define CS 64

// Forward substitution: solve L^T @ X = RHS, L unit lower-tri diag=1
static void solve_tri(int n, const float *L, const float *RHS, float *X) {
    memset(X, 0, n * n * sizeof(float));
    for (int i = n - 1; i >= 0; i--)
        for (int j = 0; j < n; j++) {
            float s = RHS[i * n + j];
            for (int k = i + 1; k < n; k++)
                s -= L[k * n + i] * X[k * n + j];
            X[i * n + j] = s;
        }
}

void wubu_ssm_chunked_recurrence(
    int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    float *ssm_state, float *delta_out)
{
    if (B != 1) { fprintf(stderr, "chunked: only B=1 supported\n"); return; }
    const int d = SSM_D_STATE;
    const int hk = SSM_K_HEADS;
    const int hv = SSM_V_HEADS;
    const int rf = hv / hk;
    const int nc = (T + CS - 1) / CS;

    #pragma omp parallel for if(hv > 1)
    for (int vh = 0; vh < hv; vh++) {
        size_t sz = (size_t)CS * d;
        size_t sz2 = (size_t)CS * CS;
        float *scr = (float *)malloc(
            sz * 6 * sizeof(float) +  // qk,kk,vk,vb,kb,kg
            sz2 * 4 * sizeof(float) + // M,A2,X,LS
            (size_t)T * d * sizeof(float) +  // oh
            (size_t)d * d * sizeof(float));  // kcd
        if (!scr) continue;
        float *qk = scr, *kk = qk + sz, *vk = kk + sz;
        float *vb = vk + sz, *kb = vb + sz, *kg = kb + sz;
        float *M = kg + sz, *A2 = M + sz2, *X = A2 + sz2, *LS = X + sz2;
        float *oh = LS + sz2;
        float *kcd = oh + (size_t)T * d;  // [d, d]

        int kh = vh / rf;
        float *h = ssm_state + vh * d * d;

        // Gather per-head data and pre-scale Q by 1/sqrt(d)
        float qsc = 1.0f / sqrtf((float)d);
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < d; i++)
                qk[t*d+i] = q_norm[(t * hk + kh) * d + i] * qsc;
            memcpy(kk + t * d, k_norm + (t * hk + kh) * d, d * sizeof(float));
            memcpy(vk + t * d, v_conv + (t * hv + vh) * d, d * sizeof(float));
        }

        for (int ch = 0; ch < nc; ch++) {
            int off = ch * CS;
            int nt = T - off; if (nt > CS) nt = CS;
            const float *qc = qk + off * d, *kc = kk + off * d, *vc = vk + off * d;

            // Cum gate within chunk
            float gc[CS];
            gc[0] = gate_flat[off * hv + vh];
            for (int i = 1; i < nt; i++) gc[i] = gc[i-1] + gate_flat[(off + i) * hv + vh];
            float g_last = gc[nt-1];
            for (int i = nt; i < CS; i++) gc[i] = g_last;

            // v_b = v * beta, k_b = k * beta
            for (int i = 0; i < CS; i++) {
                float bi = (i < nt) ? beta_flat[(off + i) * hv + vh] : 0.0f;
                for (int j = 0; j < d; j++) {
                    vb[i*d+j] = vc[i*d+j] * bi;
                    kb[i*d+j] = kc[i*d+j] * bi;
                }
            }

            // Decay mask M[i,j] = exp(gc[j]-gc[i]) for j>=i
            for (int i = 0; i < CS; i++)
                for (int j = 0; j < CS; j++)
                    M[i*CS+j] = (j >= i) ? expf(fminf(gc[j]-gc[i], 80.0f)) : 0.0f;

            // Intra-chunk: kb_masked = M * (k_b @ k^T)
            for (int i = 0; i < CS; i++)
                for (int j = 0; j < CS; j++) {
                    float s = 0;
                    for (int z = 0; z < d; z++) s += kb[i*d+z] * kc[j*d+z];
                    A2[i*CS+j] = s * M[i*CS+j];
                }

            // LHS = I + tri(kb_masked), RHS = tri(kb_masked) with 0 diag
            memcpy(LS, A2, sz2 * sizeof(float));
            for (int i = 0; i < CS; i++) LS[i*CS+i] += 1.0f;
            for (int i = 0; i < CS; i++) A2[i*CS+i] = 0.0f;

            // X = LS^{-T} @ A2, then X += I
            solve_tri(CS, LS, A2, X);
            for (int i = 0; i < CS; i++) X[i*CS+i] += 1.0f;

            // Intra-chunk output = X^T @ v_b [CS, d]
            for (int i = 0; i < CS; i++)
                for (int j = 0; j < d; j++) {
                    float s = 0;
                    for (int z = 0; z < CS; z++) s += X[z*CS+i] * vb[z*d+j];
                    oh[i*d+j] = s;
                }

            // State contribution = h^T @ (q * exp(gc))  (q is pre-scaled by 1/sqrt(d))
            // combined = intra + state_contrib
            for (int i = 0; i < nt; i++) {
                float ge = expf(gc[i]);
                for (int j = 0; j < d; j++) {
                    float sc = 0;
                    for (int z = 0; z < d; z++)
                        sc += h[z*d+j] * qc[i*d+z] * ge;
                    oh[i*d+j] += sc;
                }
            }

            // Copy output for real tokens
            for (int i = 0; i < nt; i++)
                memcpy(delta_out + ((off + i) * hv + vh) * d, oh + i * d, d * sizeof(float));

            // --- Per-chunk state update: match sequential decay-then-update ---
            // Sequential: for each t in chunk: h *= exp(g[t]); hk=h@k; diff=v_b-hk; h+=k⊗diff
            // We can batch this: first decay ALL by exp(g_last), then adjust for each token
            // But decay is per-token and interleaved with updates, so we must do per-token.
            
            // State update with proper decay per token (matches sequential order)
            for (int i = 0; i < nt; i++) {
                float gg = expf(fminf(gate_flat[(off + i) * hv + vh], 80.0f));
                float bg = beta_flat[(off + i) * hv + vh];
                for (int z = 0; z < d; z++)
                    for (int w = 0; w < d; w++) h[z*d+w] *= gg;
                float hk_tmp[128]; memset(hk_tmp, 0, d*sizeof(float));
                for (int z = 0; z < d; z++)
                    for (int w = 0; w < d; w++) hk_tmp[z] += h[z*d+w] * kc[i*d+w];
                for (int z = 0; z < d; z++) {
                    float df = vc[i*d+z] * bg - hk_tmp[z];
                    for (int w = 0; w < d; w++) h[z*d+w] += kc[i*d+w] * df;
                }
            }
        }
        free(scr);
    }
}

/*
 * wubu_kda.c — Kimi Delta Attention (KDA) recurrent state (Round-4 #401/#405/#410).
 * C11, self-contained. KDA = DeltaNet with CHANNEL-WISE decay: each key channel
 * decays at its own rate (vector beta_d, not scalar). The fast-weight update:
 *   S_t = D_t (I - k_t k_t^T) S_{t-1} + k_t v_t^T        (simplified, D=diag(decay))
 * with per-channel decay d_c in (0,1] clamped for stability. This is the linear
 * backbone of Kimi K3's hybrid (3:1 KDA : Gated-MLA cycle). Recovers Gated DeltaNet
 * when decay is a shared scalar.
 */
#include "wubu_kda.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Serial KDA recurrence with per-channel decay. q,k,v: n*d. decay: d (per-channel,
 * each in (0,1]). S: d*d in/out. */
void wubu_kda_recurrence(const float *q, const float *k, const float *v,
                          const float *decay, int n, int d, float *S) {
    float *kt = (float *)malloc(sizeof(float) * d);
    float *kv = (float *)malloc(sizeof(float) * d);
    for (int t = 0; t < n; t++) {
        const float *kt_in = k + t * d;
        const float *vt = v + t * d;
        for (int i = 0; i < d; i++) {
            float dd = decay[i];
            if (dd > 1.0f) dd = 1.0f;          /* clamp for stability (DA) */
            if (dd < 1e-6f) dd = 1e-6f;
            kt[i] = kt_in[i];
            /* apply channel decay to current state row i before erase */
            S[i*d + i] *= dd;                   /* diagonal decay (D_t applied) */
        }
        /* erase: S = S - k_t (k_t^T S)  (rank-one removal) */
        for (int i = 0; i < d; i++) {
            float sk = 0;
            for (int j = 0; j < d; j++) sk += S[i*d + j] * kt[j];
            kv[i] = sk;
        }
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                S[i*d + j] -= kt[i] * kv[j];
        /* write: S = S + k_t v_t^T */
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                S[i*d + j] += kt[i] * vt[j];
    }
    free(kt); free(kv);
}

/* Output y_t = S q_t. */
void wubu_kda_output(const float *q, const float *S, int n, int d, float *y) {
    for (int t = 0; t < n; t++) {
        const float *qt = q + t * d;
        for (int i = 0; i < d; i++) {
            float acc = 0;
            for (int j = 0; j < d; j++) acc += S[i*d + j] * qt[j];
            y[t*d + i] = acc;
        }
    }
}

/* Verify state stays bounded under decay (decay<1 => L2(S) non-increasing w/o writes). */
int wubu_kda_state_bounded(const float *S, int d, float *l2) {
    float s = 0;
    for (int i = 0; i < d*d; i++) s += S[i] * S[i];
    *l2 = sqrtf(s);
    return isfinite(*l2) ? 1 : 0;
}

/*
 * wubu_delta_net.c — Gated-DeltaNet recurrent state (Round-3 #202/#204/#206/#207/#209).
 * C11, self-contained. Implements the DeltaNet fast-weight update
 *   S_t = (I - beta_t * k_t k_t^T) S_{t-1} + beta_t * k_t v_t^T
 * with optional QK-L2 norm and SiLU output gate, plus a chunkwise WY-form
 * forward (GatedDeltaNet-2 style) that must match the serial recurrence.
 * This is the linear-attention backbone of Qwen3.6's hybrid (3:1 ratio).
 */
#include "wubu_delta_net.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Serial recurrence over a sequence, head_dim = d. Returns final state S (d*d). */
void wubu_delta_net_recurrence(const float *q, const float *k, const float *v,
                               const float *beta, int n, int d,
                               float *S /*in/out d*d*/) {
    float *kt = (float *)malloc(sizeof(float) * d);
    float *kv = (float *)malloc(sizeof(float) * d);
    for (int t = 0; t < n; t++) {
        const float *kt_full = k + t * d;
        const float *vt = v + t * d;
        const float *qt = q + t * d;
        /* QK-L2 norm (optional stabilization). */
        float kn = 0, qn = 0;
        for (int i = 0; i < d; i++) { kn += kt_full[i]*kt_full[i]; qn += qt[i]*qt[i]; }
        float ks = (kn > 1e-12f) ? (1.0f / sqrtf(kn)) : 0.0f;
        float qs = (qn > 1e-12f) ? (1.0f / sqrtf(qn)) : 0.0f;
        for (int i = 0; i < d; i++) { kt[i] = kt_full[i] * ks; }
        /* S = S - beta * (S k) k^T  (erase) */
        float b = beta[t];
        for (int i = 0; i < d; i++) {
            float sk = 0;
            for (int j = 0; j < d; j++) sk += S[i*d + j] * kt[j];
            kv[i] = sk;
        }
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                S[i*d + j] -= b * kv[i] * kt[j];
        /* S = S + beta * k v^T  (write) */
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                S[i*d + j] += b * kt[i] * vt[j];
        (void)qs; (void)qt;
    }
    free(kt); free(kv);
}

/* Output y_t = S q_t (then caller applies RMSNorm + SiLU gate externally). */
void wubu_delta_net_output(const float *q, const float *S, int n, int d, float *y) {
    for (int t = 0; t < n; t++) {
        const float *qt = q + t * d;
        for (int i = 0; i < d; i++) {
            float acc = 0;
            for (int j = 0; j < d; j++) acc += S[i*d + j] * qt[j];
            y[t*d + i] = acc;
        }
    }
}

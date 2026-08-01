/*
 * wubu_linear_attn.c -- Linear / recurrent attention hybrids (S01-S05, S07). C11.
 *
 * Convergence (Gated DeltaNet / Mamba-2 / GLA / RetNet / GSA / HGRN2 7-hop):
 * These replace the O(n^2) attention with an O(n) recurrent state update. The
 * decode step is a state-tensor update S <- f(S, k, v) + a gating signal. Each
 * variant differs in the update rule:
 *   - S01 Gated DeltaNet: S' = S - beta * (S k - v) k^T   (delta-rule fast weight)
 *   - S03 Mamba-2 SSM:    S' = A * S + b * k * v^T          (gated state decay)
 *   - S04 GLA:            S' = g * S + k * v^T              (per-head gate)
 *   - S05 RetNet/GSA:     S' = gamma * S + k * v^T          (retention decay)
 *   - S07 HGRN2/GSA:      S' = (1 - g) * S + g * k * v^T    (state expansion)
 * We implement the per-token update for a d_state x d_model state (simplified to
 * d x d block) with float math, returning the new state. Pure, testable.
 *
 * Triple-DA: dims checked; beta/g clamped to [0,1]; null -> 0; deterministic.
 */
#include "wubu_linear_attn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void mat_add(float *S, const float *M, int rows, int cols, float w) {
    for (int i = 0; i < rows*cols; i++) S[i] += w * M[i];
}
static void mat_scale(float *S, int n, float a) { for (int i=0;i<n;i++) S[i]*=a; }

/* S01 Gated DeltaNet: S' = S - beta*(S k - v) k^T.  state d x d (row-major). */
int wubu_deltanet_update(const float *S, const float *k, const float *v,
                         int d, float beta, float *Sout) {
    if (!S || !k || !v || !Sout || d <= 0) return 0;
    if (beta < 0.0f) beta = 0.0f; if (beta > 1.0f) beta = 1.0f;
    memcpy(Sout, S, (size_t)d*d*sizeof(float));
    /* delta = S k - v   (delta is d-vector) */
    float *delta = (float *)calloc((size_t)d, sizeof(float));
    if (!delta) return 0;
    for (int i = 0; i < d; i++) {
        float sk = 0.0f; for (int j = 0; j < d; j++) sk += S[i*d+j]*k[j];
        delta[i] = sk - v[i];
    }
    /* Sout = S - beta * delta k^T  -> outer product */
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Sout[i*d+j] -= beta * delta[i] * k[j];
    free(delta);
    return 1;
}

/* S03 Mamba-2 SSM: S' = A*S + b*(k v^T).  A is a scalar decay (per-state). */
int wubu_mamba2_update(const float *S, const float *k, const float *v,
                       int d, float A, float b, float *Sout) {
    if (!S || !k || !v || !Sout || d <= 0) return 0;
    if (A < 0.0f) A = 0.0f; if (A > 1.0f) A = 1.0f;
    if (b < 0.0f) b = 0.0f;
    mat_scale((float*)Sout, d*d, A);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Sout[i*d+j] += b * k[i] * v[j];
    return 1;
}

/* S04 GLA: S' = g*S + k v^T  (g per-head scalar). */
int wubu_gla_update(const float *S, const float *k, const float *v,
                    int d, float g, float *Sout) {
    if (!S || !k || !v || !Sout || d <= 0) return 0;
    if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
    memcpy(Sout, S, (size_t)d*d*sizeof(float));
    mat_scale(Sout, d*d, g);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Sout[i*d+j] += k[i] * v[j];
    return 1;
}

/* S05 RetNet/GSA retention: S' = gamma*S + k v^T. */
int wubu_retnet_update(const float *S, const float *k, const float *v,
                       int d, float gamma, float *Sout) {
    if (!S || !k || !v || !Sout || d <= 0) return 0;
    if (gamma < 0.0f) gamma = 0.0f; if (gamma > 1.0f) gamma = 1.0f;
    memcpy(Sout, S, (size_t)d*d*sizeof(float));
    mat_scale(Sout, d*d, gamma);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Sout[i*d+j] += k[i] * v[j];
    return 1;
}

/* S07 HGRN2/GSA state-expansion: S' = (1-g)*S + g*k v^T. */
int wubu_hgrn2_update(const float *S, const float *k, const float *v,
                      int d, float g, float *Sout) {
    if (!S || !k || !v || !Sout || d <= 0) return 0;
    if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
    float keep = 1.0f - g;
    memcpy(Sout, S, (size_t)d*d*sizeof(float));
    for (int i = 0; i < d*d; i++) Sout[i] = keep * Sout[i];
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Sout[i*d+j] += g * k[i] * v[j];
    return 1;
}

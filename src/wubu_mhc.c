/*
 * wubu_mhc.c — Manifold-Constrained Hyper-Connections (Round-3 #213/#214/#215/#216).
 * C11, self-contained. mHC widens the residual stream by factor `exp` and mixes
 * via pre/post mapping matrices constrained to a manifold: pre/post projections
 * use NON-NEGATIVE (sigmoid) weights to avoid signal cancellation, and the
 * residual mixing matrix is constrained so the identity mapping is preserved at
 * init (sum of rows = I) -- restoring the stability residual connections give.
 */
#include "wubu_mhc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

struct wubu_mhc {
    int exp;       /* expansion rate (residual stream width multiplier) */
    int dim;       /* base hidden dim */
    float *pre;    /* exp*dim x dim, sigmoid-constrained (non-negative) */
    float *post;   /* dim x exp*dim, sigmoid-constrained */
    float *mix;    /* exp*dim x exp*dim, manifold-constrained (rows sum to identity) */
};

wubu_mhc_t *wubu_mhc_create(int dim, int exp) {
    if (dim <= 0 || exp <= 0) return NULL;
    wubu_mhc_t *m = (wubu_mhc_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->dim = dim; m->exp = exp;
    int w = exp * dim;
    m->pre  = (float *)malloc(sizeof(float) * w * dim);
    m->post = (float *)malloc(sizeof(float) * dim * w);
    m->mix  = (float *)malloc(sizeof(float) * w * w);
    if (!m->pre || !m->post || !m->mix) { wubu_mhc_free(m); return NULL; }
    /* Identity init: pre = I (first dim block), post = I, mix = I (manifold). */
    for (int i = 0; i < w * dim; i++) m->pre[i] = -1e3f;
    for (int i = 0; i < dim * w; i++) m->post[i] = -1e3f;
    for (int i = 0; i < w * w; i++) m->mix[i] = 0;
    /* Exact-identity init: diagonal logits +1e3 (sigmoid->1), off-diag -1e3.
     * pre is w x dim (rows w=exp*dim, cols dim): identity only on first dim x dim. */
    for (int i = 0; i < dim; i++) {
        m->pre[i * dim + i] = 1e3f;     /* pre diagonal (w x dim) */
        m->post[i * w + i] = 1e3f;      /* post diagonal (dim x w) */
    }
    for (int i = 0; i < w; i++) m->mix[i * (w + 1)] = 1.0f;  /* I on manifold */
    return m;
}

void wubu_mhc_free(wubu_mhc_t *m) {
    if (!m) return;
    free(m->pre); free(m->post); free(m->mix); free(m);
}

/* Apply sigmoid non-negativity in-place to a weight buffer (constraint). */
void wubu_mhc_apply_nonneg(float *w, int n) {
    for (int i = 0; i < n; i++) w[i] = 1.0f / (1.0f + expf(-w[i]));
}

/* Verify identity-mapping property: mix rows sum to 1 on diagonal blocks. */
int wubu_mhc_identity_ok(const wubu_mhc_t *m) {
    if (!m) return 0;
    int w = m->exp * m->dim;
    for (int i = 0; i < w; i++) {
        float row = 0;
        for (int j = 0; j < w; j++) row += m->mix[i*w + j];
        if (fabsf(row - 1.0f) > 1e-4f) return 0;
    }
    return 1;
}

/* Set weights to EXACT identity (diagonal logits +1e3 so sigmoid->1, off-diag
 * -1e3 so sigmoid->0; mix = I). Used for tests/baselines where passthrough is
 * required (the sigmoid constraint alone only gives approximate identity). */
void wubu_mhc_set_identity(wubu_mhc_t *m) {
    int w = m->exp * m->dim, dim = m->dim;
    for (int i = 0; i < w * dim; i++) m->pre[i] = -1e3f;
    for (int i = 0; i < dim * w; i++) m->post[i] = -1e3f;
    for (int i = 0; i < w; i++) m->mix[i * (w + 1)] = 1.0f;
    for (int i = 0; i < dim; i++) { m->pre[i * (dim + 1)] = 1e3f; m->post[i * (w + 1)] = 1e3f; }
}
void wubu_mhc_forward(const wubu_mhc_t *m, const float *x, float *r_out, float *y_out) {
    if (!m || !x || !y_out) return;
    int w = m->exp * m->dim, dim = m->dim;
    /* Numerically safe sigmoid: clamp argument to [-50,50] so expf never hits
     * overflow/underflow (expf(+-1000) is implementation-fragile -> NaN under
     * some -ffast-math/-march paths; expf(+-50) already saturates to ~1e-22). */
    float *r = (float *)calloc(w, sizeof(float));
    for (int i = 0; i < w; i++) {
        float acc = 0;
        for (int j = 0; j < dim; j++) {
            float a = m->pre[i*dim + j];
            if (a >  50.0f) a =  50.0f;
            if (a < -50.0f) a = -50.0f;
            acc += (1.0f / (1.0f + expf(-a))) * x[j];
        }
        r[i] = acc;
    }
    float *r2 = (float *)calloc(w, sizeof(float));
    for (int i = 0; i < w; i++)
        for (int j = 0; j < w; j++) r2[i] += m->mix[i*w + j] * r[j];
    for (int i = 0; i < dim; i++) {
        float acc = 0;
        for (int j = 0; j < w; j++) {
            float a = m->post[i*w + j];
            if (a >  50.0f) a =  50.0f;
            if (a < -50.0f) a = -50.0f;
            acc += (1.0f / (1.0f + expf(-a))) * r2[j];
        }
        y_out[i] = acc;
    }
    if (r_out) memcpy(r_out, r2, sizeof(float) * w);
    free(r); free(r2);
}

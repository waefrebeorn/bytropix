/*
 * wubu_manifold_train.c — manifold-native training primitives.
 * See include/wubu_manifold_train.h for the design and the honest scope.
 */
#include "wubu_manifold_train.h"
#include "wubu_mobius.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Deterministic LCG so anchors are reproducible across runs. */
static unsigned lcg(unsigned *s)
{
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

static float frand(unsigned *s)
{
    return (float)(lcg(s) % 100000u) / 100000.0f;   /* [0,1) */
}

int wubu_manifold_head_init(wubu_manifold_head_t *h, int dim, int n_classes,
                            float R, unsigned seed)
{
    if (!h || dim <= 0 || n_classes <= 0 || !(R > 0.0f)) return -1;
    memset(h, 0, sizeof(*h));
    h->dim = dim;
    h->n_classes = n_classes;
    h->R = R;
    h->anchor = (float *)malloc((size_t)dim * (size_t)n_classes * sizeof(float));
    if (!h->anchor) return -1;

    /* Draw a direction, then place the anchor at radius 0.35R. This is
     * inside the ball by construction and spreads the classes. */
    for (int k = 0; k < n_classes; k++) {
        float *a = h->anchor + (size_t)k * (size_t)dim;
        float n2 = 0.0f;
        for (int j = 0; j < dim; j++) {
            float v = frand(&seed) * 2.0f - 1.0f;
            a[j] = v;
            n2 += v * v;
        }
        float n = sqrtf(n2);
        if (n < 1e-6f) { a[0] = 0.35f * R; continue; }
        float s = (0.35f * R) / n;
        for (int j = 0; j < dim; j++) a[j] *= s;
    }
    return 0;
}

void wubu_manifold_head_free(wubu_manifold_head_t *h)
{
    if (!h) return;
    free(h->anchor);
    h->anchor = NULL;
    h->dim = h->n_classes = 0;
    h->R = 0.0f;
}

int wubu_manifold_logits(const wubu_manifold_head_t *h, const float *x,
                         float *logits)
{
    if (!h || !h->anchor || !x || !logits) return -1;
    int d = h->dim;
    float R = h->R;
    float *u = (float *)malloc((size_t)d * sizeof(float));
    float *v = (float *)malloc((size_t)d * sizeof(float));
    if (!u || !v) { free(u); free(v); return -1; }

    wubu_log_map(x, d, R, u);
    float inv = 1.0f / (2.0f * R * R);
    for (int k = 0; k < h->n_classes; k++) {
        const float *a = h->anchor + (size_t)k * (size_t)d;
        wubu_log_map(a, d, R, v);
        float s = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = u[j] - v[j];
            s += diff * diff;
        }
        logits[k] = -s * inv;
    }
    free(u); free(v);
    return 0;
}

float wubu_manifold_ce(const wubu_manifold_head_t *h, const float *x, int label,
                       float *dtan, float *danchor)
{
    if (!h || !h->anchor || !x || label < 0 || label >= h->n_classes) return 0.0f;
    int d = h->dim, K = h->n_classes;
    float R = h->R;

    float *logits = (float *)malloc((size_t)K * sizeof(float));
    float *u = (float *)malloc((size_t)d * sizeof(float));
    float *v = (float *)malloc((size_t)d * sizeof(float));
    float *gdu = (float *)calloc((size_t)d, sizeof(float));
    float *dv = (float *)malloc((size_t)d * sizeof(float));
    if (!logits || !u || !v || !gdu || !dv) {
        free(logits); free(u); free(v); free(gdu); free(dv);
        return 0.0f;
    }

    wubu_manifold_logits(h, x, logits);
    float mx = logits[0];
    for (int k = 1; k < K; k++) if (logits[k] > mx) mx = logits[k];
    float sum = 0.0f;
    for (int k = 0; k < K; k++) { logits[k] = expf(logits[k] - mx); sum += logits[k]; }
    if (sum <= 0.0f) sum = 1.0f;
    float loss = 0.0f;
    for (int k = 0; k < K; k++) logits[k] /= sum;
    if (logits[label] > 1e-30f) loss = -logf(logits[label]);
    else loss = 30.0f;   /* cap: log(1e-30) is not a useful gradient */

    wubu_log_map(x, d, R, u);
    float inv = 1.0f / (R * R);
    for (int k = 0; k < K; k++) {
        float g = logits[k] - (k == label ? 1.0f : 0.0f);
        const float *a = h->anchor + (size_t)k * (size_t)d;
        wubu_log_map(a, d, R, v);
        for (int j = 0; j < d; j++) {
            float diff = u[j] - v[j];
            /* d logit_k / du = -(u - v_k)/R^2 */
            gdu[j] += g * (-diff * inv);
            if (danchor) {
                /* d logit_k / dv_k = +(u - v_k)/R^2, then chain through
                 * log_map_backward into ambient anchor coordinates. */
                dv[j] = g * (diff * inv);
            }
        }
        if (danchor) {
            float *da = (float *)calloc((size_t)d, sizeof(float));
            if (da) {
                wubu_log_map_backward(a, d, R, v, dv, da);
                float *dst = danchor + (size_t)k * (size_t)d;
                for (int j = 0; j < d; j++) dst[j] += da[j];
                free(da);
            }
        }
    }
    if (dtan) memcpy(dtan, gdu, (size_t)d * sizeof(float));

    free(logits); free(u); free(v); free(gdu); free(dv);
    return loss;
}

int wubu_manifold_lora(const float *h, int dim, float R,
                       const float *A, const float *B, int rank, float *out)
{
    if (!h || dim <= 0 || !(R > 0.0f) || !out) return -1;
    float *u = (float *)malloc((size_t)dim * sizeof(float));
    if (!u) return -1;
    wubu_log_map(h, dim, R, u);

    if (A && B && rank > 0) {
        float *r = (float *)calloc((size_t)rank, sizeof(float));
        if (!r) { free(u); return -1; }
        for (int i = 0; i < rank; i++) {
            float s = 0.0f;
            for (int j = 0; j < dim; j++)
                s += A[(size_t)i * (size_t)dim + (size_t)j] * u[j];
            r[i] = s;
        }
        for (int j = 0; j < dim; j++) {
            float s = 0.0f;
            for (int i = 0; i < rank; i++)
                s += B[(size_t)j * (size_t)rank + (size_t)i] * r[i];
            u[j] += s;
        }
        free(r);
    }

    wubu_exp_map(u, dim, R, out);
    free(u);
    return 0;
}

float wubu_manifold_hierarchy_loss(const float *parent, const float *child,
                                   const float *sibling, int dim, float R,
                                   float margin)
{
    if (!parent || !child || !sibling || dim <= 0 || !(R > 0.0f)) return 0.0f;
    float dp = wubu_poincare_dist(child, parent, dim, R);
    float ds = wubu_poincare_dist(child, sibling, dim, R);
    if (!isfinite(dp) || !isfinite(ds)) return 0.0f;
    float loss = dp - ds + margin;
    return loss > 0.0f ? loss : 0.0f;
}

float wubu_manifold_radius_penalty(const float *x, int dim, float R,
                                   float lo, float hi, float *dtan)
{
    if (!x || dim <= 0 || !(R > 0.0f) || !(lo > 0.0f) || !(hi > lo) || !(hi < R))
        return 0.0f;

    float *u = (float *)malloc((size_t)dim * sizeof(float));
    if (!u) return 0.0f;
    wubu_log_map(x, dim, R, u);

    float n2 = 0.0f;
    for (int i = 0; i < dim; i++) n2 += u[i] * u[i];
    float n = sqrtf(n2);
    if (n < 1e-8f) { free(u); return 0.0f; }

    /* Radii in tangent coordinates: r = R*tanh(||u||/R) => ||u|| = R*atanh(r/R) */
    float lo_u = R * atanhf(lo / R);
    float hi_u = R * atanhf(hi / R);

    /* Hinge at a 5% inward margin. A hinge exactly on the boundary only
     * converges asymptotically (radius approaches lo from below forever),
     * so the band would never be strictly occupied in finite steps. */
    float m = 0.05f * (hi_u - lo_u);
    float lo_t = lo_u + m;
    float hi_t = hi_u - m;

    float pen = 0.0f, g = 0.0f;
    if (n < lo_t)      { float d = lo_t - n; pen = d * d; g = -2.0f * d; }
    else if (n > hi_t) { float d = n - hi_t; pen = d * d; g =  2.0f * d; }

    if (dtan) {
        if (pen > 0.0f) {
            for (int i = 0; i < dim; i++) dtan[i] = g * (u[i] / n);
        } else {
            for (int i = 0; i < dim; i++) dtan[i] = 0.0f;
        }
    }
    free(u);
    return pen;
}

int wubu_manifold_lora_backward(const float *h, int dim, float R,
                                const float *A, const float *B, int rank,
                                const float *dout, float *dA, float *dB)
{
    if (!h || dim <= 0 || !(R > 0.0f) || !dout || !dA || !dB || rank <= 0 ||
        !A || !B) return -1;

    float *u   = (float *)malloc((size_t)dim * sizeof(float));
    float *up  = (float *)malloc((size_t)dim * sizeof(float));
    float *out = (float *)malloc((size_t)dim * sizeof(float));
    float *dup = (float *)malloc((size_t)dim * sizeof(float));
    float *r   = (float *)calloc((size_t)rank, sizeof(float));
    float *dr  = (float *)calloc((size_t)rank, sizeof(float));
    if (!u || !up || !out || !dup || !r || !dr) {
        free(u); free(up); free(out); free(dup); free(r); free(dr);
        return -1;
    }

    wubu_log_map(h, dim, R, u);
    for (int i = 0; i < rank; i++) {
        float s = 0.0f;
        for (int j = 0; j < dim; j++)
            s += A[(size_t)i * (size_t)dim + (size_t)j] * u[j];
        r[i] = s;
    }
    for (int k = 0; k < dim; k++) {
        float s = u[k];
        for (int i = 0; i < rank; i++)
            s += B[(size_t)k * (size_t)rank + (size_t)i] * r[i];
        up[k] = s;
    }
    wubu_exp_map(up, dim, R, out);

    /* dL/du' from dL/dout through the exp-map Jacobian. */
    wubu_exp_map_backward(up, dim, R, out, dout, dup);

    /* dL/dr = B^T dup ; dL/dB = dup (x) r ; dL/dA = dr (x) u */
    for (int i = 0; i < rank; i++) {
        float s = 0.0f;
        for (int k = 0; k < dim; k++)
            s += B[(size_t)k * (size_t)rank + (size_t)i] * dup[k];
        dr[i] = s;
    }
    for (int k = 0; k < dim; k++) {
        for (int i = 0; i < rank; i++)
            dB[(size_t)k * (size_t)rank + (size_t)i] = dup[k] * r[i];
    }
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < dim; j++)
            dA[(size_t)i * (size_t)dim + (size_t)j] = dr[i] * u[j];
    }

    free(u); free(up); free(out); free(dup); free(r); free(dr);
    return 0;
}

int wubu_manifold_train_selftest(void)
{
    wubu_manifold_head_t h;
    if (wubu_manifold_head_init(&h, 4, 3, 1.0f, 7u) != 0) return 1;

    float logits[3];
    /* The anchor's own logit must be the maximum (distance zero). */
    for (int k = 0; k < 3; k++) {
        const float *a = h.anchor + (size_t)k * 4;
        if (wubu_manifold_logits(&h, a, logits) != 0) return 2;
        for (int j = 0; j < 3; j++)
            if (j != k && logits[j] >= logits[k]) return 3;
    }

    /* B = 0 adapter is the identity. */
    float x[4] = {0.1f, -0.05f, 0.02f, 0.0f};
    float y[4];
    if (wubu_manifold_lora(x, 4, 1.0f, NULL, NULL, 0, y) != 0) return 4;
    for (int j = 0; j < 4; j++)
        if (fabsf(x[j] - y[j]) > 1e-5f) return 5;

    wubu_manifold_head_free(&h);
    return 0;
}

/*
 * wubu_mhc_mh.c -- Multi-head Hyper-Connections (the 2512.24880 form).
 *
 * The KAHUNA's residual architecture: a GROUP of nh hidden streams, a
 * manifold-constrained (row-softmax) mixing matrix, gated writes, and an
 * exact-identity initialization that degenerates to a plain residual
 * connection -- the function-preserving oracle that proves the math.
 */
#include "wubu_mhc_mh.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct wubu_mhc_mh {
    int nh, d;
    float *mix;   /* [nh*nh], row-major; row i = mixing weights for h[i] */
    uint32_t seed;
};

/* splitmix64 -- deterministic, no external RNG */
static uint64_t smix(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float rnd01(uint64_t *s)
{
    return (float)((double)(smix(s) >> 11) / (double)(1ull << 53));
}

wubu_mhc_mh_t *wubu_mhc_mh_create(int nh, int d, uint32_t seed)
{
    if (nh < 1 || d < 1) return NULL;
    wubu_mhc_mh_t *m = (wubu_mhc_mh_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->nh = nh;
    m->d = d;
    m->seed = seed;
    m->mix = (float *)malloc((size_t)nh * nh * sizeof(float));
    if (!m->mix) { free(m); return NULL; }
    /* deterministic random init, then constrain rows (convex) */
    uint64_t s = seed;
    for (int i = 0; i < nh * nh; i++) m->mix[i] = rnd01(&s);
    wubu_mhc_mh_constrain(m);
    return m;
}

void wubu_mhc_mh_free(wubu_mhc_mh_t *m)
{
    if (!m) return;
    free(m->mix);
    free(m);
}

void wubu_mhc_mh_set_identity(wubu_mhc_mh_t *m)
{
    if (!m) return;
    memset(m->mix, 0, (size_t)m->nh * m->nh * sizeof(float));
    for (int i = 0; i < m->nh; i++) m->mix[i * m->nh + i] = 1.0f;
}

void wubu_mhc_mh_constrain(wubu_mhc_mh_t *m)
{
    if (!m) return;
    for (int i = 0; i < m->nh; i++) {
        float mx = -1e30f;
        for (int k = 0; k < m->nh; k++) {
            float v = m->mix[i * m->nh + k];
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < m->nh; k++)
            sum += expf(m->mix[i * m->nh + k] - mx);
        float inv = 1.0f / sum;
        for (int k = 0; k < m->nh; k++)
            m->mix[i * m->nh + k] = expf(m->mix[i * m->nh + k] - mx) * inv;
    }
}

int wubu_mhc_mh_read(const wubu_mhc_mh_t *m, const float *const *h,
                     int i, float *out)
{
    if (!m || !h || !out || i < 0 || i >= m->nh) return -1;
    const float *row = &m->mix[i * m->nh];
    for (int j = 0; j < m->d; j++) {
        float acc = 0.0f;
        for (int k = 0; k < m->nh; k++) acc += row[k] * h[k][j];
        out[j] = acc;
    }
    return 0;
}

int wubu_mhc_mh_write(wubu_mhc_mh_t *m, float *h_i, const float *y,
                      float alpha)
{
    if (!m || !h_i || !y) return -1;
    const float beta = 1.0f - alpha;
    for (int j = 0; j < m->d; j++) h_i[j] = alpha * h_i[j] + beta * y[j];
    return 0;
}

int wubu_mhc_mh_nh(const wubu_mhc_mh_t *m) { return m ? m->nh : 0; }
int wubu_mhc_mh_dim(const wubu_mhc_mh_t *m) { return m ? m->d : 0; }

const float *wubu_mhc_mh_mixing_row(const wubu_mhc_mh_t *m, int i)
{
    if (!m || i < 0 || i >= m->nh) return NULL;
    return &m->mix[i * m->nh];
}

int wubu_mhc_mh_manifold_ok(const wubu_mhc_mh_t *m)
{
    if (!m) return 0;
    for (int i = 0; i < m->nh; i++) {
        float s = 0.0f;
        for (int k = 0; k < m->nh; k++) {
            float v = m->mix[i * m->nh + k];
            if (v < -1e-4f || v > 1.0f + 1e-4f) return 0;
            s += v;
        }
        if (fabsf(s - 1.0f) > 1e-4f) return 0;
    }
    return 1;
}

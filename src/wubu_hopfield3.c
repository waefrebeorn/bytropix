/*
 * wubu_hopfield3.c -- Hopfield frontier, batch 2 (Theme IP). C11.
 */
#include "wubu_hopfield3.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float dot(const float *a, const float *b, int d)
{
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

int wubu_mem_compress_init(wubu_mem_compress_t *m, int k, int d)
{
    if (!m || k <= 0 || d <= 0) return -1;
    m->basis = (float **)malloc(sizeof(float *) * k);
    m->weights = (float *)calloc(k, sizeof(float));
    if (!m->basis || !m->weights) return -1;
    for (int i = 0; i < k; i++)
        m->basis[i] = (float *)calloc(d, sizeof(float));
    m->k = k; m->d = d; m->n = 0;
    return 0;
}

int wubu_mem_compress_add(wubu_mem_compress_t *m, const float *pat)
{
    if (!m || !pat) return -1;
    int d = m->d;
    /* Gram-Schmidt against the existing basis */
    float *r = (float *)malloc(sizeof(float) * d);
    if (!r) return -1;
    memcpy(r, pat, sizeof(float) * d);
    for (int i = 0; i < m->k && i < m->n; i++) {
        float coef = dot(r, m->basis[i], d) / (dot(m->basis[i], m->basis[i], d) + 1e-9f);
        for (int j = 0; j < d; j++) r[j] -= coef * m->basis[i][j];
    }
    float nrm = sqrtf(dot(r, r, d));
    int slot = (m->n < m->k) ? m->n : (m->n % m->k);
    if (nrm > 1e-6f) {
        for (int j = 0; j < d; j++) m->basis[slot][j] = r[j] / nrm;
        m->weights[slot] = m->weights[slot] * 0.9f + nrm * 0.1f;
    }
    m->n++;
    free(r);
    return 0;
}

int wubu_mem_compress_recall(const wubu_mem_compress_t *m, const float *cue,
                             float *out)
{
    if (!m || !cue || !out) return -1;
    int d = m->d, k = (m->n < m->k) ? m->n : m->k;
    for (int j = 0; j < d; j++) out[j] = 0;
    for (int i = 0; i < k; i++) {
        float coef = dot(cue, m->basis[i], d) * m->weights[i];
        for (int j = 0; j < d; j++) out[j] += coef * m->basis[i][j];
    }
    return 0;
}

void wubu_mem_compress_free(wubu_mem_compress_t *m)
{
    if (!m) return;
    for (int i = 0; i < m->k; i++) free(m->basis[i]);
    free(m->basis); free(m->weights);
    memset(m, 0, sizeof(*m));
}

float wubu_mem_spectral_overlap(const float *cue, const float **bank,
                                int n, int d)
{
    if (!cue || !bank || n <= 0) return 0;
    float cn = sqrtf(dot(cue, cue, d)) + 1e-9f;
    float best = 0;
    for (int i = 0; i < n; i++) {
        float b = dot(cue, bank[i], d) / cn;
        if (b > best) best = b;
    }
    return best;
}

int wubu_mem_dedup(const float **bank, int n, int d, const float *pat,
                   float tol)
{
    if (!bank || !pat) return -1;
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < d; j++) { float e = bank[i][j] - pat[j]; s += e * e; }
        if (sqrtf(s) < tol) return i;
    }
    return -1;
}

int wubu_mem_read_t(const float **bank, int n, int d, const float *cue,
                    float beta, float *out)
{
    if (!bank || !cue || !out || n <= 0) return -1;
    float mx = -1e30f;
    for (int i = 0; i < n; i++) {
        float s = beta * dot(cue, bank[i], d);
        if (s > mx) mx = s;
    }
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float s = beta * dot(cue, bank[i], d);
        out[i] = expf(s - mx);
        sum += out[i];
    }
    if (sum > 0) for (int i = 0; i < n; i++) out[i] /= sum;
    return 0;
}

float wubu_mem_chain(const float *last, const float *next, int d)
{
    if (!last || !next) return 0;
    return dot(last, next, d);
}

float wubu_mem_energy(const float **bank, int n, const float *cue,
                      int d, float beta)
{
    if (!bank || !cue) return 0;
    float mx = -1e30f, sum = 0;
    for (int i = 0; i < n; i++) {
        float s = beta * dot(cue, bank[i], d);
        if (s > mx) mx = s;
    }
    for (int i = 0; i < n; i++) {
        float s = beta * dot(cue, bank[i], d);
        sum += expf(s - mx);
    }
    return -(mx + logf(sum + 1e-12f));
}

int wubu_mem_corrupt(const float *pat, const float *ref, int d, float tol)
{
    if (!pat || !ref) return -1;
    float s = 0;
    for (int i = 0; i < d; i++) { float e = pat[i] - ref[i]; s += e * e; }
    return sqrtf(s) > tol ? 1 : 0;
}

int wubu_mem_prune(const float **bank, const float *utility, int n,
                   float th, int *keep, int cap)
{
    if (!bank || !utility || !keep || cap <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n && k < cap; i++)
        if (utility[i] >= th) keep[k++] = i;
    return k;
}

int wubu_mem_attn_bias(const float *pattern, int d, float scale, float *bias)
{
    if (!pattern || !bias) return -1;
    for (int i = 0; i < d; i++) bias[i] = pattern[i] * scale;
    return 0;
}

int wubu_mem_snapshot(const float **bank, int n, int d, float *buf)
{
    if (!bank || !buf) return -1;
    for (int i = 0; i < n; i++)
        memcpy(buf + i * d, bank[i], sizeof(float) * d);
    return n * d;
}

int wubu_mem_restore(float **bank, int n, int d, const float *buf)
{
    if (!bank || !buf) return -1;
    for (int i = 0; i < n; i++)
        memcpy(bank[i], buf + i * d, sizeof(float) * d);
    return 0;
}

float wubu_mem_capacity(int n_patterns, int dim)
{
    /* the exponential-capacity bound ~ alpha * P^2 (P = dim) */
    float P = (float)dim;
    return 0.1f * P * P;
}

int wubu_mem_condense(const float **bank, int n, int d, float tol,
                      float **out, int cap)
{
    if (!bank || !out || cap <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n && k < cap; i++) {
        int dup = 0;
        for (int j = 0; j < k && !dup; j++) {
            float s = 0;
            for (int t = 0; t < d; t++) { float e = bank[i][t] - out[j][t]; s += e * e; }
            if (sqrtf(s) < tol) dup = 1;
        }
        if (!dup) memcpy(out[k++], bank[i], sizeof(float) * d);
    }
    return k;
}

int wubu_mem_spectral_cleanup(wubu_mem_compress_t *m, float min_energy)
{
    if (!m) return -1;
    int k = 0;
    for (int i = 0; i < m->k; i++)
        if (m->weights[i] >= min_energy) k++;
    return k;
}

float wubu_mem_beta_tune(float beta, float recall_err, float lr)
{
    /* higher error -> sharper (raise beta); the gradient step */
    float g = recall_err > 0 ? 1.0f : -1.0f;
    beta += lr * g;
    return beta < 0.1f ? 0.1f : (beta > 50.0f ? 50.0f : beta);
}

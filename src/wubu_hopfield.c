/*
 * wubu_hopfield.c -- Modern Hopfield / associative memory (Theme IL).
 * C11, deterministic, no third-party deps.
 */
#include "wubu_hopfield.h"
#include <math.h>
#include <string.h>

int wubu_hopfield_retrieve(const float *X, int n_pat, int dim,
                           const float *xi, float beta, float *out)
{
    if (!X || !xi || !out || n_pat <= 0 || dim <= 0) return -1;
    /* softmax scores over the patterns: s_i = softmax(beta * X_i . xi) */
    float s[256];
    if (n_pat > 256) n_pat = 256;
    float m = -1e30f;
    for (int i = 0; i < n_pat; i++) {
        float dot = 0;
        for (int d = 0; d < dim; d++) dot += X[i * dim + d] * xi[d];
        s[i] = beta * dot;
        if (s[i] > m) m = s[i];
    }
    float sum = 0;
    for (int i = 0; i < n_pat; i++) {
        s[i] = expf(s[i] - m);
        sum += s[i];
    }
    for (int i = 0; i < n_pat; i++) s[i] /= sum;
    /* out = X^T s */
    for (int d = 0; d < dim; d++) {
        float v = 0;
        for (int i = 0; i < n_pat; i++) v += X[i * dim + d] * s[i];
        out[d] = v;
    }
    return 0;
}

float wubu_hopfield_beta_attention(int dim)
{
    if (dim <= 0) return 1.0f;
    return 1.0f / sqrtf((float)dim);
}

float wubu_hopfield_capacity(int dim, float alpha)
{
    if (dim <= 0 || alpha <= 0) return 1.0f;
    return expf(alpha * (float)dim);
}

int wubu_hopfield_denoise(const float *X, int n_pat, int dim,
                          const float *xi, float beta,
                          float tol, int max_iter, float *out)
{
    if (!X || !xi || !out || n_pat <= 0 || dim <= 0) return -1;
    if (tol <= 0) tol = 1e-6f;
    if (max_iter <= 0) max_iter = 32;
    float cur[256];
    if (dim > 256) dim = 256;
    memcpy(cur, xi, (size_t)dim * sizeof(float));
    int it = 0;
    for (; it < max_iter; it++) {
        float next[256];
        if (wubu_hopfield_retrieve(X, n_pat, dim, cur, beta, next) != 0)
            return -1;
        float delta = 0;
        for (int d = 0; d < dim; d++) delta += fabsf(next[d] - cur[d]);
        memcpy(cur, next, (size_t)dim * sizeof(float));
        if (delta < tol) break;
    }
    memcpy(out, cur, (size_t)dim * sizeof(float));
    return it;
}

float wubu_hopfield_decay(float weight, int age, float halflife)
{
    if (age <= 0) return weight;
    if (halflife <= 0) return 0;
    return weight * expf(-((float)age / halflife) * 0.6931471805599453f);
}

float wubu_hopfield_consolidate(float weight, float reward, float alpha)
{
    if (reward < 0) reward = 0;
    if (alpha < 0) alpha = 0;
    return weight + alpha * reward;
}

int wubu_hopfield_topk(const float *X, int n_pat, int dim,
                       const float *xi, int k, int *out_idx)
{
    if (!X || !xi || !out_idx || n_pat <= 0 || k <= 0) return -1;
    if (k > n_pat) k = n_pat;
    float scores[256];
    int idx[256];
    if (n_pat > 256) n_pat = 256;
    for (int i = 0; i < n_pat; i++) {
        float dot = 0;
        for (int d = 0; d < dim; d++) dot += X[i * dim + d] * xi[d];
        scores[i] = fabsf(dot);
        idx[i] = i;
    }
    /* insertion sort by descending score (small k) */
    for (int i = 0; i < n_pat; i++) {
        for (int j = i + 1; j < n_pat; j++) {
            if (scores[j] > scores[i]) {
                float ts = scores[i]; scores[i] = scores[j]; scores[j] = ts;
                int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
            }
        }
    }
    for (int i = 0; i < k; i++) out_idx[i] = idx[i];
    return k;
}

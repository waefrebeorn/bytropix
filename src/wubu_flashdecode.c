/*
 * wubu_flashdecode.c -- FlashDecoding-style decode attention (doc 015).
 * Self-contained C11. See header. Default chunk gives ~8 parallel KV chunks.
 */
#include "wubu_flashdecode.h"
#include <math.h>
#include <string.h>
#include <omp.h>

/* Online-softmax decode attention for one query head.
 * Reference (equivalent) math: out = sum_t softmax(q·K_t) * V_t.
 * We compute it chunk-by-chunk with running (m, l, acc) and merge. */
void wubu_flashdecode_head(const float *q, const float *Kc, const float *Vc,
                           int head_dim, int n_kv_heads, int h_kv,
                           int64_t cache_len, float scale, int chunk,
                           float *out) {
    if (cache_len <= 0) { memset(out, 0, (size_t)head_dim * sizeof(float)); return; }
    if (chunk <= 0) chunk = (int)((cache_len + 7) / 8);  /* ~8 parallel chunks */
    if (chunk < 1) chunk = 1;

    const int64_t stride = (int64_t)n_kv_heads * head_dim;
    const int64_t base = (int64_t)h_kv * head_dim;

    float m = -1e30f;     /* running max */
    float l = 0.0f;       /* running log-sum-exp (sum of exp) */
    float *acc = (float *)calloc((size_t)head_dim, sizeof(float)); /* partial V sum */
    if (!acc) { memset(out, 0, (size_t)head_dim * sizeof(float)); return; }

    for (int64_t c0 = 0; c0 < cache_len; c0 += chunk) {
        int64_t c1 = c0 + chunk; if (c1 > cache_len) c1 = cache_len;
        int64_t clen = c1 - c0;

        /* --- local reduction over this chunk --- */
        float m_c = -1e30f;
        /* dot products (parallel over positions within the chunk) */
        float *s = (float *)malloc((size_t)clen * sizeof(float));
        if (!s) { free(acc); memset(out, 0, (size_t)head_dim * sizeof(float)); return; }
        #pragma omp parallel for schedule(static)
        for (int64_t t = c0; t < c1; t++) {
            const float *kt = Kc + t * stride + base;
            float d = 0.0f;
            for (int i = 0; i < head_dim; i++) d += q[i] * kt[i];
            float sc = d * scale;
            s[t - c0] = sc;
            #pragma omp critical
            { if (sc > m_c) m_c = sc; }
        }
        /* second pass: exp + weighted V */
        float l_c = 0.0f;
        float *acc_c = (float *)calloc((size_t)head_dim, sizeof(float));
        if (!acc_c) { free(s); free(acc); memset(out, 0, (size_t)head_dim * sizeof(float)); return; }
        for (int64_t t = c0; t < c1; t++) {
            float p = expf(s[t - c0] - m_c);
            l_c += p;
            const float *vt = Vc + t * stride + base;
            for (int i = 0; i < head_dim; i++) acc_c[i] += p * vt[i];
        }
        /* --- merge chunk into running (online-softmax correction) --- */
        float m_new = (m_c > m) ? m_c : m;
        float alpha_old = expf(m - m_new);      /* correction for previous acc */
        float alpha_new = expf(m_c - m_new);    /* correction for this chunk   */
        l = alpha_old * l + alpha_new * l_c;
        for (int i = 0; i < head_dim; i++)
            acc[i] = alpha_old * acc[i] + alpha_new * acc_c[i];
        m = m_new;
        free(s); free(acc_c);
    }
    float inv_l = (l > 0) ? 1.0f / l : 0.0f;
    for (int i = 0; i < head_dim; i++) out[i] = acc[i] * inv_l;
    free(acc);
}

void wubu_flashdecode_all(const float *Q, const float *Kc, const float *Vc,
                           int head_dim, int n_q_heads, int n_kv_heads,
                           int64_t cache_len, float scale, int chunk,
                           float *out) {
    int group = n_q_heads / n_kv_heads;
    if (group < 1) group = 1;
    #pragma omp parallel for
    for (int h = 0; h < n_q_heads; h++) {
        int h_kv = h / group;
        wubu_flashdecode_head(Q + (size_t)h * head_dim, Kc, Vc,
                              head_dim, n_kv_heads, h_kv,
                              cache_len, scale, chunk, out + (size_t)h * head_dim);
    }
}

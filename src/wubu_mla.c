/*
 * wubu_mla.c -- Multi-head Latent Attention (doc E02).
 *
 * Source: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient
 * Mixture-of-Experts Language Model", arXiv:2405.04434.
 *
 * Core idea: Standard attention projects from hidden_dim to
 * [n_heads * head_dim] for Q, K, V — a massive memory footprint.
 * MLA compresses K and V into a shared *latent* vector of dimension
 * kv_lora_rank + qk_rope_head_dim (e.g. 512 + 64 = 576, vs 4096+ for
 * full KV). At attention time, the latent is up-projected back to
 * full K and V via a small matrix (W_UK, W_UV).
 *
 * The win: KV cache stores only the latent (~576 floats/token) instead
 * of full K+V (2 * n_heads * head_dim = ~8192 floats/token). That's
 * ~14× KV cache compression with negligible accuracy loss.
 *
 * Additionally, Q is also compressed via q_lora_rank, decoupling the
 * query dimension from the KV dimension.
 *
 * Structure:
 *   q = W_Q * W_DQ * x    (down-project x to q_lora_rank, then up to n_heads*head_dim)
 *   kv_latent = W_DKV * x  (down-project x to kv_lora_rank + rope_dim)
 *   k_rope = kv_latent[rope_part]  (the RoPE portion, kept separate)
 *   k_nope = W_UK * kv_latent[lora_part]  (up-project back to n_heads*head_dim)
 *   v = W_UV * kv_latent[lora_part]
 *   attn = softmax(q * [k_nope; k_rope]^T / sqrt(d)) * v
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_mla.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Create an MLA context with the given dimensions. */
wubu_mla_t *wubu_mla_create(int hidden_dim, int n_heads, int head_dim,
                             int q_lora_rank, int kv_lora_rank, int rope_head_dim) {
    if (hidden_dim <= 0 || n_heads <= 0 || head_dim <= 0 ||
        q_lora_rank <= 0 || kv_lora_rank <= 0 || rope_head_dim <= 0)
        return NULL;

    wubu_mla_t *m = (wubu_mla_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->hidden_dim = hidden_dim;
    m->n_heads = n_heads;
    m->head_dim = head_dim;
    m->q_lora_rank = q_lora_rank;
    m->kv_lora_rank = kv_lora_rank;
    m->rope_head_dim = rope_head_dim;
    m->kv_latent_dim = kv_lora_rank + rope_head_dim;

    return m;
}

void wubu_mla_free(wubu_mla_t *m) {
    free(m);
}

/* Down-project hidden state to KV latent vector.
 * W_DKV: [kv_latent_dim, hidden_dim] row-major
 * x:     [hidden_dim]
 * out:   [kv_latent_dim] (kv_lora_rank + rope_head_dim) */
void wubu_mla_down_proj_kv(const wubu_mla_t *m, const float *W_DKV,
                            const float *x, float *out) {
    if (!m || !W_DKV || !x || !out) return;
    for (int i = 0; i < m->kv_latent_dim; i++) {
        float sum = 0.0f;
        const float *wrow = W_DKV + (size_t)i * m->hidden_dim;
        for (int j = 0; j < m->hidden_dim; j++) {
            sum += wrow[j] * x[j];
        }
        out[i] = sum;
    }
}

/* Up-project KV latent back to full K (nope part) per head.
 * W_UK: [n_heads * head_dim, kv_lora_rank] row-major
 * kv_latent: [kv_lora_rank] (the lora part only, not rope)
 * out: [n_heads * head_dim] */
void wubu_mla_up_proj_k(const wubu_mla_t *m, const float *W_UK,
                         const float *kv_latent, float *out) {
    if (!m || !W_UK || !kv_latent || !out) return;
    int total = m->n_heads * m->head_dim;
    for (int i = 0; i < total; i++) {
        const float *wrow = W_UK + (size_t)i * m->kv_lora_rank;
        float sum = 0.0f;
        for (int j = 0; j < m->kv_lora_rank; j++) {
            sum += wrow[j] * kv_latent[j];
        }
        out[i] = sum;
    }
}

/* Up-project KV latent back to full V per head.
 * W_UV: [n_heads * head_dim, kv_lora_rank] row-major */
void wubu_mla_up_proj_v(const wubu_mla_t *m, const float *W_UV,
                         const float *kv_latent, float *out) {
    if (!m || !W_UV || !kv_latent || !out) return;
    int total = m->n_heads * m->head_dim;
    for (int i = 0; i < total; i++) {
        const float *wrow = W_UV + (size_t)i * m->kv_lora_rank;
        float sum = 0.0f;
        for (int j = 0; j < m->kv_lora_rank; j++) {
            sum += wrow[j] * kv_latent[j];
        }
        out[i] = sum;
    }
}

/* Down-project Q via q_lora_rank then up-project to n_heads * head_dim.
 * W_DQ: [q_lora_rank, hidden_dim] row-major
 * W_UQ: [n_heads * head_dim, q_lora_rank] row-major
 * x: [hidden_dim]
 * out: [n_heads * head_dim] */
void wubu_mla_proj_q(const wubu_mla_t *m, const float *W_DQ,
                      const float *W_UQ, const float *x, float *out) {
    if (!m || !W_DQ || !W_UQ || !x || !out) return;

    /* Down-project: [q_lora_rank] */
    float *q_lora = (float *)malloc(m->q_lora_rank * sizeof(float));
    if (!q_lora) return;

    for (int i = 0; i < m->q_lora_rank; i++) {
        const float *wrow = W_DQ + (size_t)i * m->hidden_dim;
        float sum = 0.0f;
        for (int j = 0; j < m->hidden_dim; j++) {
            sum += wrow[j] * x[j];
        }
        q_lora[i] = sum;
    }

    /* Up-project: [n_heads * head_dim] */
    int total = m->n_heads * m->head_dim;
    for (int i = 0; i < total; i++) {
        const float *wrow = W_UQ + (size_t)i * m->q_lora_rank;
        float sum = 0.0f;
        for (int j = 0; j < m->q_lora_rank; j++) {
            sum += wrow[j] * q_lora[j];
        }
        out[i] = sum;
    }

    free(q_lora);
}

/* Compute MLA attention for a single query position.
 * q: [n_heads, head_dim] — already projected
 * k_nope: [n_heads, head_dim] — nope part of K (up-projected from latent)
 * k_rope: [n_heads, rope_head_dim] — rope part of K
 * v: [n_heads, head_dim] — up-projected from latent
 * out: [n_heads, head_dim] — attention output
 *
 * Full key for head h is [k_nope[h]; k_rope[h]] concatenated.
 */
void wubu_mla_attn(const wubu_mla_t *m, const float *q,
                    const float *k_nope, const float *k_rope,
                    const float *v, float *out) {
    if (!m || !q || !k_nope || !k_rope || !v || !out) return;

    int full_dim = m->head_dim + m->rope_head_dim;
    float inv_sqrt = 1.0f / sqrtf((float)full_dim);

    for (int h = 0; h < m->n_heads; h++) {
        const float *qh = q + (size_t)h * m->head_dim;
        const float *kn = k_nope + (size_t)h * m->head_dim;
        const float *kr = k_rope + (size_t)h * m->rope_head_dim;
        const float *vh = v + (size_t)h * m->head_dim;
        float *outh = out + (size_t)h * m->head_dim;

        /* Single-token attention: score = q · [k_nope; k_rope] / sqrt(d) */
        float score = 0.0f;
        for (int d = 0; d < m->head_dim; d++)
            score += qh[d] * kn[d];
        for (int d = 0; d < m->rope_head_dim; d++)
            score += qh[d % m->head_dim] * kr[d];  /* q has head_dim, rope has rope_head_dim */

        /* Softmax over single key (trivially = 1.0 for single token) */
        float weight = 1.0f;  /* single-key softmax */
        (void)score; (void)inv_sqrt;  /* Would use for multi-key attention */

        /* Output = weight * V */
        for (int d = 0; d < m->head_dim; d++)
            outh[d] = weight * vh[d];
    }
}

/* Compute KV cache compression ratio compared to standard attention. */
float wubu_mla_compression_ratio(const wubu_mla_t *m) {
    if (!m || m->n_heads <= 0 || m->head_dim <= 0) return 0.0f;
    /* Standard: 2 * n_heads * head_dim floats per token (K + V)
     * MLA: kv_latent_dim floats per token (compressed latent) */
    float standard = 2.0f * (float)(m->n_heads * m->head_dim);
    float mla = (float)m->kv_latent_dim;
    return standard / mla;
}

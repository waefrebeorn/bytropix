/*
 * wubu_mla.h -- Multi-head Latent Attention (doc E02).
 *
 * Source: DeepSeek-V2, arXiv:2405.04434.
 *
 * Compresses KV cache by projecting K/V into a shared latent vector,
 * reducing per-token KV storage from 2*n_heads*head_dim to
 * kv_lora_rank + rope_head_dim (~14x compression).
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_MLA_H
#define WUBU_MLA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int hidden_dim;      /* model hidden dimension (e.g. 4096) */
    int n_heads;          /* number of attention heads */
    int head_dim;         /* dimension per head (e.g. 128) */
    int q_lora_rank;      /* Q down-projection rank (e.g. 1536) */
    int kv_lora_rank;     /* KV down-projection rank (e.g. 512) */
    int rope_head_dim;    /* RoPE dimension per head (e.g. 64) */
    int kv_latent_dim;    /* kv_lora_rank + rope_head_dim (cached) */
} wubu_mla_t;

/* Create/destroy. */
wubu_mla_t *wubu_mla_create(int hidden_dim, int n_heads, int head_dim,
                             int q_lora_rank, int kv_lora_rank, int rope_head_dim);
void wubu_mla_free(wubu_mla_t *m);

/* Down-project hidden state to KV latent vector. */
void wubu_mla_down_proj_kv(const wubu_mla_t *m, const float *W_DKV,
                            const float *x, float *out);

/* Up-project KV latent to full K (nope part). */
void wubu_mla_up_proj_k(const wubu_mla_t *m, const float *W_UK,
                         const float *kv_latent, float *out);

/* Up-project KV latent to full V. */
void wubu_mla_up_proj_v(const wubu_mla_t *m, const float *W_UV,
                         const float *kv_latent, float *out);

/* Project Q: down via q_lora_rank, then up to n_heads*head_dim. */
void wubu_mla_proj_q(const wubu_mla_t *m, const float *W_DQ,
                      const float *W_UQ, const float *x, float *out);

/* Compute single-token MLA attention. */
void wubu_mla_attn(const wubu_mla_t *m, const float *q,
                    const float *k_nope, const float *k_rope,
                    const float *v, float *out);

/* Compute KV compression ratio vs standard attention. */
float wubu_mla_compression_ratio(const wubu_mla_t *m);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MLA_H */

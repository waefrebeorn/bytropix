/*
 * wubu_flash_prefill.h -- Fused tiled prefill attention (doc H01).
 *
 * Source: Dao et al., "FlashAttention", NeurIPS 2022.
 *
 * FlashAttention tiles the prefill attention computation into blocks
 * that fit in cache, computing partial softmax via the online-softmax
 * (running max + log-sum-exp) reduction. O(S) memory instead of O(S²).
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_FLASH_PREFILL_H
#define WUBU_FLASH_PREFILL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fused prefill attention with online softmax.
 *
 * Q, K, V: [n_heads, seq_len, head_dim] row-major
 * out:     [n_heads, seq_len, head_dim] row-major
 * B_tc:    tile size (0 = auto-select 64)
 *
 * Computes: out[h,i,:] = softmax_j(Q[i]·K[j]^T / sqrt(d)) · V[j]
 * with O(seq_len * head_dim) memory (no S×S materialization).
 */
void wubu_flash_prefill_attn(const float *Q, const float *K, const float *V,
                               float *out, int n_heads, int seq_len, int head_dim,
                               int B_tc);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FLASH_PREFILL_H */

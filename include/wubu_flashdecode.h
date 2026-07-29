/*
 * wubu_flashdecode.h -- FlashDecoding-style decode attention (doc 015).
 *
 * The decode-attention bottleneck is the serial read over the whole KV cache
 * for each query head (O(cache_len) KV loads, one position at a time).
 * FlashDecoding adds a PARALLEL dimension over the KV sequence: split K/V into
 * chunks, compute a *partial* online-softmax (running max + log-sum-exp + partial
 * V-weighted sum) per chunk in parallel, then merge the chunk partials with the
 * standard online-softmax correction. Mathematically identical to full softmax
 * (the merge lemma: max(a,b) + renormalize preserves the result).
 *
 * The K/V cache layout mirrors wubu_gqa_forward's kv_cache_read_head:
 *   element for (token t, kv-head h) at byte offset (t * n_kv_heads + h) * head_dim
 * K/V are given as flat F32 arrays (one contiguous cache, head_dim per slot).
 */
#ifndef WUBU_FLASHDECODE_H
#define WUBU_FLASHDECODE_H

#include <stdint.h>

/* Decode attention for a SINGLE query head.
 * q          : [head_dim] query
 * Kc, Vc     : flat cache [cache_len * n_kv_heads * head_dim], F32
 * n_kv_heads : number of KV heads (for grouped-query indexing)
 * h_kv       : which KV head this query head attends to
 * cache_len  : number of K/V positions to attend over
 * scale      : 1/sqrt(head_dim) (pre-applied to dot products)
 * chunk      : KV chunk size (parallelism granularity); <=0 -> cache_len/8
 * out        : [head_dim] result (overwritten)
 */
void wubu_flashdecode_head(const float *q, const float *Kc, const float *Vc,
                           int head_dim, int n_kv_heads, int h_kv,
                           int64_t cache_len, float scale, int chunk,
                           float *out);

/* Convenience: attend all n_q_heads query heads (each routed to its GQA KV
 * head) in parallel. Q is [n_q_heads * head_dim]; writes out[n_q_heads*head_dim].
 * group_size = n_q_heads / n_kv_heads. */
void wubu_flashdecode_all(const float *Q, const float *Kc, const float *Vc,
                           int head_dim, int n_q_heads, int n_kv_heads,
                           int64_t cache_len, float scale, int chunk,
                           float *out);

#endif /* WUBU_FLASHDECODE_H */

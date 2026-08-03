/*
 * wubu_evict2026c.h -- the KV-eviction frontier, final (IO). C11.
 * Agnostic: an eviction-policy table + the tracker ops. Covers
 * H2O heavy-hitter retention, StreamingLLM sinks, KVQuant,
 * accumulated-attention tracking, KV-reconstruction importance,
 * outlier storage, reconstruction-based importance, LSH thresholds,
 * proxy-token adaptation, eviction-aware RoPE, correctness audits,
 * block-paged eviction, batched-request eviction, KVQuant kernels,
 * semantic eviction, speculative-decode interaction, attention
 * scaling, 1M+ context retention, hybrid eviction, multimodal.
 */
#ifndef WUBU_EVICT2026C_H
#define WUBU_EVICT2026C_H

#include <stdint.h>

/* IO01: H2O heavy-hitter token retention. */
int wubu_evictc_h2o(const float *attention, int n, float th, int *keep);

/* IO02: StreamingLLM attention-sink keep. */
int wubu_evictc_sink(int *tokens, int n, int sink, int keep_n);

/* IO10: KVQuant 3-bit + outlier split. */
int wubu_evictc_kvquant(const float *kv, int n, int32_t *quant, int32_t *outlier);

/* IO13: accumulated-attention tracker (O(1) update). */
int wubu_evictc_track(float *running_sum, int i, float new_val);

/* IO17: KV-reconstruction importance. */
int wubu_evictc_recon_importance(const float *orig, const float *recon, int n, float th);

/* IO24: outlier channel sparse store. */
int wubu_evictc_outlier(const float *kv, int n, float th, int *sparse_idx);

/* IO35: reconstruction-based importance (page granularity). */
int wubu_evictc_page_import(const float *pages, int n_pages, float th);

/* IO38: LSH threshold tuning. */
float wubu_evictc_lsh_thresh(float correlation, float base);

/* IO39: proxy-token count adaptation. */
int wubu_evictc_proxy(int prompt_len, int *proxy_count);

/* IO43: eviction-aware RoPE. */
int wubu_evictc_rope_reencode(const int *positions, int n, int shift, int *new_pos);

/* IO44: compressed-cache correctness audit. */
int wubu_evictc_audit(float perplexity, float th);

/* IO47: block-paged eviction. */
int wubu_evictc_block_paged(const int *kv_table, int n_blocks, int block_size, int *to_evict);

/* IO49: batched-request eviction. */
int wubu_evictc_batch(const float *criticality, int n, float th, int *evicted);

/* IO51: KVQuant 3-bit encode/decode. */
int wubu_evictc_kvquant_kernel(const float *kv, int n, int8_t *out);

/* IO56: semantic eviction via ANN. */
int wubu_evictc_ann(const float *sim_scores, int n, float th, int *keep);

/* IO57: speculative-decode cache interaction. */
int wubu_evictc_spec(const float *draft_scores, int n, float th, int *retain);

/* IO58: eviction-aware attention scaling. */
int wubu_evictc_scaling(float *attn, int n, float factor);

/* IO62: 1M+ context cost-modeled retention. */
int wubu_evictc_1m(long ctx_size, long threshold);

/* IO65: hybrid attention eviction. */
int wubu_evictc_hybrid(const float *attn_scores, const float *ssm_scores, int n, float th);

/* IO66: multimodal eviction. */
int wubu_evictc_mm(const float *vision_scores, const float *text_scores, int n, float th);

#endif
/*
 * wubu_token2.h -- the tokenization frontier, complete (IT). C11.
 * Agnostic: a tokenizer-state + data tables, the caller picks the
 * policy. Covers benchmarks, remapping, entropy monitors, merging,
 * caches, normalization, regularization, fallbacks, streaming,
 * growth, augmentation, byte-level RoPE, multi-token prediction,
 * trie indexes, serialization, diff, coverage, watermarks.
 */
#ifndef WUBU_TOKEN2_H
#define WUBU_TOKEN2_H

#include <stdint.h>

/* IT21: multilingual token-efficiency benchmark score. */
float wubu_tok2_bench(long tokens, long chars);

/* IT22: token-id remapping (vocab swap). */
int wubu_tok2_remap(const int *old_ids, int n, const int *map, int *out);

/* IT23: entropy monitor (distribution-shift detection). */
int wubu_tok2_shift(const uint32_t *counts, int n, const uint32_t *ref,
                    float th);

/* IT24: BPE merge-pair scoring. */
float wubu_tok2_pair_score(long pair_count, long a_count, long b_count);

/* IT25: tokenizer cache. */
typedef struct { uint64_t key; int n; int valid; } wubu_tok2_cache_t;
int wubu_tok2_cache_get(wubu_tok2_cache_t *c, uint64_t key, int fallback);
void wubu_tok2_cache_put(wubu_tok2_cache_t *c, uint64_t key, int n);

/* IT26: Unicode normalization guard (the byte-class check). */
int wubu_tok2_norm_guard(const unsigned char *s, int len, int allow_nfd);

/* IT27: token-length regularization. */
int wubu_tok2_len_reg(long growth, long cap);

/* IT28: byte-fallback decode. */
int wubu_tok2_byte_fallback(const unsigned char *s, int len, int *ok);

/* IT31: the pair-frequency table update. */
int wubu_tok2_pair_freq(uint32_t *freq, int n, int a, int b);

/* IT32: embedded-token density. */
float wubu_tok2_density(long tokens, long embedding_bytes);

/* IT33: determinism check. */
int wubu_tok2_deterministic(const uint32_t *a, const uint32_t *b, int n);

/* IT34: the token-budget planner. */
long wubu_tok2_budget_plan(long prompt_len, float growth, long max_budget);

/* IT35: subword-entity alignment. */
int wubu_tok2_entity_align(int start, int end, int n_tokens);

/* IT36: streaming encode (the incremental state). */
typedef struct { uint32_t acc; int pending; } wubu_tok2_stream_t;
int wubu_tok2_stream(wubu_tok2_stream_t *s, unsigned char byte, uint32_t *tok);

/* IT38: token dropout for robustness. */
int wubu_tok2_dropout(const uint32_t *ids, int n, float p, uint32_t *out);

/* IT40: byte-level RoPE. */
float wubu_tok2_byte_rope(float x, int byte_pos, float theta);

/* IT42: multi-token prediction targets. */
int wubu_tok2_next_n(const uint32_t *ids, int n, int k, uint32_t *out);

/* IT43: token-trie prefix index. */
int wubu_tok2_trie(const uint32_t *ids, int n, uint32_t prefix, int *depth);

/* IT44: tokenizer serialization (the portable blob). */
int wubu_tok2_serialize(const uint32_t *vocab, int n, uint8_t *buf, int cap);

/* IT46: byte-pair health monitor. */
float wubu_tok2_pair_health(long merges, long total);

/* IT47: token-efficiency-aware prefill (skip the redundant). */
long wubu_tok2_skip_redundant(long tokens, float redundancy);

/* IT48: tokenizer-free fallback (byte-level ids). */
int wubu_tok2_fallback(const unsigned char *s, int len, uint32_t *out);

/* IT49: vocabulary coverage (OOV rate). */
float wubu_tok2_coverage(long in_vocab, long total);

/* IT50: token-boundary watermark. */
int wubu_tok2_watermark(const uint32_t *ids, int n, uint32_t key);

#endif

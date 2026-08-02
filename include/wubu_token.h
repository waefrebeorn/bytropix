/*
 * wubu_token.h -- tokenization frontier (Theme IT). C11.
 * Bit-level BPE, tokenizer-free UTF-8 embeddings, byte-entropy merges,
 * lexical density, token-merge cache, vocab pruning, roundtrip audit,
 * multi-script, token entropy coding, adaptive vocab, efficiency
 * metrics, embedding compression, telemetry, versioning, OOV policy.
 */
#ifndef WUBU_TOKEN_H
#define WUBU_TOKEN_H

#include <stdint.h>
#include <stddef.h>

/* IT01: bit-level BPE -- the token cost below the byte boundary. */
int wubu_tok_bit_bpe_cost(int byte_len, int bits_per_symbol);

/* IT02: tokenizer-free UTF-8 byte embedding (the direct byte vector). */
int wubu_tok_utf8_embed(const unsigned char *s, int len, float *out, int d);

/* IT04: byte-entropy-aware merge score. */
float wubu_tok_entropy_merge(const uint32_t *counts, int n);

/* IT05: lexical density -> effective window. */
int wubu_tok_density_window(int tokens, float density, int max_window);

/* IT06: token-merge cache (frequent-path memoization). */
typedef struct { uint32_t key; int n; int valid; } wubu_tok_cache_t;
int wubu_tok_cache_get(wubu_tok_cache_t *c, uint32_t key, int fallback);
void wubu_tok_cache_put(wubu_tok_cache_t *c, uint32_t key, int n);

/* IT07: vocab pruning -- remap the used ids. */
int wubu_tok_prune(const int *used, int vocab, int *remap, int *kept);

/* IT08: roundtrip audit (encode/decode fidelity). */
int wubu_tok_roundtrip(const unsigned char *s, int len,
                       const unsigned char *back, int back_len);

/* IT12: tokens-per-information-unit metric. */
float wubu_tok_efficiency(int tokens, int info_bits);

/* IT16: OOV handling policy. */
int wubu_tok_oov(int token_id, int vocab, int fallback_id);

/* IT10: token-level entropy coding (post-token). */
size_t wubu_tok_entropy_size(const uint32_t *counts, int n, long total);

#endif

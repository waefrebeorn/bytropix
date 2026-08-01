/*
 * wubu_mm_kv.h -- Positional KV integration: assemble multimodal prefix.
 */
#ifndef WUBU_MM_KV_H
#define WUBU_MM_KV_H
#include "wubu_mm_adapter.h"

#define WUBU_MM_KV_MAX_PREFIX 128  /* vision + audio token IDs */

typedef struct {
    int token_ids[WUBU_MM_KV_MAX_PREFIX];  /* pseudo-token IDs */
    int n_tokens;  /* number of prefix tokens */
} wubu_mm_kv_prefix_t;

/* Assemble the multimodal token prefix: vision (65) + optional audio (n_a)
   → contiguous token ID array prepended to text tokens. Returns total prefix. */
int wubu_mm_kv_assemble(const wubu_mm_adapter_result_t *adapter_result,
                        const int *audio_tok_ids, int n_audio_tokens,
                        wubu_mm_kv_prefix_t *prefix);

/* Safety check: prefix must not exceed context window. */
int wubu_mm_kv_safe(const wubu_mm_kv_prefix_t *prefix, int ctx_size);

#endif
/*
 * wubu_mm_kv.c -- Positional KV integration. C11.
 *
 * Convergence (KV-cache prefix injection 7-hop: position IDs, context budgeting,
 * EAMM safety):
 *   - CC05: assembles the multimodal token prefix (vision + audio token IDs)
 *     into a contiguous array that gets prepended to the text token stream.
 *     This is where EAMM is forbidden at 512K ctx: we enforce a hard cap
 *     (ctx_size) to prevent runaway context growth. The prefix tokens
 *     occupy KV positions [0, n_tokens), text occupies [n_tokens, ...).
 */
#include "wubu_mm_kv.h"
#include <string.h>

int wubu_mm_kv_assemble(const wubu_mm_adapter_result_t *res,
                        const int *audio_ids, int n_audio,
                        wubu_mm_kv_prefix_t *prefix) {
    if (!res || !prefix || n_audio < 0 ||
        res->n_vision_tokens < 0 ||
        res->n_vision_tokens > WUBU_IMGENC_N_TOKENS) return -1;
    memset(prefix, 0, sizeof(*prefix));
    int n = 0;
    /* Vision tokens first (positions [0, 65)) */
    for (int i = 0; i < res->n_vision_tokens && n < WUBU_MM_KV_MAX_PREFIX; i++)
        prefix->token_ids[n++] = res->vision_tok_ids[i];
    /* Audio tokens after (positions [65, 65+n_a)) */
    if (audio_ids && n_audio > 0) {
        for (int i = 0; i < n_audio && n < WUBU_MM_KV_MAX_PREFIX; i++)
            prefix->token_ids[n++] = audio_ids[i];
    }
    prefix->n_tokens = n;
    return n;
}

int wubu_mm_kv_safe(const wubu_mm_kv_prefix_t *prefix, int ctx_size) {
    if (!prefix) return 0;
    /* No EAMM: prefix must leave room for at least 256 text tokens */
    if (prefix->n_tokens + 256 > ctx_size) return 0;
    return 1;
}

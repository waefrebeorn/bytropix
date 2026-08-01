/*
 * wubu_mm_adapter.c -- Multimodal adapter: align → quantize → KV token IDs. C11.
 *
 * Convergence (BLIP-2/Q-Former adapter 7-hop: cross-attention, projection alignment):
 *   - CC04/CC06: projects vision/audio embeddings into text space (via
 *     wubu_mm_align), then quantizes them to pseudo-token IDs via nearest-
 *     neighbor lookup against a fixed visual vocabulary of 256 anchor vectors.
 *     These pseudo-token IDs can be prepended to the text token stream and
 *     fed directly to gen_text's tokenizer/KV-cache pipeline — no model
 *     changes needed. The model "sees" the image as a prefix of ~65 tokens.
 */
#include "wubu_mm_adapter.h"
#include <math.h>
#include <string.h>

static float lcg_randf(unsigned *seed) {
    *seed = (*seed * 1103515245U + 12345U) & 0x7fffffff;
    return (float)((double)*seed / (double)0x7fffffff);
}

int wubu_mmvocab_init(wubu_mmvocab_t *v, unsigned seed) {
    if (!v) return -1;
    unsigned s = seed ? seed : 777;
    for (int i = 0; i < WUBU_MM_VOCAB_SIZE; i++)
        for (int j = 0; j < WUBU_MM_TEXT_DIM; j++)
            v->anchors[i][j] = (lcg_randf(&s) - 0.5f) * 2.0f;
    v->init = 1;
    return 0;
}

int wubu_mmvocab_quantize(const wubu_mmvocab_t *v, const float *tokens,
                          int n_tokens, int *out_ids) {
    if (!v || !v->init || !tokens || !out_ids || n_tokens < 0) return -1;
    for (int t = 0; t < n_tokens; t++) {
        const float *tok = &tokens[t * WUBU_MM_TEXT_DIM];
        int best_id = 0;
        /* Compute distance to first anchor */
        float best_dist = 0.0f;
        for (int j = 0; j < WUBU_MM_TEXT_DIM; j++) {
            float d = tok[j] - v->anchors[0][j];
            best_dist += d * d;
        }
        for (int k = 1; k < WUBU_MM_VOCAB_SIZE; k++) {
            float dist = 0.0f;
            for (int j = 0; j < WUBU_MM_TEXT_DIM; j++) {
                float d = tok[j] - v->anchors[k][j];
                dist += d * d;
            }
            if (dist < best_dist) { best_dist = dist; best_id = k; }
        }
        out_ids[t] = best_id;
    }
    return 0;
}

int wubu_mm_adapter_run(const wubu_mm_align_t *align,
                        const wubu_mmvocab_t *vocab,
                        const float *vision_tokens,
                        wubu_mm_adapter_result_t *out) {
    if (!align || !vocab || !vision_tokens || !out) return -1;
    memset(out, 0, sizeof(*out));
    /* 1. Align vision tokens to text space */
    float aligned[WUBU_MM_ADAPTER_MAX_VISION_TOKENS * WUBU_MM_TEXT_DIM];
    if (wubu_mm_align_vision(align, vision_tokens, aligned) != 0) return -1;
    /* 2. Quantize to pseudo-token IDs via nearest-neighbor */
    out->n_vision_tokens = WUBU_IMGENC_N_TOKENS;
    if (wubu_mmvocab_quantize(vocab, aligned, out->n_vision_tokens,
                              out->vision_tok_ids) != 0) return -1;
    return 0;
}

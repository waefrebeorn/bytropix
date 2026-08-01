/*
 * wubu_mm_adapter.h -- Multimodal adapter: align into KV-cache token stream.
 */
#ifndef WUBU_MM_ADAPTER_H
#define WUBU_MM_ADAPTER_H
#include "wubu_imgenc.h"
#include "wubu_mm_align.h"

#define WUBU_MM_ADAPTER_VISION_POS 0   /* vision tokens start at KV position 0 */
#define WUBU_MM_ADAPTER_MAX_VISION_TOKENS WUBU_IMGENC_N_TOKENS
#define WUBU_MM_ADAPTER_MAX_AUDIO_TOKENS 32

/* Result: pseudo-token IDs derived from nearest-neighbor lookup of aligned
   embeddings against a small "visual vocab" of anchor centroids. */
typedef struct {
    int n_vision_tokens;
    int n_audio_tokens;
    int vision_tok_ids[WUBU_MM_ADAPTER_MAX_VISION_TOKENS];
    int audio_tok_ids[WUBU_MM_ADAPTER_MAX_AUDIO_TOKENS];
} wubu_mm_adapter_result_t;

/* Visual vocab: N anchors in 512-dim text space. */
#define WUBU_MM_VOCAB_SIZE 256
typedef struct {
    float anchors[WUBU_MM_VOCAB_SIZE][WUBU_MM_TEXT_DIM];
    int init;
} wubu_mmvocab_t;

int wubu_mmvocab_init(wubu_mmvocab_t *v, unsigned seed);
/* Nearest-neighbor assign: each aligned token → nearest anchor id. */
int wubu_mmvocab_quantize(const wubu_mmvocab_t *v, const float *aligned_tokens,
                          int n_tokens, int *out_ids);

/* Full pipeline: vision tokens → aligned → quantized token IDs */
int wubu_mm_adapter_run(const wubu_mm_align_t *align,
                        const wubu_mmvocab_t *vocab,
                        const float *vision_tokens,  /* 65 × 128 */
                        wubu_mm_adapter_result_t *out);

#endif
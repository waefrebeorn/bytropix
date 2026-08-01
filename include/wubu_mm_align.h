/*
 * wubu_mm_align.h -- Cross-modal alignment: vision/audio → text space (CC03).
 */
#ifndef WUBU_MM_ALIGN_H
#define WUBU_MM_ALIGN_H

#define WUBU_MM_VISION_DIM 128   /* matches wubu_vision embed_dim */
#define WUBU_MM_AUDIO_DIM 40    /* mel bins → token dim */
#define WUBU_MM_TEXT_DIM 512    /* target text embedding dim (KV-cache aligned) */
#define WUBU_MM_VISION_NTOKENS 65  /* CLA + 64 patches */

typedef struct {
    float vision_proj[WUBU_MM_VISION_DIM][WUBU_MM_TEXT_DIM]; /* 128×512 */
    float vision_bias[WUBU_MM_TEXT_DIM];
    float audio_proj[WUBU_MM_AUDIO_DIM][WUBU_MM_TEXT_DIM];   /* 40×512 */
    float audio_bias[WUBU_MM_TEXT_DIM];
    int init;
} wubu_mm_align_t;

int wubu_mm_align_init(wubu_mm_align_t *m, unsigned seed);
/* Project vision tokens (65 × 128) → text space (65 × 512). */
int wubu_mm_align_vision(const wubu_mm_align_t *m, const float *vision_tokens, float *out);
/* Project audio tokens (n_frames × 40) → text space (n_frames × 512). */
int wubu_mm_align_audio(const wubu_mm_align_t *m, const float *audio_mel, int n_frames, float *out);

#endif
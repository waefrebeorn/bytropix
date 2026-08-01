/*
 * wubu_mm_align.c -- Cross-modal alignment: vision/audio → text embedding space. C11.
 *
 * Convergence (CLIP/SigLIP cross-modal 7-hop: projection alignment):
 *   - CC03: learned linear projection maps vision/audio features into the
 *     same dim as the text model's embedding space (512). This is the
 *     alignment step — CLIP does it via contrastive training; at home
 *     we use random init (alignment quality improves with training, but
 *     the mechanism is identical). The projection is just matmul + bias.
 */
#include "wubu_mm_align.h"
#include "wubu_imgenc.h"
#include <math.h>
#include <string.h>

static float lcg_randf(unsigned *seed) {
    *seed = (*seed * 1103515245U + 12345U) & 0x7fffffff;
    return (float)((double)*seed / (double)0x7fffffff);
}

int wubu_mm_align_init(wubu_mm_align_t *m, unsigned seed) {
    if (!m) return -1;
    unsigned s = seed ? seed : 888;
    float scale = sqrtf(2.0f / (float)(WUBU_MM_VISION_DIM + WUBU_MM_TEXT_DIM));
    for (int i = 0; i < WUBU_MM_VISION_DIM; i++)
        for (int j = 0; j < WUBU_MM_TEXT_DIM; j++)
            m->vision_proj[i][j] = (lcg_randf(&s) - 0.5f) * 2.0f * scale;
    for (int j = 0; j < WUBU_MM_TEXT_DIM; j++) m->vision_bias[j] = 0.0f;
    scale = sqrtf(2.0f / (float)(WUBU_MM_AUDIO_DIM + WUBU_MM_TEXT_DIM));
    for (int i = 0; i < WUBU_MM_AUDIO_DIM; i++)
        for (int j = 0; j < WUBU_MM_TEXT_DIM; j++)
            m->audio_proj[i][j] = (lcg_randf(&s) - 0.5f) * 2.0f * scale;
    for (int j = 0; j < WUBU_MM_TEXT_DIM; j++) m->audio_bias[j] = 0.0f;
    m->init = 1;
    return 0;
}

int wubu_mm_align_vision(const wubu_mm_align_t *m, const float *vin, float *out) {
    if (!m || !m->init || !vin || !out) return -1;
    for (int t = 0; t < WUBU_IMGENC_N_TOKENS; t++) {
        for (int j = 0; j < WUBU_MM_TEXT_DIM; j++) {
            float s = m->vision_bias[j];
            const float *tok = &vin[t * WUBU_MM_VISION_DIM];
            for (int i = 0; i < WUBU_MM_VISION_DIM; i++)
                s += tok[i] * m->vision_proj[i][j];
            out[t * WUBU_MM_TEXT_DIM + j] = s;
        }
    }
    return 0;
}

int wubu_mm_align_audio(const wubu_mm_align_t *m, const float *aud, int nframes, float *out) {
    if (!m || !m->init || !aud || !out || nframes < 0) return -1;
    for (int t = 0; t < nframes; t++) {
        for (int j = 0; j < WUBU_MM_TEXT_DIM; j++) {
            float s = m->audio_bias[j];
            const float *frm = &aud[t * WUBU_MM_AUDIO_DIM];
            for (int i = 0; i < WUBU_MM_AUDIO_DIM; i++)
                s += frm[i] * m->audio_proj[i][j];
            out[t * WUBU_MM_TEXT_DIM + j] = s;
        }
    }
    return 0;
}

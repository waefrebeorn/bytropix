/*
 * wubu_imgenc.h -- Vision encoder (ViT patch embedding from scratch). C11.
 */
#ifndef WUBU_IMGENC_H
#define WUBU_IMGENC_H

#define WUBU_IMGENC_PATCH 8
#define WUBU_IMGENC_CHANNELS 3
#define WUBU_IMGENC_IMAGE 64
#define WUBU_IMGENC_PATCH_DIM (WUBU_IMGENC_PATCH*WUBU_IMGENC_PATCH*WUBU_IMGENC_CHANNELS)
#define WUBU_IMGENC_EMBED_DIM 128
#define WUBU_IMGENC_N_PATCHES ((WUBU_IMGENC_IMAGE/WUBU_IMGENC_PATCH) * (WUBU_IMGENC_IMAGE/WUBU_IMGENC_PATCH))
#define WUBU_IMGENC_N_TOKENS (WUBU_IMGENC_N_PATCHES + 1)

typedef struct {
    float proj[WUBU_IMGENC_PATCH_DIM][WUBU_IMGENC_EMBED_DIM];
    float pos[WUBU_IMGENC_N_TOKENS][WUBU_IMGENC_EMBED_DIM];
    float cls[WUBU_IMGENC_EMBED_DIM];
    int init;
} wubu_imgenc_t;

int wubu_imgenc_init(wubu_imgenc_t *v, unsigned seed);
int wubu_imgenc_encode(const wubu_imgenc_t *v, const float *img_64x64x3, float *out_tokens);

#endif
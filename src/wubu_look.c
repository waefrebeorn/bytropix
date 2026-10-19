/*
 * wubu_look.c -- frame pair -> regions worth encoding.
 */
#include "wubu_look.h"
#include "wubu_eye.h"
#include <stdlib.h>

int wubu_look(const wubu_img *prev, const wubu_img *curr,
              int patch, int tile, int max_rois,
              wubu_look_roi *out, float *reuse_ratio)
{
    if (!curr || !curr->data || !out || max_rois <= 0) return -1;
    if (patch <= 0 || tile <= 0) return -1;
    if (reuse_ratio) *reuse_ratio = 1.0f;

    /* First frame: nothing to reuse, but no "change" either. */
    if (!prev) return 0;
    if (prev->w != curr->w || prev->h != curr->h) return -1;

    /* 1. How much of the frame is static? */
    int gw = (curr->w + patch - 1) / patch;
    int gh = (curr->h + patch - 1) / patch;
    uint8_t *mask = (uint8_t *)malloc((size_t)gw * gh);
    if (!mask) return -1;

    wubu_eye_stats st;
    if (wubu_eye_patch_mask(prev->data, curr->data, curr->w, curr->h,
                            patch, 0, 1, mask, &st) != 0) {
        free(mask);
        return -1;
    }
    if (reuse_ratio) *reuse_ratio = st.reused_ratio;

    /* 2. Where is the change dense? Build the absolute difference image. */
    wubu_img diff = {0};
    if (wubu_img_alloc(&diff, curr->w, curr->h) != 0) { free(mask); return -1; }

    for (int y = 0; y < curr->h; y++) {
        const uint8_t *a = prev->data + (size_t)y * prev->stride;
        const uint8_t *b = curr->data + (size_t)y * curr->stride;
        uint8_t *d = diff.data + (size_t)y * diff.stride;
        for (int x = 0; x < curr->w; x++) {
            int v = (int)a[x] - (int)b[x];
            d[x] = (uint8_t)(v < 0 ? -v : v);
        }
    }

    wubu_roi rois[32];
    int want = max_rois < 32 ? max_rois : 32;
    int nr = wubu_foveated_rois(&diff, tile, want, 1, rois, want);

    int n = 0;
    for (int i = 0; i < nr && n < max_rois; i++) {
        out[n].x = rois[i].x;
        out[n].y = rois[i].y;
        out[n].w = rois[i].w;
        out[n].h = rois[i].h;
        out[n].energy = rois[i].score;
        if (out[n].energy > 0.0) n++;
    }

    wubu_img_free(&diff);
    free(mask);
    return n;
}


int wubu_look_budget(const wubu_img *prev, const wubu_img *curr,
                     int patch, int tile, int max_rois,
                     int token_budget, int tokens_per_tile,
                     wubu_look_roi *out, float *reuse_ratio,
                     int *tokens_used)
{
    if (token_budget <= 0 || tokens_per_tile <= 0 || !tokens_used) return -1;
    if (max_rois <= 0 || !out) return -1;

    int n = wubu_look(prev, curr, patch, tile, max_rois, out, reuse_ratio);
    if (n < 0) return -1;

    /* ROI list arrives ranked by energy, so take the best that fit. */
    int used = 0, kept = 0;
    for (int i = 0; i < n; i++) {
        if (used + tokens_per_tile > token_budget) break;
        out[kept++] = out[i];
        used += tokens_per_tile;
    }
    *tokens_used = used;
    return kept;
}

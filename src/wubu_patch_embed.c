/*
 * wubu_patch_embed.c — ViT patch embedding implementation.
 *
 * C11, self-contained, opaque struct.
 *
 * Implements the standard ViT patch embedding:
 *   1. Extract p x p patches from the image grid (non-overlapping)
 *   2. Flatten each patch to a vector of length (in_channels * patch^2)
 *   3. Project to hidden_dim via learned linear: out = W * x + b
 *   4. Add positional embeddings if present
 *
 * Also provides a conv2d-equivalent path (strided patch extraction)
 * that produces the same result when the linear projection W is
 * reshaped to a conv kernel.
 */
#include "wubu_patch_embed.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct wubu_patch_embed_ctx {
    int patch_size;
    int in_channels;
    int hidden_dim;
    int image_size;
    int num_patches;          /* (image_size / patch_size)^2 */
    int patch_pixels;         /* patch_size^2 */
    int patch_flat_dim;       /* in_channels * patch_size^2 */

    const float *proj_w;      /* [hidden_dim, patch_flat_dim] row-major */
    const float *proj_b;      /* [hidden_dim] or NULL */
    const float *pos_emb;     /* [num_patches, hidden_dim] or NULL */
};

wubu_patch_embed_t *wubu_patch_embed_init(
    int patch_size, int in_channels, int hidden_dim, int image_size,
    const float *proj_w, const float *proj_b,
    const float *pos_emb)
{
    if (patch_size <= 0 || in_channels <= 0 || hidden_dim <= 0 ||
        image_size <= 0 || !proj_w)
        return NULL;
    if (image_size % patch_size != 0)
        return NULL;

    wubu_patch_embed_t *ctx = (wubu_patch_embed_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->patch_size    = patch_size;
    ctx->in_channels   = in_channels;
    ctx->hidden_dim    = hidden_dim;
    ctx->image_size    = image_size;
    ctx->proj_w        = proj_w;
    ctx->proj_b        = proj_b;
    ctx->pos_emb       = pos_emb;
    ctx->num_patches   = (image_size / patch_size) * (image_size / patch_size);
    ctx->patch_pixels  = patch_size * patch_size;
    ctx->patch_flat_dim = in_channels * ctx->patch_pixels;

    return ctx;
}

int wubu_patch_embed_num_patches(const wubu_patch_embed_t *ctx)
{
    return ctx ? ctx->num_patches : 0;
}

int wubu_patch_embed_forward(const wubu_patch_embed_t *ctx,
                             const float *img, float *out)
{
    if (!ctx || !img || !out) return 0;
    int ps    = ctx->patch_size;
    int ch    = ctx->in_channels;
    int H     = ctx->image_size;
    int W     = ctx->image_size;
    int hid   = ctx->hidden_dim;
    int np    = ctx->num_patches;
    int pfd   = ctx->patch_flat_dim;  /* ch * ps * ps */

    /* Workspace for one flattened patch. */
    float *patch_flat = (float *)malloc((size_t)pfd * sizeof(float));
    if (!patch_flat) return 0;

    int patches_y = H / ps;
    int patches_x = W / ps;

    for (int py = 0; py < patches_y; py++) {
        for (int px = 0; px < patches_x; px++) {
            int patch_idx = py * patches_x + px;
            float *pf = patch_flat;

            /* Flatten patch: [ch, ps, ps] -> [pfd] */
            for (int c = 0; c < ch; c++) {
                const float *row = img + (size_t)c * H * W;
                for (int dy = 0; dy < ps; dy++) {
                    const float *scan = row + (size_t)(py * ps + dy) * W + (px * ps);
                    for (int dx = 0; dx < ps; dx++) {
                        *pf++ = scan[dx];
                    }
                }
            }

            /* Project: out[patch] = proj_w * patch_flat + proj_b */
            float *out_row = out + (size_t)patch_idx * hid;
            for (int o = 0; o < hid; o++) {
                const float *wrow = ctx->proj_w + (size_t)o * pfd;
                float acc = ctx->proj_b ? ctx->proj_b[o] : 0.0f;
                for (int i = 0; i < pfd; i++) {
                    acc += wrow[i] * patch_flat[i];
                }
                out_row[o] = acc;
            }

            /* Add positional embedding if present */
            if (ctx->pos_emb) {
                const float *pe = ctx->pos_emb + (size_t)patch_idx * hid;
                for (int o = 0; o < hid; o++) out_row[o] += pe[o];
            }
        }
    }

    free(patch_flat);
    return np;
}

/*
 * Conv2d-equivalent path: assumes patches are already extracted as a grid
 * of [num_patches, patch_flat_dim] patches (e.g., by a stride-16 conv).
 * This avoids re-extraction and directly applies the projection.
 */
int wubu_patch_embed_forward_grid(const wubu_patch_embed_t *ctx,
                                  const float *patches, float *out)
{
    if (!ctx || !patches || !out) return 0;
    int hid = ctx->hidden_dim;
    int np  = ctx->num_patches;
    int pfd = ctx->patch_flat_dim;

    for (int p = 0; p < np; p++) {
        const float *pf = patches + (size_t)p * pfd;
        float *out_row = out + (size_t)p * hid;
        for (int o = 0; o < hid; o++) {
            const float *wrow = ctx->proj_w + (size_t)o * pfd;
            float acc = ctx->proj_b ? ctx->proj_b[o] : 0.0f;
            for (int i = 0; i < pfd; i++) acc += wrow[i] * pf[i];
            out_row[o] = acc;
        }
        if (ctx->pos_emb) {
            const float *pe = ctx->pos_emb + (size_t)p * hid;
            for (int o = 0; o < hid; o++) out_row[o] += pe[o];
        }
    }
    return np;
}

void wubu_patch_embed_close(wubu_patch_embed_t *ctx)
{
    if (!ctx) return;
    free(ctx);
}

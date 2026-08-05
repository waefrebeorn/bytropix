#ifndef WUBU_PATCH_EMBED_H
#define WUBU_PATCH_EMBED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_patch_embed.h — ViT patch embedding for vision models.
 *
 * Extracts non-overlapping patches (p x p) from an image tensor
 * and projects them to hidden dimension via a learned linear layer.
 * Supports both flatten+linear and conv2d-equivalent path.
 *
 * C11, opaque struct, minimal includes. Self-contained.
 */

typedef struct wubu_patch_embed_ctx wubu_patch_embed_t;

/*
 * Initialize patch embedding.
 *   patch_size:      spatial patch side (e.g. 16)
 *   in_channels:     image channels (e.g. 3 for RGB)
 *   hidden_dim:      projection output dimension
 *   image_size:      image spatial size (height=width, square image)
 *   proj_w:          [hidden_dim, in_channels * patch_size^2] row-major
 *   proj_b:          [hidden_dim] or NULL
 *   pos_emb:         [num_patches, hidden_dim] positional embeddings, or NULL
 * Returns opaque ctx, or NULL on error. Caller does NOT own proj_w/b/pos.
 */
wubu_patch_embed_t *wubu_patch_embed_init(
    int patch_size, int in_channels, int hidden_dim, int image_size,
    const float *proj_w, const float *proj_b,
    const float *pos_emb);

/*
 * Extract patches from image and project.
 *   img:  [in_channels, H, W] row-major, H=image_size, W=image_size
 *   out:  [num_patches, hidden_dim] — caller allocates
 * Returns num_patches, or 0 on error.
 */
int wubu_patch_embed_forward(const wubu_patch_embed_t *ctx,
                             const float *img, float *out);

/*
 * Same but assumes image is already a 2D grid of patches (NHWC patch grid).
 * Used for conv2d-equivalent path where patches are extracted via strided conv.
 */
int wubu_patch_embed_forward_grid(const wubu_patch_embed_t *ctx,
                                  const float *patches, float *out);

void wubu_patch_embed_close(wubu_patch_embed_t *ctx);

/* Query: number of patches (image_size / patch_size)^2 */
int wubu_patch_embed_num_patches(const wubu_patch_embed_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PATCH_EMBED_H */

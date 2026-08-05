#ifndef WUBU_MMROPE_H
#define WUBU_MMROPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_mmrope.h — 3D Multimodal RoPE for MiniMax H3.
 *
 * H3-Omni-Transformer uses 3D MM-RoPE to encode positional
 * relationships across temporal and two spatial dimensions (t, h, w).
 *
 * Self-contained C11. Opaque struct + minimal includes.
 */

typedef struct wubu_mmrope_ctx wubu_mmrope_t;

/*
 * Initialize 3D MM-RoPE.
 *   head_dim:       attention head dimension (must be divisible by 3)
 *   theta_t:        temporal RoPE base (e.g. 10000.0)
 *   theta_h:        spatial height RoPE base
 *   theta_w:        spatial width RoPE base
 *   num_tok_t:      number of temporal positions
 *   num_tok_h:      number of spatial height positions
 *   num_tok_w:      number of spatial width positions
 *   pos_t:          [seq_len] temporal position indices
 *   pos_h:          [seq_len] height position indices
 *   pos_w:          [seq_len] width position indices
 * Returns opaque ctx, or NULL on error.
 */
wubu_mmrope_t *wubu_mmrope_init(int head_dim,
                                float theta_t, float theta_h, float theta_w,
                                int num_tok_t, int num_tok_h, int num_tok_w,
                                const int *pos_t, const int *pos_h, const int *pos_w);

/*
 * Apply 3D MM-RoPE to a query or key tensor.
 *   qk:        [seq_len, n_heads, head_dim] — modified in place
 *   seq_len:   sequence length (= len of pos arrays)
 *   n_heads:   number of attention heads
 * The head_dim is split into 3 equal segments: (t, h, w) RoPE.
 */
void wubu_mmrope_apply(const wubu_mmrope_t *ctx,
                       float *qk, int seq_len, int n_heads);

void wubu_mmrope_close(wubu_mmrope_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MMROPE_H */

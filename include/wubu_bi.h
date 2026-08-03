/* wubu_bi.h -- Block Importance oracle (ShortGPT, arXiv:2403.03853).
 *
 * BI(l) = mean hidden-state norm change at layer l.
 * Low BI = redundant (shrink candidate). High BI = critical (grow candidate).
 * The amoeba uses these to decide what to morph.
 */
#ifndef WUBU_BI_H
#define WUBU_BI_H

#include "wubu.h"

/* Compute per-layer hidden-state norms (proxy for hidden-state change).
 * Caller frees(*out_norms). Returns 0 on success, -1 on error. */
int wubu_bi_norms(const wubu_model_t *m, const wubu_buf_t *b,
                  const uint16_t *tokens, int n_tokens,
                  float **out_norms, int *out_n_layers);

/* Block Importance: mean absolute difference of successive norms.
 * Lower BI = more redundant. Caller frees(*out_bis). */
int wubu_bi_compute(const wubu_model_t *m, const wubu_buf_t *b,
                    const uint16_t *tokens, int n_tokens,
                    float **out_bis, int *out_n_layers);

/* Rank layers by BI ascending (most redundant first). Caller frees(*out_rank). */
int wubu_bi_rank(const float *bis, int n_layers, int **out_rank);

/* Shrink candidate: the layer with the lowest BI (below threshold).
 * Returns layer index, or -1 if no shrink candidate. */
int wubu_bi_shrink_candidate(const float *bis, int n_layers, float threshold);

/* Grow candidate: the layer with the highest BI (above threshold).
 * Returns layer index, or -1 if no grow candidate. */
int wubu_bi_grow_candidate(const float *bis, int n_layers, float threshold);

#endif /* WUBU_BI_H */

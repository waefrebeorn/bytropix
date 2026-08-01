/*
 * wubu_sparse_attn.h -- Block-sparse attention pattern generators
 * (L11 NSA / L12 MoBA). Pure emitters of keep-masks.
 */
#ifndef WUBU_SPARSE_ATTN_H
#define WUBU_SPARSE_ATTN_H

#include <stdint.h>

/* L11 NSA: per-query top-k block keep-mask over nblk x nblk blocks. */
int wubu_block_sparse_mask(const float *scores, int nblk, int k, uint8_t *mask);

/* L12 MoBA: per-query top-k segment flags over nq x nseg segments. */
int wubu_moba_topk(const float *scores, int nq, int nseg, int k, uint8_t *flags);

#endif /* WUBU_SPARSE_ATTN_H */

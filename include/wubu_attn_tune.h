/*
 * wubu_attn_tune.h -- Attention/dispatch auto-tuners (L06/N19/O11).
 */
#ifndef WUBU_ATTN_TUNE_H
#define WUBU_ATTN_TUNE_H

/* L06 Quest blockwise top-k: select top-k block indices by score. */
int wubu_quest_topk(const float *scores, int nb, int k, int *out_ids);

/* N19 Adaptive chunk: prefill/decode chunk in [min_c, max_c], capped at seq. */
int wubu_adaptive_chunk(int seq, int batch, int min_c, int max_c);

/* O11 Split-K auto-tune: split-K in [1, Kmax] for ~target_tiles. */
int wubu_splitk_tune(int tokens, int target_tiles, int Kmax);

#endif /* WUBU_ATTN_TUNE_H */

/*
 * wubu_parallel_spec.h -- Parallel speculative decoding (V01-V04) + length-gen PE (W01-W03).
 */
#ifndef WUBU_PARALLEL_SPEC_H
#define WUBU_PARALLEL_SPEC_H

/* V01 EAGLE-3 feature drafting (argmax of feature score). */
int wubu_eagle3_draft(const float *feat_score, int n, int *drafted);
/* V02 P-EAGLE parallel verify. */
int wubu_peagle_verify(const int *drafts, const char *match, int K, int *accepted);
/* V03 tree-attention parent array. */
int wubu_tree_attn_parents(int n, int *parents);
/* V04 Kangaroo double-early-exit accept. */
int wubu_kangaroo_accept(int shallow_match, int deep_match);
/* W01 NoPE flag. */
int wubu_nope_enabled(void);
/* W02 ALiBi distance bias. */
int wubu_alibi_bias(float *bias, int n, int d, float slope);
/* W03 FFN-first ordering flag. */
int wubu_ffn_first_enabled(void);

#endif /* WUBU_PARALLEL_SPEC_H */

/*
 * wubu_eval_qat.h -- Long-context eval harness (Z01-Z05) + QAT (AA01-AA04).
 */
#ifndef WUBU_EVAL_QAT_H
#define WUBU_EVAL_QAT_H

/* Z01 NIAH multi-needle injection positions. */
int wubu_niah_inject(int len, int nneedle, int *pos);
/* Z02 RULER retrieval. */
int wubu_ruler_retrieve(const int *key, const int *val, int n, int qkey, int *out);
/* Z03 RULER multi-hop. */
int wubu_ruler_multihop(const int *key, const int *next, int n, int start, int depth, int *out);
/* Z04 RULER aggregation. */
int wubu_ruler_aggregate(const int *ctx, int n, int target);
/* Z05 synthetic haystack. */
int wubu_haystack_gen(int len, int *tokens);
/* AA01 fake-quant. */
float wubu_fakequant(float x, float step, float mn, float mx);
/* AA02 QAT STE. */
int wubu_qat_ste(float x, float step, float mn, float mx, float *out_q, int *grad_pass);
/* AA03 per-channel dequant. */
float wubu_dequant_pc(int q, float scale, int zero);
/* AA04 noise injection. */
float wubu_noise_inject(float w, unsigned seed, float amp);

#endif /* WUBU_EVAL_QAT_H */

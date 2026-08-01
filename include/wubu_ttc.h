/*
 * wubu_ttc.h -- Test-time-compute + multi-agent KV (Q08/Q15/Q20/R01/R03).
 */
#ifndef WUBU_TTC_H
#define WUBU_TTC_H

/* Q08 PolyKV coherence gate (shared agent KV prefix reuse). */
int wubu_polykv_coherent(const float *sumA, const float *sumB, int d, float thr);

/* Q15 HotPrefix hotness priority. */
float wubu_hotprefix_priority(int freq, int age, float halflife);

/* Q20 test-time budget allocator (reasoning steps). */
int wubu_ttc_budget_steps(int budget, float cost);

/* R01 inference-time scaling controller. */
float wubu_scaling_factor(float q, float smin, float smax);

/* R03 CATTS contrastive adaptive token scaling. */
int wubu_catts_tokens(int draft_len, float conf);

#endif /* WUBU_TTC_H */

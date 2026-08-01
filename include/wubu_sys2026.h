/*
 * wubu_sys2026.h -- System/scheduling 2026 KV methods (Q12/Q13/Q14/Q16/Q17/Q18/R02).
 */
#ifndef WUBU_SYS2026_H
#define WUBU_SYS2026_H

/* Q12 TARDIS GPU spill count when over gpu_cap. */
int wubu_tardis_spill(const int *ages, int n, int gpu_cap);

/* Q13 KVDrive tier: 0 GPU / 1 DRAM / 2 SSD. */
int wubu_kvdrive_tier(int age, int dram_thr, int ssd_thr);

/* Q14 ScoutAttention CPU-precompute eligibility. */
int wubu_scout_eligible(int Lcur, int cur, int a);

/* Q16 AlignedServe max shared-prefix length across requests. */
int wubu_aligned_lcp(const int *a, int alen, const int *reqs, int nreq, int rlen);

/* Q17 CoDec prefix-shared decode eligibility. */
int wubu_codec_share(int lenA, int lenB, int min_share);

/* Q18 SparKV overhead-aware KV load decision. */
int wubu_sparkv_load(float access_p, float benefit, float cost);

/* R02 agentic context-efficiency gate. */
int wubu_agentic_ctx(float ctx_cost, float budget);

#endif /* WUBU_SYS2026_H */

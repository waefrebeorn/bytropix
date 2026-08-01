/*
 * wubu_kv2026c.h -- Remaining 2026 KV methods (Q11/Q19/R04/R05).
 */
#ifndef WUBU_KV2026C_H
#define WUBU_KV2026C_H

/* Q11 DASH-KV hash-based token scheduling. */
int wubu_dashkv_schedule(const float *keys, int n, int d, const float *scores,
                         int nbuckets, int *out_bucket);

/* Q19 HeteroCache per-head bits in [bmin,bmax]. */
int wubu_hetero_bits(const float *entropy, int nheads, int bmin, int bmax, int *out);

/* R04 reasoning redundancy profiler (mean redundancy of reasoning tokens). */
float wubu_redundancy_profile(const float *redundancy, const char *is_reasoning, int n);

/* R05 multi-agent KV coherence (mean pairwise cosine). */
float wubu_multiagent_coherence(const float *sums, int n_agents, int d);

#endif /* WUBU_KV2026C_H */

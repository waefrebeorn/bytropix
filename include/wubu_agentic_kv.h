/*
 * wubu_agentic_kv.h -- Hybrid scheduler + multimodal/agentic KV (S06/U01/U02/U03/U04/U05).
 */
#ifndef WUBU_AGENTIC_KV_H
#define WUBU_AGENTIC_KV_H

/* S06 hybrid: 1 = recurrent, 0 = full attention. */
int wubu_hybrid_is_recurrent(int L, int period);
/* U01 shared-KV source layer. */
int wubu_shared_kv_source(int L, int off);
/* U02 CSA mean-pool compress. */
int wubu_csa_compress(const float *keys, int n, int d, int group, float *out);
/* U03 vision-block hash. */
unsigned wubu_vision_hash(const int *tok, int n);
/* U04 LOOK-M keep top-k vision ids. */
int wubu_lookm_keep(const float *score, int n, int keep, int *out);
/* U05 agentic compaction mask (1=keep). */
int wubu_agentic_compact(const float *saliency, int n, int keep, char *out);

#endif /* WUBU_AGENTIC_KV_H */

/*
 * wubu_kv2026.h -- Fresh 2026 KV-cache methods (Q02/Q03/Q07/Q09/Q10).
 */
#ifndef WUBU_KV2026_H
#define WUBU_KV2026_H

/* Q02 ChunkKV chunk-level eviction: keep top-`keep` chunks by mean score. */
int wubu_chunkkv_evict(const float *scores, int n, int nchunks, int keep, int *out);

/* Q03 KVzip query-agnostic importance = attention variance across heads. */
int wubu_kvzip_importance(const float *attn, int n, int nheads, float *out);

/* Q07 LAVa (layer,head) dynamic keep budget in [1,cap]. */
int wubu_lava_budget(float e_layer, float e_head, int cap);

/* Q09 FreeKV speculative top-k retrieval. */
int wubu_freekv_topk(const float *scores, int n, int k, int *out);

/* Q10 TTKV temporal-tiered placement: 0 HOT / 1 WARM / 2 COLD. */
int wubu_ttkv_tier(int age, int warm_thr, int cold_thr);

#endif /* WUBU_KV2026_H */

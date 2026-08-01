/*
 * wubu_kv2026b.h -- More 2026 KV-cache methods (Q01/Q04/Q05/Q06).
 */
#ifndef WUBU_KV2026B_H
#define WUBU_KV2026B_H

/* Q01 CentroidKV: keep one representative (nearest centroid) per cluster. */
int wubu_centroidkv(const float *keys, int n, int d, int k, int *out);

/* Q04 R-KV: per-token redundancy = max cosine to another token. */
int wubu_rkv_redundancy(const float *keys, int n, int d, float *out);

/* Q05 OBCache: Hessian-saliency proxy = |grad|^2 per token. */
int wubu_obcache_saliency(const float *grad, int n, int d, float *out);

/* Q06 KeyDiff: keep `keep` most-distinct tokens by cosine. */
int wubu_keydiff_evict(const float *keys, int n, int d, int keep, int *out);

#endif /* WUBU_KV2026B_H */

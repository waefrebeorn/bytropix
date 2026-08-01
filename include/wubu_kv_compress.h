/*
 * wubu_kv_compress.h -- Attention-score KV compression (L07 SnapKV / L09 CIA).
 */
#ifndef WUBU_KV_COMPRESS_H
#define WUBU_KV_COMPRESS_H

#define WUBU_KV_COMPRESS_MAX 8192

/* L09 CIA: keep top keep_frac of slots by cumulative attention score. */
int wubu_kv_keep_top_score(const float *scores, int n, float keep_frac,
                           int *out_ids);

/* L07 SnapKV: keep top keep_clusters clusters (by mean attention). */
int wubu_kv_keep_clusters(const float *scores, int n, int nclusters,
                         int keep_clusters, int *out_ids);

/* L08 PyramidKV: adjust keep_frac by layer depth (shallow keeps more). */
float wubu_pyramid_keep(float base_keep, float depth_frac, float pyramid);

#endif /* WUBU_KV_COMPRESS_H */

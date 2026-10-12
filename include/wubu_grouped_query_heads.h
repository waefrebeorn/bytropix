#ifndef WUBU_GROUPED_QUERY_HEADS_H
#define WUBU_GROUPED_QUERY_HEADS_H
/* wubu_grouped_query_heads.h — Grouped-Query Attention head mapping (BRAIN #234).
 *
 * GQA shares one KV head across a group of `n_q / n_kv` query heads, cutting
 * KV-cache memory vs MHA. The forward (wubu_gqa_kv_head) maps each query head
 * to its KV head. Covered by test_gqa with measured assertions.
 */
#include <stddef.h>

/* Map query head `q` (0..n_q-1) to its KV head (0..n_kv-1) under grouping.
 * Requires n_kv>0, n_q % n_kv == 0, 0<=q<n_q. Returns the KV head index, or
 * -1 on invalid args. */
int wubu_gqa_kv_head(int q, int n_q, int n_kv);

/* Number of query heads that share one KV head (the group size). Returns
 * n_q/n_kv, or -1 if not divisible / invalid. */
int wubu_gqa_group_size(int n_q, int n_kv);

#endif

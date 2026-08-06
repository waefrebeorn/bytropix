/* wubu_kv_coherence_diag.h — post-forward KV coherence diagnose (Phase 7)
 *
 * The diagnostic operator. After each model forward pass over the
 * KV namespace, this module:
 *
 *   1. Iterates every encoded file in /kv/in/<path>
 *   2. Computes the coherence score for each (via wubu_kv_embedding)
 *   3. Feeds the per-file utilization (attention_mass) to wubu_kv_shrink
 *   4. Feeds the coherence scores to wubu_grow_kv for diagnose
 *   5. Runs grow/shrink sweeps to adapt the KV namespace
 *
 * This is the "immune system" cycle — diagnose, decide, mutate:
 *
 *   forward → attention → coherence → grow/shrink → next forward
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 7 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_COHERENCE_DIAG_H
#define WUBU_KV_COHERENCE_DIAG_H

#include <stddef.h>
#include "wubu_kv_embedding.h"
#include "wubu_coherence_reward.h"
#include "wubu_grow_kv.h"
#include "wubu_kv_shrink.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The diagnostic state: tracks per-file coherence over iterations.
 * This is the "triple-DA" oracle from the universal manifold design
 * (docs/universal-manifold-design.md layer 5):
 *
 *   D = Detect    (coherence below threshold → under-coherent file)
 *   A = Analyze   (why — low mass? high entropy? low consistency?)
 *   A = Act       (grow toward it, shrink away from dead regions)
 */
typedef struct wubu_kv_diag wubu_kv_diag_t;

/* Create the diagnostic operator.
 * All handles are caller-owned (created elsewhere, passed in). */
wubu_kv_diag_t *wubu_kv_diag_create(wubu_kv_embedding_t *kv,
                                     wubu_grow_kv_t *grow,
                                     wubu_kv_shrink_t *shrink);

/* The post-forward diagnostic cycle:
 *
 * n_files / paths: the KV paths read during this forward pass
 * n_files / attention: [n_files][n_query_tokens_i][n_context_tokens_i]
 *   attention[i] is the flattened attention row for file i's query
 * n_query / n_context: per-file query/context token counts
 * context_start / context_len: per-file context region in the KV
 * query_start / query_len: per-file query token positions
 *
 * This computes coherence for each file, feeds utilization to shrink,
 * feeds scores to grow, and runs the grow/shrink sweeps.
 *
 * Returns a wubu_reward_result_t (caller frees with wubu_reward_result_free).
 * Also populates out_summary (if non-NULL) with grow/shrink counts. */
typedef struct {
    int n_grown;    /* KV blocks grown this cycle */
    int n_pruned;   /* KV regions pruned this cycle */
    int n_under;    /* files below coherence threshold */
    float mean_score; /* mean coherence across all files */
} wubu_kv_diag_summary_t;

int wubu_kv_diag_cycle(wubu_kv_diag_t *diag,
                        const char **paths,
                        const float **attention,
                        const size_t *n_query_tokens,
                        const size_t *n_context_tokens,
                        const size_t *context_start,
                        const size_t *context_len,
                        const size_t *query_start,
                        const size_t *query_len,
                        int n_files,
                        wubu_reward_result_t *out_reward,
                        wubu_kv_diag_summary_t *out_summary);

/* Just compute coherence summaries without triggering grow/shrink.
 * Returns the per-file coherence in out (n_files entries). */
int wubu_kv_diag_measure(wubu_kv_diag_t *diag,
                          const char **paths,
                          const float **attention,
                          const size_t *n_query_tokens,
                          const size_t *n_context_tokens,
                          const size_t *context_start,
                          const size_t *context_len,
                          const size_t *query_start,
                          const size_t *query_len,
                          int n_files,
                          wubu_coherence_t *out_scores);

/* Free the diagnostic operator. Does NOT free the sub-operators. */
void wubu_kv_diag_free(wubu_kv_diag_t *diag);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_COHERENCE_DIAG_H */

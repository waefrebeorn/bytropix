/* wubu_coherence_reward.h — the coherence-based reward signal for KV-FS training
 *
 * The training reward is: how well did the model attend to the files
 * it read from /kv/in/? The coherence score (from wubu_kv_embedding)
 * becomes the RL reward — files that the model understands get reinforced.
 *
 * This module wraps the coherence computation with:
 *   - Batched coherence across a training batch
 *   - Temporal consistency (comparing attention across consecutive forwards)
 *   - Reward shaping (coherence delta × loss reduction)
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 2 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_COHERENCE_REWARD_H
#define WUBU_COHERENCE_REWARD_H

#include <stddef.h>
#include "wubu_kv_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A per-file coherence snapshot for a batch */
typedef struct {
    const char *path;       /* /kv/in/<relpath> */
    wubu_coherence_t coh;   /* the coherence measurement */
} wubu_reward_entry_t;

/* A batched reward result */
typedef struct {
    wubu_reward_entry_t *entries;  /* per-file */
    int                  n_entries;
    float                mean_score; /* mean coherence across all files */
    float                reward;     /* composite RL reward [0, 1] */
} wubu_reward_result_t;

/* Compute the coherence reward for a batch.
 *
 * fs_dataset is the dataset (for file list + KV region lookups).
 * attention_per_file maps each file path to its attention row.
 *   (In practice, the executor produces attention from the forward pass
 *    over /kv/in/<path> tokens.)
 *
 * n_files: number of files in the batch.
 * For each file i:
 *   - attention[i] is [n_query_tokens[i]][n_context_tokens]
 *     (flattened row-major)
 *   - context_start[i], context_len[i] locate the file in context
 *   - query_start[i], query_len[i] locate the query tokens
 *
 * Returns 0 on success with result in *out, -1 on error.
 * Caller frees with wubu_reward_result_free(). */
int wubu_coherence_reward_compute(const wubu_kv_embedding_t *kv,
                                   const char **paths,
                                   const float **attention,
                                   const size_t *n_query_tokens,
                                   const size_t *n_context_tokens,
                                   const size_t *context_start,
                                   const size_t *context_len,
                                   const size_t *query_start,
                                   const size_t *query_len,
                                   int n_files,
                                   wubu_reward_result_t *out);

/* Free the reward result. */
void wubu_reward_result_free(wubu_reward_result_t *r);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_COHERENCE_REWARD_H */

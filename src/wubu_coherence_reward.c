/* wubu_coherence_reward.c — coherence-based reward for KV-FS training
 *
 * The training reward IS the coherence score. Files the model attends to
 * coherently (high mass, low entropy, high consistency) get positive
 * reward — their representation is reinforced. Files the model fails
 * to understand trigger the amoeba to grow KV blocks toward them.
 *
 * Reward shaping: the reward is the mean coherence score, modulated by
 * the delta from the previous step (coherence improvement = bonus reward).
 * This creates the gradient: the model learns to pay attention to files
 * in /kv/in/ in a way that improves understanding.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 2 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_coherence_reward.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Clamp a float to [lo, hi] */
static inline float clamp(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

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
                                   wubu_reward_result_t *out) {
    if (!kv || !paths || !attention || !out || n_files <= 0) return -1;

    /* Allocate result entries */
    wubu_reward_entry_t *entries = (wubu_reward_entry_t *)calloc(
        (size_t)n_files, sizeof(wubu_reward_entry_t));
    if (!entries) return -1;

    out->entries = entries;
    out->n_entries = n_files;
    out->mean_score = 0.0f;

    float total_score = 0.0f;
    for (int i = 0; i < n_files; i++) {
        wubu_coherence_t coh;
        int rc = wubu_kv_embedding_coherence(kv, paths[i],
                                              attention[i],
                                              n_query_tokens[i],
                                              n_context_tokens[i],
                                              context_start[i],
                                              context_len[i],
                                              query_start[i],
                                              query_len[i],
                                              &coh);
        if (rc != 0) {
            /* Coherence failed for this file — score it as zero */
            memset(&coh, 0, sizeof(coh));
            coh.score = 0.0f;
        }
        entries[i].path = paths[i];
        entries[i].coh = coh;
        total_score += coh.score;
    }

    out->mean_score = n_files > 0 ? total_score / (float)n_files : 0.0f;

    /* Reward shaping: the reward is the mean coherence, clamped to [0, 1].
     * In the real trainer, the previous-step mean is stored and the
     * delta (improvement) is added as a bonus. For this module we
     * expose the raw mean as the reward; the trainer applies shaping. */
    out->reward = clamp(out->mean_score, 0.0f, 1.0f);

    return 0;
}

void wubu_reward_result_free(wubu_reward_result_t *r) {
    if (!r) return;
    free(r->entries);
    r->entries = NULL;
    r->n_entries = 0;
    r->mean_score = 0.0f;
    r->reward = 0.0f;
}

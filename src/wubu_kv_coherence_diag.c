/* wubu_kv_coherence_diag.c — post-forward KV coherence diagnose
 *
 * The diagnostic operator: after each forward pass over /kv/in/,
 * it computes coherence for every file, feeds utilization to the
 * shrink operator, feeds scores to the grow operator, and runs
 * the grow/shrink sweeps. This closes the metabolism loop:
 *
 *   forward → attention → coherence → grow/shrink → next forward
 *
 * The Triple-DA oracle (docs/universal-manifold-design.md layer 5):
 *   D = Detect: coherence < threshold → file is under-coherent
 *   A = Analyze: breakdown into mass/entropy/consistency
 *   A = Act:     grow toward under-coherent, shrink dead regions
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 7 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_coherence_diag.h"
#include <stdlib.h>
#include <string.h>

struct wubu_kv_diag {
    wubu_kv_embedding_t *kv;
    wubu_grow_kv_t      *grow;
    wubu_kv_shrink_t    *shrink;
};

wubu_kv_diag_t *wubu_kv_diag_create(wubu_kv_embedding_t *kv,
                                     wubu_grow_kv_t *grow,
                                     wubu_kv_shrink_t *shrink) {
    if (!kv) return NULL;
    wubu_kv_diag_t *d = (wubu_kv_diag_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->kv = kv;
    d->grow = grow;
    d->shrink = shrink;
    return d;
}

void wubu_kv_diag_free(wubu_kv_diag_t *d) {
    if (!d) return;
    /* Sub-operators are caller-owned — don't free them. */
    free(d);
}

/* Measure-only: compute coherence for each file */
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
                          wubu_coherence_t *out_scores) {
    if (!diag || !paths || !attention || !out_scores || n_files <= 0) return -1;
    for (int i = 0; i < n_files; i++) {
        int rc = wubu_kv_embedding_coherence(diag->kv, paths[i],
                                              attention[i],
                                              n_query_tokens[i],
                                              n_context_tokens[i],
                                              context_start[i],
                                              context_len[i],
                                              query_start[i],
                                              query_len[i],
                                              &out_scores[i]);
        if (rc != 0) {
            /* Coherence failed for this file — zero it out */
            memset(&out_scores[i], 0, sizeof(wubu_coherence_t));
        }
    }
    return 0;
}

/* The full diagnose cycle: measure → shrink.feed → grow.diagnose →
 * grow.sweep → shrink.sweep */
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
                        wubu_kv_diag_summary_t *out_summary) {
    if (!diag || !paths || !attention || n_files <= 0) return -1;

    /* Step 1: Compute coherence + reward for all files */
    wubu_reward_result_t reward;
    int rc = wubu_coherence_reward_compute(diag->kv, paths, attention,
                                            n_query_tokens, n_context_tokens,
                                            context_start, context_len,
                                            query_start, query_len,
                                            n_files, &reward);
    if (rc != 0) {
        if (out_reward) memset(out_reward, 0, sizeof(*out_reward));
        if (out_summary) memset(out_summary, 0, sizeof(*out_summary));
        return -1;
    }

    /* Step 2: Feed utilization (attention_mass) to shrink operator */
    if (diag->shrink) {
        /* Extract attention_mass as utilization per file */
        float *util = (float *)malloc((size_t)n_files * sizeof(float));
        if (util) {
            for (int i = 0; i < n_files; i++)
                util[i] = reward.entries[i].coh.attention_mass;
            wubu_kv_shrink_feed(diag->shrink, paths, util, n_files);
            free(util);
        }
    }

    /* Step 3: Feed coherence scores to grow operator's diagnose */
    int n_under = 0;
    if (diag->grow) {
        /* Extract scores as the coherence signal for grow */
        float *scores = (float *)malloc((size_t)n_files * sizeof(float));
        if (scores) {
            for (int i = 0; i < n_files; i++)
                scores[i] = reward.entries[i].coh.score;
            n_under = wubu_grow_kv_diagnose(diag->grow, paths, scores, n_files);
            free(scores);
        }
    }

    /* Step 4: Run grow sweep (grow toward under-coherent files) */
    int n_grown = 0;
    if (diag->grow)
        n_grown = wubu_grow_kv_grow(diag->grow, n_under);

    /* Step 5: Run shrink sweep (prune cold regions) */
    int n_pruned = 0;
    if (diag->shrink)
        n_pruned = wubu_kv_shrink_sweep(diag->shrink, NULL);

    /* Populate summary */
    if (out_summary) {
        out_summary->n_grown = n_grown;
        out_summary->n_pruned = n_pruned;
        out_summary->n_under = n_under;
        out_summary->mean_score = reward.mean_score;
    }
    if (out_reward)
        *out_reward = reward;
    else
        wubu_reward_result_free(&reward);

    return 0;
}

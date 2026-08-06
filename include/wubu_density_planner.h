/* wubu_density_planner.h — information density planning (AN23-core)
 *
 * The AGI's resource allocator. Decides what data gets absorbed into
 * model weights (permanent memory) vs stays in the KV cache (context
 * memory) vs gets pruned entirely.
 *
 * Density = coherence_score / parameter_cost. High-density files
 * (high coherence, low weight cost to absorb) get promoted to
 * weights. Low-density files stay as KV context or get pruned.
 *
 * The planner runs after each coherence diagnose cycle:
 *
 *   1. RECEIVE: per-file coherence scores from wubu_kv_coherence_diag
 *   2. RANK:    sort files by density (coherence / n_tokens)
 *   3. DECIDE:  given weight budget W:
 *     - top D% → absorb into weights (promote to wubu_scalable_model)
 *     - middle → keep in KV cache at F16 tier
 *     - bottom → prune (wubu_kv_shrink) or archive to disk
 *   4. ACT:     call wubu_scalable_mark_hot / wubu_kv_tier_down / shrink
 *
 * This is the "user experience always + AGI always" guarantee:
 * the AGI keeps learning and adapting, the UX stays responsive because
 * the density planner keeps the hot path small and F32.
 *
 * Design: docs/wubu1-scalable-model-design.md §Density Planning.
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_DENSITY_PLANNER_H
#define WUBU_DENSITY_PLANNER_H

#include <stddef.h>
#include "wubu_kv_embedding.h"
#include "wubu_coherence_reward.h"
#include "wubu_model_scalable.h"
#include "wubu_kv_tiering.h"
#include "wubu_kv_shrink.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_density_planner wubu_density_planner_t;

/* Policy configuration for density planning.
 *
 * The planner operates in three bands:
 *   - ABSORB threshold: density >= absorb_threshold → promote to weights
 *   - KEEP band: density between absorb and keep thresholds → KV cache
 *   - PRUNE: density < prune_threshold → shrink/prune
 *
 * density = coherence.score / n_tokens (bits of understanding per token)
 *
 * Memory budgets:
 *   - weight_budget: bytes of model weights we're willing to grow
 *   - kv_budget: bytes of KV cache we're willing to keep
 *   - The planner ensures total(KB cache) + total(weight bytes) <= kv_budget + weight_budget
 */
typedef struct {
    float absorb_threshold;  /* coherence/param ratio to absorb  (0.001) */
    float keep_threshold;    /* coherence/param ratio to keep in KV (0.0001) */
    float prune_threshold;   /* below this → prune (0.00005) */
    size_t weight_budget;    /* max bytes to add to model weights (64MB) */
    size_t kv_budget;        /* max bytes for KV cache (128MB) */
    int    absorb_batch;      /* max files to absorb per cycle (8) */
} wubu_density_planner_cfg_t;

wubu_density_planner_cfg_t wubu_density_planner_default_cfg(void);

/* The absorption record: files promoted from KV cache to weights */
typedef struct {
    char   path[128];
    float  density;     /* coherence density */
    float  coherence;   /* raw coherence score */
    size_t n_tokens;    /* token count */
    int    absorbed;    /* 1 = promoted to weights */
} wubu_absorb_record_t;

/* Create the planner. All handles are caller-owned. */
wubu_density_planner_t *wubu_density_planner_create(
    wubu_kv_embedding_t *kv,
    wubu_scalable_model_t *model,
    wubu_kv_tiering_t *tiering,
    wubu_kv_shrink_t *shrink,
    const wubu_density_planner_cfg_t *cfg);

/* Run the density planning cycle:
 *
 * 1. Computes density = coherence.score / n_tokens for each file
 * 2. Ranks files by density (high to low)
 * 3. Files in the absorb band get marked hot (→ model weight absorption)
 * 4. Files in the keep band get tiered to F16/Q8_K in the KV cache
 * 5. Files in the prune band get scheduled for KV shrinking
 *
 * n_files / paths: files to evaluate
 * n_files / coherence: per-file wubu_coherence_t (from diagnostic)
 *
 * Returns the number of files absorbed (promoted to weight-hot).
 * Also populates out_absorbed (if non-NULL, caller does NOT free —
 * pointer into planner's internal buffer, valid until next cycle). */
int wubu_density_planner_cycle(wubu_density_planner_t *planner,
                                const char **paths,
                                const wubu_coherence_t *coherence,
                                int n_files,
                                wubu_absorb_record_t **out_absorbed,
                                int *out_n_absorbed);

/* Get the planner's memory usage summary. */
void wubu_density_planner_stats(wubu_density_planner_t *planner,
                                 size_t *out_absorbed_bytes,
                                 size_t *out_kv_bytes,
                                 size_t *out_pruned_bytes);

/* Free the planner. Does NOT free sub-operators. */
void wubu_density_planner_free(wubu_density_planner_t *planner);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_DENSITY_PLANNER_H */

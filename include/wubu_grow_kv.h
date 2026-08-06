/* wubu_grow_kv.h — the KV-space amoeba grow operator
 *
 * Extends the amoeba grow/shrink pattern (wubu_amoeba.h, wubu_gravity.h)
 * from transformer layers to KV cache blocks. When the coherence
 * diagnosis (wubu_kv_coherence_diag) identifies a file in /kv/in/ with
 * low coherence (the model doesn't understand it), the KV grow operator:
 *
 *   1. Computes the "gravity vector" toward the under-coherent region
 *      (Euclidean attractor — research/060, INDEX AN12).
 *   2. Grows a new KV block toward that region (the Euclidean attractor
 *      pulls blocks to where coherence is low).
 *   3. The new block is zero-initialized — it doesn't change existing
 *      attention, but on the next forward the query can attend to it.
 *   4. The coherence improvement on the next forward is the growth's
 *      reward — if it doesn't help, the 5+1 recovery rolls it back.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 5 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_GROW_KV_H
#define WUBU_GROW_KV_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration for the KV grow operator */
typedef struct {
    double coherence_threshold;   /* below this → grow toward the file (0.5) */
    double min_score_delta;       /* minimum coherence improvement to accept (0.05) */
    uint32_t max_kv_blocks;       /* safety ceiling on total KV blocks */
    uint32_t block_size;          /* floats per KV block */
} wubu_grow_kv_cfg_t;

/* Default config */
wubu_grow_kv_cfg_t wubu_grow_kv_default_cfg(void);

/* The KV grow operator state */
typedef struct wubu_grow_kv wubu_grow_kv_t;

/* Create the KV grow operator over a KV namespace + embedding bridge.
 * Returns NULL on failure. */
wubu_grow_kv_t *wubu_grow_kv_create(wubu_kv_embedding_t *kv,
                                     const wubu_grow_kv_cfg_t *cfg);

/* DIAGNOSE: scan every encoded file in /kv/in/, compute coherence for
 * each, and rank them by descending need (worst coherence first).
 * Returns the number of under-coherent files (below threshold).
 * The ranked list is stored internally for GROW to consume.
 *
 * NOTE: in the real trainer, the attention matrix comes from the model's
 * forward pass over /kv/in/. For the diagnose hook, the caller provides
 * pre-computed coherence scores (from the post-forward coherence reward).
 *
 * n_files / paths / scores: arrays of length n_files.
 * Returns the count of files below coherence_threshold. */
int wubu_grow_kv_diagnose(wubu_grow_kv_t *g,
                           const char **paths, const float *scores,
                           int n_files);

/* GROW: grow KV blocks toward the lowest-coherence files.
 * Grows up to max_grow blocks (or all under-coherent files, whichever is
 * less). Each growth mounts a new KV block at a neighboring path
 * (e.g., /kv/in/<path>/grow<N>) that the model can attend to.
 * Returns the number of blocks grown. */
int wubu_grow_kv_grow(wubu_grow_kv_t *g, int max_grow);

/* SHRINK: prune the KV block with lowest utilization / coherence.
 * Marks the block's mount as tombstoned and returns the path.
 * Returns 0 on success, -1 if nothing to shrink. */
int wubu_grow_kv_shrink(wubu_grow_kv_t *g);

/* Get the current list of under-coherent file paths (after diagnose).
 * Returns the count; out_paths is filled with up to cap paths.
 * Caller does NOT own the strings. */
int wubu_grow_kv_undercoherent(const wubu_grow_kv_t *g,
                                const char **out_paths, int cap);

/* Free the grow operator. Does NOT free kv or fs. */
void wubu_grow_kv_free(wubu_grow_kv_t *g);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GROW_KV_H */

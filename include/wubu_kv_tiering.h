/* wubu_kv_tiering.h — precision tiering for KV cache (Phase 9)
 *
 * The KV cache is stored in a precision tier determined by how
 * important each file's KV region is to the model. Hot files (high
 * attention mass, recently accessed) stay F32. Cold files get
 * compressed to F16, then Q8_K, then Q4_K to save memory.
 *
 * The tiering decision uses the wubu_weight_t type system:
 *   F32  (type=0) — full precision, baseline
 *   F16  (type=1) — half precision, 2× savings, ~1e-3 error
 *   Q8_K (type=15) — 8-bit K-quant, 4× savings, ~1e-2 error
 *   Q4_K (type=12) — 4-bit K-quant, 8× savings, ~1e-1 error
 *
 * The tier is tracked per KV path, allowing the model to lazily
 * dequantize on access (hot path stays in fast memory).
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 9 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_TIERING_H
#define WUBU_KV_TIERING_H

#include <stddef.h>
#include "wubu_kv_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Precision tier for a KV region */
typedef enum {
    KV_TIER_F32 = 0,   /* full precision (baseline) */
    KV_TIER_F16 = 1,   /* half precision (2× compression) */
    KV_TIER_Q8_K = 2,  /* 8-bit K-quant (4× compression) */
    KV_TIER_Q4_K = 3,  /* 4-bit K-quant (8× compression) */
} wubu_kv_tier_t;

typedef struct wubu_kv_tiering wubu_kv_tiering_t;

/* Tiering policy configuration */
typedef struct {
    float hot_threshold;    /* attention_mass >= this → keep current tier (0.1) */
    float warm_threshold;   /* attention_mass >= this → up-tier to F16 (0.01) */
    /* below warm_threshold → down-tier to Q4_K */
    int   decay_iters;      /* consecutive cold passes before deq (... 100) */
    int   max_compression_tier; /* deepest tier allowed (KV_TIER_Q4_K) */
    size_t total_kv_floats; /* total KV tensor size (for budget tracking) */
    size_t budget_floats;   /* max floats before tiering pressure begins */
} wubu_kv_tiering_cfg_t;

/* Default config: tiers kick in when KV usage exceeds 80% of budget. */
wubu_kv_tiering_cfg_t wubu_kv_tiering_default_cfg(void);

/* Create the tiering operator over a KV embedding layer. */
wubu_kv_tiering_t *wubu_kv_tiering_create(wubu_kv_embedding_t *kv,
                                           const wubu_kv_tiering_cfg_t *cfg,
                                           float *kv_base, size_t total_bytes);

/* Evaluate tiering given per-file attention utilization from the
 * latest forward pass. Re-tiers the coldest files downward and
 * the hottest upward, modifying the tier metadata in-place.
 *
 * n_files / paths / attention_mass: per-file utilization [0, 1].
 * Returns the number of files re-tiered. */
int wubu_kv_tiering_eval(wubu_kv_tiering_t *t,
                          const char **paths,
                          const float *attention_mass,
                          int n_files);

/* Query the current tier of a KV path. Returns KV_TIER_F32 by
 * default (new files start hot). Returns -1 if path not found. */
int wubu_kv_tiering_get(const wubu_kv_tiering_t *t, const char *path);

/* Report current memory usage at each tier. */
void wubu_kv_tiering_stats(const wubu_kv_tiering_t *t,
                            size_t *out_f32_bytes,
                            size_t *out_f16_bytes,
                            size_t *out_q8k_bytes,
                            size_t *out_q4k_bytes);

/* Free the tiering operator. Does NOT free kv or kv_base. */
void wubu_kv_tiering_free(wubu_kv_tiering_t *t);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_TIERING_H */

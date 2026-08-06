/* wubu_kv_tiering.c — precision tiering for KV cache
 *
 * Tracks per-KV-path precision tier (F32→F16→Q8_K→Q4_K) based on
 * attention utilization. Hot files stay F32; cold files get compressed.
 *
 * The tier is a metadata layer — the actual KV tensor data stays in
 * the tier indicated by the tracker; the model's load/store kernels
 * check the tier and dequantize on access as needed.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 9 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_tiering.h"
#include "gguf_reader.h"  /* for GGML_TYPE_* */
#include <stdlib.h>
#include <string.h>

#define MAX_TIERED_FILES 256

typedef struct {
    char    path[128];
    int     tier;           /* current wubu_kv_tier_t */
    int     cold_streak;    /* consecutive passes below warm_threshold */
    int     active;
    size_t  n_floats;       /* size of this file's KV region */
} tier_file_t;

struct wubu_kv_tiering {
    wubu_kv_embedding_t   *kv;
    wubu_kv_tiering_cfg_t  cfg;
    float                 *kv_base;
    size_t                 total_bytes;
    tier_file_t            files[MAX_TIERED_FILES];
    int                    n_files;
    /* byte budget tracking */
    size_t bytes_f32;
    size_t bytes_f16;
    size_t bytes_q8k;
    size_t bytes_q4k;
};

wubu_kv_tiering_cfg_t wubu_kv_tiering_default_cfg(void) {
    wubu_kv_tiering_cfg_t cfg;
    cfg.hot_threshold = 0.1f;
    cfg.warm_threshold = 0.01f;
    cfg.decay_iters = 100;
    cfg.max_compression_tier = KV_TIER_Q4_K;
    cfg.total_kv_floats = 0;
    cfg.budget_floats = 0;
    return cfg;
}

wubu_kv_tiering_t *wubu_kv_tiering_create(wubu_kv_embedding_t *kv,
                                           const wubu_kv_tiering_cfg_t *cfg,
                                           float *kv_base, size_t total_bytes) {
    if (!kv || !cfg) return NULL;
    wubu_kv_tiering_t *t = (wubu_kv_tiering_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->kv = kv;
    t->cfg = *cfg;
    t->kv_base = kv_base;
    t->total_bytes = total_bytes;
    t->n_files = 0;

    /* Initialize all encoded files as F32 */
    size_t n = wubu_kv_embedding_file_count(kv);
    for (size_t i = 0; i < n && (size_t)t->n_files < MAX_TIERED_FILES; i++) {
        const char *p = wubu_kv_embedding_get_path(kv, i);
        if (!p) continue;
        uint32_t blk; size_t off, nf;
        if (wubu_kv_embedding_region(kv, p, &blk, &off, &nf) == 0) {
            tier_file_t *tf = &t->files[t->n_files++];
            strncpy(tf->path, p, sizeof(tf->path) - 1);
            tf->path[sizeof(tf->path) - 1] = '\0';
            tf->tier = KV_TIER_F32;
            tf->cold_streak = 0;
            tf->active = 1;
            tf->n_floats = nf;
            t->bytes_f32 += nf * sizeof(float);
        }
    }
    return t;
}

void wubu_kv_tiering_free(wubu_kv_tiering_t *t) {
    if (!t) return;
    free(t);
}

/* Byte size of n_floats at a given tier */
static size_t tier_bytes(int tier, size_t n_floats) {
    switch (tier) {
        case KV_TIER_F32:  return n_floats * sizeof(float);
        case KV_TIER_F16:  return n_floats * 2;        /* half */
        case KV_TIER_Q8_K: return (n_floats * 8 + 7) / 8; /* 1 byte per ~8... actually 4+1+3=8 bits per 16 */
        case KV_TIER_Q4_K: return (n_floats * 4 + 7) / 8; /* 4 bits per elem */
        default: return n_floats * sizeof(float);
    }
}

static tier_file_t *find_file(wubu_kv_tiering_t *t, const char *path) {
    for (int i = 0; i < t->n_files; i++) {
        if (t->files[i].active &&
            strcmp(t->files[i].path, path) == 0)
            return &t->files[i];
    }
    return NULL;
}

/* Tier transition table: F32 → F16 → Q8_K → Q4_K → F32 (coldest loops back) */
static int tier_down(int tier) {
    switch (tier) {
        case KV_TIER_F32:  return KV_TIER_F16;
        case KV_TIER_F16:  return KV_TIER_Q8_K;
        case KV_TIER_Q8_K: return KV_TIER_Q4_K;
        default:           return KV_TIER_Q4_K;
    }
}

static int tier_up(int tier) {
    switch (tier) {
        case KV_TIER_Q4_K: return KV_TIER_Q8_K;
        case KV_TIER_Q8_K: return KV_TIER_F16;
        case KV_TIER_F16:
        case KV_TIER_F32:  return KV_TIER_F32;
        default:           return KV_TIER_F32;
    }
}

/* Bytes saved by moving from tier `from` to `to` for n_floats */
static long tier_savings(int from, int to, size_t n_floats) {
    return (long)tier_bytes(from, n_floats) - (long)tier_bytes(to, n_floats);
}

/* Recompute byte stats from scratch */
static void recompute_stats(wubu_kv_tiering_t *t) {
    t->bytes_f32 = t->bytes_f16 = t->bytes_q8k = t->bytes_q4k = 0;
    for (int i = 0; i < t->n_files; i++) {
        if (!t->files[i].active) continue;
        switch (t->files[i].tier) {
            case KV_TIER_F32:  t->bytes_f32 += tier_bytes(KV_TIER_F32, t->files[i].n_floats); break;
            case KV_TIER_F16:  t->bytes_f16 += tier_bytes(KV_TIER_F16, t->files[i].n_floats); break;
            case KV_TIER_Q8_K: t->bytes_q8k += tier_bytes(KV_TIER_Q8_K, t->files[i].n_floats); break;
            case KV_TIER_Q4_K: t->bytes_q4k += tier_bytes(KV_TIER_Q4_K, t->files[i].n_floats); break;
        }
    }
}

int wubu_kv_tiering_eval(wubu_kv_tiering_t *t,
                          const char **paths,
                          const float *attention_mass,
                          int n_files) {
    if (!t || !paths || !attention_mass || n_files <= 0) return 0;
    int n_retiered = 0;

    /* First pass: update cold streaks and compute total budget usage */
    size_t total_used = 0;
    for (int i = 0; i < n_files; i++) {
        tier_file_t *tf = find_file(t, paths[i]);
        if (!tf) continue;
        if (attention_mass[i] >= t->cfg.hot_threshold) {
            tf->cold_streak = 0;
        } else if (attention_mass[i] >= t->cfg.warm_threshold) {
            tf->cold_streak = (tf->cold_streak > 0) ? tf->cold_streak - 1 : 0;
        } else {
            tf->cold_streak++;
        }
        total_used += tier_bytes(tf->tier, tf->n_floats);
    }

    /* Check budget pressure */
    size_t budget = t->cfg.budget_floats * sizeof(float);
    if (budget == 0) {
        /* No budget set — use total_bytes as the budget.
         * Tier aggressively if usage > 80% of total. */
        budget = (t->total_bytes * 8) / 10;
    }
    int under_pressure = (total_used > budget);

    /* Second pass: re-tier based on utilization */
    for (int i = 0; i < n_files; i++) {
        tier_file_t *tf = find_file(t, paths[i]);
        if (!tf) continue;

        int old_tier = tf->tier;

        if (attention_mass[i] >= t->cfg.hot_threshold) {
            /* Hot → up-tier to F32 (or stay) */
            if (tf->tier != KV_TIER_F32) {
                tf->tier = tier_up(tf->tier);
                tf->cold_streak = 0;
            }
        } else if (attention_mass[i] >= t->cfg.warm_threshold) {
            /* Warm → at most F16 */
            if (tf->tier != KV_TIER_F16 && tf->tier != KV_TIER_F32) {
                tf->tier = tier_up(tf->tier);
                tf->cold_streak = 0;
            }
        } else {
            /* Cold → down-tier if streak exceeds decay_iters.
             * Always allow F32→F16 for cold files (save memory).
             * Deeper tiers require memory pressure. */
            if (tf->cold_streak >= t->cfg.decay_iters) {
                if (tf->tier == KV_TIER_F32) {
                    /* F32 → F16 always allowed for cold files */
                    tf->tier = tier_down(tf->tier);
                    n_retiered++;
                } else if (tf->tier < t->cfg.max_compression_tier &&
                           under_pressure) {
                    tf->tier = tier_down(tf->tier);
                    n_retiered++;
                }
            }
        }

        if (tf->tier != old_tier) n_retiered++;
    }

    recompute_stats(t);
    return n_retiered;
}

int wubu_kv_tiering_get(const wubu_kv_tiering_t *t, const char *path) {
    if (!t || !path) return -1;
    for (int i = 0; i < t->n_files; i++) {
        if (t->files[i].active &&
            strcmp(t->files[i].path, path) == 0)
            return t->files[i].tier;
    }
    return -1;
}

void wubu_kv_tiering_stats(const wubu_kv_tiering_t *t,
                            size_t *out_f32_bytes,
                            size_t *out_f16_bytes,
                            size_t *out_q8k_bytes,
                            size_t *out_q4k_bytes) {
    if (!t) {
        if (out_f32_bytes) *out_f32_bytes = 0;
        if (out_f16_bytes) *out_f16_bytes = 0;
        if (out_q8k_bytes) *out_q8k_bytes = 0;
        if (out_q4k_bytes) *out_q4k_bytes = 0;
        return;
    }
    if (out_f32_bytes) *out_f32_bytes = t->bytes_f32;
    if (out_f16_bytes) *out_f16_bytes = t->bytes_f16;
    if (out_q8k_bytes) *out_q8k_bytes = t->bytes_q8k;
    if (out_q4k_bytes) *out_q4k_bytes = t->bytes_q4k;
}

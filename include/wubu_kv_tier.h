/*
 * wubu_kv_tier.h -- KV cache multi-tier cold storage (doc 002).
 *
 * Three tiers on a single host:
 *   HOT  = existing gqa_k_cache / gqa_v_cache (CPU RAM)
 *   WARM = DRAM-mmap file (evictable under memory pressure)
 *   COLD = NVMe file via POSIX mmap (swap-backed; page faults are latency)
 *
 * Per-block: tier tag + last-access EMA + block_id + offset into file.
 * Eviction policy: EMA-scored LRU.
 *
 * Self-contained C11; no third-party deps.
 *
 * Usage in the engine (future integration):
 *   - Replace per-position gqa_k_cache[offset] pointer math with
 *     wubu_kv_block_t* lookups through the tier.
 *   - The tier handles promotion/demotion across tiers transparently.
 *   - WUBU_KV_TIER_LIMIT_MB env var governs warm/cold budget.
 */

#ifndef WUBU_KV_TIER_H
#define WUBU_KV_TIER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_KV_TIER_HOT  = 0,  /* CPU RAM — current tokens */
    WUBU_KV_TIER_WARM = 1,  /* DRAM-mmap file — evictable */
    WUBU_KV_TIER_COLD = 2,  /* NVMe file — swap-backed */
} wubu_kv_tier_e;

/* A single KV block (one token position, all heads).
 * data is allocated per-block; tier owns the memory lifecycle. */
typedef struct wubu_kv_block {
    wubu_kv_tier_e tier;
    uint8_t *data;       /* [block_bytes] K+V concatenated */
    size_t block_bytes;
    bool is_dirty;        /* needs write-back on demote */
    float last_access_ema; /* EMA of access recency (1.0 = just used, decays) */
    uint64_t offset_in_file; /* byte offset into warm/cold file */
    int file_fd;          /* fd for warm/cold mmap'd tier */
    void *mmap_addr;      /* mmap'd region (future: cold tier) */
    size_t mmap_size;
    int ref_count;
} wubu_kv_block_t;

typedef struct wubu_kv_tier {
    size_t hot_capacity;
    size_t hot_used;
    wubu_kv_block_t *hot_blocks;
    size_t warm_limit_bytes;
    size_t warm_used_bytes;
    int    cold_used_bytes;  /* bytes demoted to cold tier */
    size_t cold_limit_bytes;
    char *warm_file_path;
    char *cold_file_path;
    int warm_fd;
    int cold_fd;
    int warm_free_top;
    int cold_free_top;
} wubu_kv_tier_t;

/* Create a three-tier KV arena.
 * hot_capacity_blocks: number of blocks in RAM (HOT tier)
 * warm_limit_mb       : DRAM-mmap budget
 * cold_limit_mb       : NVMe/disk budget
 * warm_path           : path to warm file (created if absent)
 * cold_path           : path to cold file (created if absent) */
wubu_kv_tier_t *wubu_kv_tier_create(size_t hot_capacity_blocks,
                                           size_t warm_limit_mb,
                                           size_t cold_limit_mb,
                                           const char *warm_path,
                                           const char *cold_path);

/* Free all tiers, close files, unmap memory. */
void wubu_kv_tier_free(wubu_kv_tier_t *t);

/* Allocate a new block (hot tier). Returns NULL on OOM or tier full. */
wubu_kv_block_t *wubu_kv_tier_alloc_block(wubu_kv_tier_t *t, size_t block_bytes);

/* Read len bytes from block b at offset into dst. Returns 0 on success. */
int wubu_kv_tier_read_block(wubu_kv_tier_t *t, wubu_kv_block_t *b,
                               size_t offset, uint8_t *dst, size_t len);

/* Write len bytes into block b at offset from src. Returns 0 on success. */
int wubu_kv_tier_write_block(wubu_kv_tier_t *t, wubu_kv_block_t *b,
                               size_t offset, const uint8_t *src, size_t len);

/* Evict cold (low-EMA) blocks from warm tier down to cold tier. */
void wubu_kv_tier_evict_cold(wubu_kv_tier_t *t, size_t target_evict_bytes);

/* Query tier statistics. */
void wubu_kv_tier_stats(const wubu_kv_tier_t *t,
                             size_t *hot_blocks, size_t *warm_bytes,
                             size_t *cold_bytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_TIER_H */

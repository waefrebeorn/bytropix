/*
 * wubu_lmcache.h -- LMCache-style prefix+PD KV persistence (doc A06).
 *
 * Persists KV blocks across prefill/decode phases and across requests
 * using a file-backed cache keyed by (model_hash, prefix_hash, layer).
 *
 * Basis: LMCache, arXiv:2510.09665 (15× throughput, 2× lower latency).
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_LMCACHE_H
#define WUBU_LMCACHE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_LMCACHE_MAX_ENTRIES 256
#define WUBU_LMCACHE_KEY_LEN 4096

typedef struct {
    uint64_t key_hash;    /* FNV-1a hash of (model_id + token_ids) */
    size_t file_offset;   /* offset in cache file */
    int n_blocks;         /* number of KV blocks stored */
    uint64_t last_access; /* for LRU eviction */
} wubu_lmcache_entry_t;

typedef struct {
    char cache_dir[256];
    int n_layers;
    int block_size;
    int head_dim;
    int n_kv_heads;
    size_t block_bytes;
    wubu_lmcache_entry_t entries[WUBU_LMCACHE_MAX_ENTRIES];
    int n_entries;
    uint64_t access_counter;
    size_t n_hits;
    size_t n_misses;
    size_t n_evictions;
} wubu_lmcache_t;

/* Create/destroy. */
wubu_lmcache_t *wubu_lmcache_create(const char *cache_dir, int n_layers,
                                      int block_size, int head_dim, int n_kv_heads);
void wubu_lmcache_free(wubu_lmcache_t *c);

/* Store KV blocks for a prefix (called after prefill completes). */
int wubu_lmcache_store(wubu_lmcache_t *c, const char *model_id,
                        const int *token_ids, int n_tokens,
                        const float *kv_data, int n_blocks);

/* Load KV blocks for a prefix (called before prefill to skip compute).
 * Returns number of blocks loaded, 0 on miss, -1 on error. */
int wubu_lmcache_load(wubu_lmcache_t *c, const char *model_id,
                       const int *token_ids, int n_tokens,
                       float *kv_data, int max_blocks);

/* Stats. */
void wubu_lmcache_stats(const wubu_lmcache_t *c,
                         int *n_entries, size_t *hits, size_t *misses, size_t *evictions);
float wubu_lmcache_hit_rate(const wubu_lmcache_t *c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LMCACHE_H */

/*
 * wubu_lmcache.c -- LMCache-style prefix+PD KV persistence (doc A06).
 *
 * Source: LMCache (arXiv:2510.09665) — up to 15× throughput, 2× lower
 * latency via prefix offload + prefill/decode disaggregation.
 *
 * Core idea: Extend prefix KV reuse (D02) to persist KV blocks across
 * prefill/decode phases AND across requests. When a prefill completes,
 * its KV blocks are written to a file-backed cache keyed by
 * (model_hash, prefix_hash, layer). When a new request with the same
 * prefix arrives, the KV is loaded directly from the cache file,
 * skipping the prefill compute entirely.
 *
 * PD-disaggregation: The prefill phase writes KV to the cache; the
 * decode phase reads from it. This separation allows prefill to be
 * batched and pipelined independently from decode.
 *
 * Win: 15× throughput for shared-system-prompt workloads (RAG, chatbots),
 * 2× lower latency by skipping redundant prefill compute.
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_lmcache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* FNV-1a hash for cache keys */
static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

/* Create an LMCache context. */
wubu_lmcache_t *wubu_lmcache_create(const char *cache_dir, int n_layers,
                                      int block_size, int head_dim, int n_kv_heads) {
    if (!cache_dir || n_layers <= 0 || block_size <= 0 || head_dim <= 0 || n_kv_heads <= 0)
        return NULL;

    wubu_lmcache_t *c = (wubu_lmcache_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;

    strncpy(c->cache_dir, cache_dir, sizeof(c->cache_dir) - 1);
    c->n_layers = n_layers;
    c->block_size = block_size;
    c->head_dim = head_dim;
    c->n_kv_heads = n_kv_heads;
    c->block_bytes = (size_t)n_kv_heads * block_size * head_dim * sizeof(float);
    c->n_entries = 0;
    c->n_hits = 0;
    c->n_misses = 0;
    c->n_evictions = 0;

    /* Initialize entry table */
    for (int i = 0; i < WUBU_LMCACHE_MAX_ENTRIES; i++) {
        c->entries[i].key_hash = 0;
        c->entries[i].file_offset = 0;
        c->entries[i].n_blocks = 0;
        c->entries[i].last_access = 0;
    }

    return c;
}

void wubu_lmcache_free(wubu_lmcache_t *c) {
    free(c);
}

/* Store KV blocks for a prefix in the cache.
 * kv_data: [n_layers, n_blocks, n_kv_heads, block_size, head_dim] row-major
 * Returns number of blocks stored, or -1 on error. */
int wubu_lmcache_store(wubu_lmcache_t *c, const char *model_id,
                        const int *token_ids, int n_tokens,
                        const float *kv_data, int n_blocks) {
    if (!c || !model_id || !token_ids || n_tokens <= 0 || !kv_data || n_blocks <= 0)
        return -1;

    /* Compute cache key: hash(model_id + token_ids) */
    char key_str[WUBU_LMCACHE_KEY_LEN];
    snprintf(key_str, sizeof(key_str), "%s:", model_id);
    int key_pos = (int)strlen(key_str);
    for (int i = 0; i < n_tokens && key_pos < (int)sizeof(key_str) - 12; i++) {
        key_pos += snprintf(key_str + key_pos, sizeof(key_str) - key_pos, "%d:", token_ids[i]);
    }
    uint64_t key_hash = fnv1a(key_str);

    /* Find a free entry or evict LRU */
    int slot = -1;
    for (int i = 0; i < c->n_entries; i++) {
        if (c->entries[i].key_hash == key_hash) {
            slot = i;  /* overwrite existing */
            break;
        }
    }
    if (slot < 0) {
        if (c->n_entries < WUBU_LMCACHE_MAX_ENTRIES) {
            slot = c->n_entries++;
        } else {
            /* Evict LRU */
            uint64_t oldest = UINT64_MAX;
            for (int i = 0; i < c->n_entries; i++) {
                if (c->entries[i].last_access < oldest) {
                    oldest = c->entries[i].last_access;
                    slot = i;
                }
            }
            c->n_evictions++;
        }
    }

    /* Write KV data to a file */
    char filepath[1024];
    snprintf(filepath, sizeof(filepath), "%s/lmcache_%016lx.bin",
             c->cache_dir, (unsigned long)key_hash);
    FILE *fp = fopen(filepath, "wb");
    if (!fp) return -1;

    size_t total_bytes = (size_t)c->n_layers * n_blocks * c->block_bytes;
    size_t written = fwrite(kv_data, 1, total_bytes, fp);
    fclose(fp);
    if (written != total_bytes) return -1;

    /* Record entry */
    c->entries[slot].key_hash = key_hash;
    c->entries[slot].n_blocks = n_blocks;
    c->entries[slot].file_offset = 0;
    c->entries[slot].last_access = ++c->access_counter;

    return n_blocks;
}

/* Load KV blocks for a prefix from the cache.
 * If found, reads kv_data and returns number of blocks.
 * If not found, returns 0 (miss). */
int wubu_lmcache_load(wubu_lmcache_t *c, const char *model_id,
                       const int *token_ids, int n_tokens,
                       float *kv_data, int max_blocks) {
    if (!c || !model_id || !token_ids || n_tokens <= 0 || !kv_data || max_blocks <= 0)
        return -1;

    /* Compute cache key */
    char key_str[WUBU_LMCACHE_KEY_LEN];
    snprintf(key_str, sizeof(key_str), "%s:", model_id);
    int key_pos = (int)strlen(key_str);
    for (int i = 0; i < n_tokens && key_pos < (int)sizeof(key_str) - 12; i++) {
        key_pos += snprintf(key_str + key_pos, sizeof(key_str) - key_pos, "%d:", token_ids[i]);
    }
    uint64_t key_hash = fnv1a(key_str);

    /* Find in entry table */
    int slot = -1;
    for (int i = 0; i < c->n_entries; i++) {
        if (c->entries[i].key_hash == key_hash) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        c->n_misses++;
        return 0;  /* cache miss */
    }

    /* Read from file */
    char filepath[1024];
    snprintf(filepath, sizeof(filepath), "%s/lmcache_%016lx.bin",
             c->cache_dir, (unsigned long)key_hash);
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        c->n_misses++;
        return 0;
    }

    int n_blocks = c->entries[slot].n_blocks;
    if (n_blocks > max_blocks) n_blocks = max_blocks;
    size_t total_bytes = (size_t)c->n_layers * n_blocks * c->block_bytes;
    size_t nread = fread(kv_data, 1, total_bytes, fp);
    fclose(fp);
    if (nread != total_bytes) return -1;

    c->entries[slot].last_access = ++c->access_counter;
    c->n_hits++;
    return n_blocks;
}

/* Get cache statistics. */
void wubu_lmcache_stats(const wubu_lmcache_t *c,
                         int *n_entries, size_t *hits, size_t *misses, size_t *evictions) {
    if (!c) return;
    if (n_entries) *n_entries = c->n_entries;
    if (hits) *hits = c->n_hits;
    if (misses) *misses = c->n_misses;
    if (evictions) *evictions = c->n_evictions;
}

/* Compute hit rate. */
float wubu_lmcache_hit_rate(const wubu_lmcache_t *c) {
    if (!c) return 0.0f;
    size_t total = c->n_hits + c->n_misses;
    return (total > 0) ? (float)c->n_hits / (float)total : 0.0f;
}

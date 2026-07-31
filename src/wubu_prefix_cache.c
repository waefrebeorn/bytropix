/*
 * wubu_prefix_cache.c -- Automatic Prefix Caching implementation.
 * Pure C11, self-contained. Uses FNV-1a 64-bit hash (no OpenSSL dep).
 */

#include "wubu_prefix_cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* FNV-1a 64-bit: deterministic, good distribution, zero third-party dep. */
static uint64_t fnv1a64(const int *data, int n) {
    if (!data || n <= 0) return 0;
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        h ^= (uint64_t)(uint32_t)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Create: allocate and initialize the prefix cache. */
wubu_prefix_cache_t *wubu_prefix_cache_create(void) {
    wubu_prefix_cache_t *cache = (wubu_prefix_cache_t *)calloc(1, sizeof(*cache));
    if (!cache) return NULL;

    /* Initialize free list: all nodes except root (0) are free */
    cache->free_top = WUBU_PREFIX_MAX_NODES - 1;
    for (int i = 1; i < WUBU_PREFIX_MAX_NODES; i++) {
        cache->free_list[i - 1] = WUBU_PREFIX_MAX_NODES - i;
    }
    /* Root node (0): all children = -1 */
    for (int t = 0; t < 256; t++) {
        cache->nodes[0].children[t] = -1;
    }
    cache->nodes[0].parent = -1;
    cache->n_nodes = 1;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    cache->evictions = 0;

    return cache;
}

void wubu_prefix_cache_free(wubu_prefix_cache_t *cache) {
    free(cache);
}

/* Register a prefix: walk/create nodes, mark as cached terminal. */
wubu_prefix_hash_t wubu_prefix_cache_register(wubu_prefix_cache_t *cache,
                                               const int *token_ids, int n_tokens,
                                               void *paged_kv,
                                               int block_size) {
    if (!cache || n_tokens <= 0) return 0;

    int node = 0;  /* root */
    int depth = 0;
    int tokens_per_block = block_size > 0 ? block_size : 1;

    for (int i = 0; i < n_tokens; i++) {
        int tok = token_ids[i] & 0xFF;
        if (cache->nodes[node].children[tok] == -1) {
            if (cache->free_top <= 0) break; /* cache full */
            int new_node = cache->free_list[--cache->free_top];
            memset(&cache->nodes[new_node], 0, sizeof(wubu_prefix_node_t));
            cache->nodes[new_node].parent = node;
            for (int t = 0; t < 256; t++)
                cache->nodes[new_node].children[t] = -1;
            cache->nodes[node].children[tok] = new_node;
            cache->n_nodes++;
        }
        node = cache->nodes[node].children[tok];
        depth++;

        /* Allocate KV block at block boundaries */
        if ((i + 1) % tokens_per_block == 0 && paged_kv) {
            (void)paged_kv; /* real integration would call wubu_paged_kv_ensure() */
        }
    }

    cache->nodes[node].is_terminal = true;
    cache->nodes[node].ref_count++;
    cache->nodes[node].last_access = ++cache->access_counter;

    return fnv1a64(token_ids, depth);
}

/* Match: walk the trie, return matched tokens. */
int wubu_prefix_cache_match(wubu_prefix_cache_t *cache,
                            const int *token_ids, int n_tokens,
                            int *out_kv_blocks, int max_blocks) {
    if (!cache || n_tokens <= 0) return 0;

    int node = 0;
    int matched = 0;
    int blocks_copied = 0;

    for (int i = 0; i < n_tokens; i++) {
        int tok = token_ids[i] & 0xFF;
        int child = cache->nodes[node].children[tok];
        if (child == -1) break;

        node = child;
        matched++;

        if (cache->nodes[node].n_kv_blocks > 0 && blocks_copied < max_blocks) {
            for (int b = 0; b < cache->nodes[node].n_kv_blocks && blocks_copied < max_blocks; b++) {
                out_kv_blocks[blocks_copied++] = cache->nodes[node].kv_blocks[b];
            }
        }
    }

    if (matched > 0) {
        cache->hits++;
        cache->nodes[node].last_access = ++cache->access_counter;
    } else {
        cache->misses++;
    }

    return matched;
}

/* Release sequence's reference to prefix nodes */
void wubu_prefix_cache_release(wubu_prefix_cache_t *cache,
                               const int *token_ids, int n_tokens) {
    if (!cache || n_tokens <= 0) return;

    int node = 0;
    for (int i = 0; i < n_tokens; i++) {
        int tok = token_ids[i] & 0xFF;
        int child = cache->nodes[node].children[tok];
        if (child == -1) break;
        node = child;
        if (cache->nodes[node].ref_count > 0) {
            cache->nodes[node].ref_count--;
        }
    }
}

/* LRU eviction: find least recently used node with ref_count == 0 */
void wubu_prefix_cache_evict_lru(wubu_prefix_cache_t *cache, int count) {
    if (!cache || count <= 0) return;

    for (int evicted = 0; evicted < count; evicted++) {
        uint64_t oldest = UINT64_MAX;
        int victim = -1;

        for (int i = 1; i < cache->n_nodes; i++) {
            if (cache->nodes[i].ref_count == 0 &&
                cache->nodes[i].last_access > 0 &&
                cache->nodes[i].last_access < oldest) {
                oldest = cache->nodes[i].last_access;
                victim = i;
            }
        }

        if (victim == -1) break;

        /* Remove from parent's children */
        int parent = cache->nodes[victim].parent;
        if (parent >= 0) {
            for (int t = 0; t < 256; t++) {
                if (cache->nodes[parent].children[t] == victim) {
                    cache->nodes[parent].children[t] = -1;
                    break;
                }
            }
        }

        /* Return to free list */
        if (cache->free_top < WUBU_PREFIX_MAX_NODES) {
            cache->free_list[cache->free_top++] = victim;
        }
        cache->evictions++;
    }
}

void wubu_prefix_cache_stats(const wubu_prefix_cache_t *cache,
                             size_t *hits, size_t *misses, size_t *evictions,
                             size_t *nodes_used) {
    if (hits) *hits = cache ? cache->hits : 0;
    if (misses) *misses = cache ? cache->misses : 0;
    if (evictions) *evictions = cache ? cache->evictions : 0;
    if (nodes_used) *nodes_used = cache ? (size_t)cache->n_nodes : 0;
}

wubu_prefix_hash_t wubu_prefix_hash_compute(const int *token_ids, int n_tokens) {
    return fnv1a64(token_ids, n_tokens);
}

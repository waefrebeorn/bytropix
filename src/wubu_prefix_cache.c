/*
 * wubu_prefix_cache.c -- Automatic Prefix Caching implementation.
 * Pure C11, self-contained. See header for convergent research basis.
 */

#define _GNU_SOURCE
#include "wubu_prefix_cache.h"
#include "wubu_paged_kv.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <openssl/sha.h>

/* ---- Prefix Hash: 64-bit SHA256 truncated ---- */
wubu_prefix_hash_t wubu_prefix_hash_compute(const int *token_ids, int n_tokens) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    for (int i = 0; i < n_tokens; i++) {
        uint32_t tok = (uint32_t)token_ids[i];
        unsigned char bytes[4] = { (uint8_t)(tok >> 24), (uint8_t)(tok >> 16),
                                   (uint8_t)(tok >> 8), (uint8_t)tok };
        SHA256_Update(&ctx, bytes, 4);
    }
    SHA256_Final(hash, &ctx);
    wubu_prefix_hash_t h = 0;
    for (int i = 0; i < 8; i++) {
        h = (h << 8) | hash[i];
    }
    return h;
}

/* ---- Prefix Cache Implementation ---- */

static int alloc_node(wubu_prefix_cache_t *cache) {
    if (cache->free_top == 0) return -1;
    int idx = cache->free_list[--cache->free_top];
    memset(&cache->nodes[idx], 0, sizeof(wubu_prefix_node_t));
    cache->nodes[idx].parent = -1;
    for (int i = 0; i < 256; i++) cache->nodes[idx].children[i] = -1;
    return idx;
}

wubu_prefix_cache_t *wubu_prefix_cache_create(void) {
    wubu_prefix_cache_t *cache = (wubu_prefix_cache_t *)calloc(1, sizeof(wubu_prefix_cache_t));
    if (!cache) return NULL;
    cache->n_nodes = 1;  /* root at index 0 */
    cache->free_top = WUBU_PREFIX_MAX_NODES - 1;
    for (int i = 0; i < WUBU_PREFIX_MAX_NODES - 1; i++) {
        cache->free_list[i] = i + 1;  /* 1..MAX-1 */
    }
    cache->access_counter = 0;
    return cache;
}

void wubu_prefix_cache_free(wubu_prefix_cache_t *cache) {
    if (cache) free(cache);
}

/* Register a prefix: walk/create nodes, allocate KV blocks via paged KV */
wubu_prefix_hash_t wubu_prefix_cache_register(wubu_prefix_cache_t *cache,
                                               const int *token_ids, int n_tokens,
                                               wubu_paged_kv_t *paged_kv,
                                               int block_size) {
    if (!cache || n_tokens <= 0) return 0;

    int node = 0;  /* root */
    int depth = 0;
    int tokens_per_block = block_size;  /* assuming 1 token per block for simplicity */

    for (int i = 0; i < n_tokens; i++) {
        int tok = token_ids[i] & 0xFF;  /* use low byte as child index */
        if (cache->nodes[node].children[tok] == -1) {
            int new_node = alloc_node(cache);
            if (new_node == -1) break;
            cache->nodes[node].children[tok] = new_node;
            cache->nodes[new_node].parent = node;
            cache->n_nodes++;
        }
        node = cache->nodes[node].children[tok];
        depth++;

        /* Allocate KV block at block boundaries */
        if ((i + 1) % tokens_per_block == 0 && paged_kv) {
            int seq = 0;  /* we'd need a proper seq id in production */
            int phys_block = wubu_paged_kv_ensure(paged_kv, seq, (i + 1) / tokens_per_block);
            if (phys_block >= 0 && cache->nodes[node].n_kv_blocks < WUBU_PREFIX_MAX_LEN / 16) {
                cache->nodes[node].kv_blocks[cache->nodes[node].n_kv_blocks++] = phys_block;
            }
        }
    }

    cache->nodes[node].is_terminal = true;
    cache->nodes[node].ref_count++;
    cache->nodes[node].last_access = ++cache->access_counter;

    return wubu_prefix_hash_compute(token_ids, depth);
}

/* Match: walk the trie, return matched length and populate block IDs */
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

        for (int i = 1; i < cache->n_nodes; i++) {  /* skip root */
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
        cache->free_list[cache->free_top++] = victim;
        cache->evictions++;
    }
}

void wubu_prefix_cache_stats(const wubu_prefix_cache_t *cache,
                             size_t *hits, size_t *misses, size_t *evictions,
                             size_t *nodes_used) {
    if (hits) *hits = cache ? cache->hits : 0;
    if (misses) *misses = cache ? cache->misses : 0;
    if (evictions) *evictions = cache ? cache->evictions : 0;
    if (nodes_used) *nodes_used = cache ? cache->n_nodes : 0;
}
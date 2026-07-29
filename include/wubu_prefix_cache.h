/*
 * wubu_prefix_cache.h -- Automatic Prefix Caching for KV Cache.
 *
 * CONVERGENT WIN (Kevin-Bacon meta-analysis across vLLM, SGLang, llm-d, Furiosa):
 *   - Prefix caching is the SINGLE HIGHEST-ROI optimization for LLM serving.
 *   - vLLM: "57x faster response, double throughput on identical hardware"
 *   - llm-d: "Precise prefix-cache aware scheduling delivers order-of-magnitude gains"
 *   - SGLang: "Prefix caching + continuous batching = production standard"
 *   - Furiosa: "10x cost difference between cached and uncached tokens"
 *
 * The insight: most multi-turn conversations share long prefixes (system prompts,
 * few-shot examples, RAG context). Recomputing KV for shared prefixes is pure
 * waste. Hash the prefix -> store KV blocks -> exact match = zero recompute.
 *
 * Design (vLLM/SGLang compatible):
 *   - Hash = SHA256(token_ids[0..prefix_len]) truncated to 64-bit
 *   - Reference counting per prefix node (for eviction)
 *   - LRU eviction when cache full
 *   - Thread-safe for concurrent requests
 *   - Zero-copy: stores physical block IDs, not data copies
 */

#ifndef WUBU_PREFIX_CACHE_H
#define WUBU_PREFIX_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_PREFIX_MAX_LEN 4096      /* max tokens in a prefix */
#define WUBU_PREFIX_MAX_NODES 16384   /* max prefix tree nodes */
#define WUBU_PREFIX_HASH_BITS 64      /* 64-bit hash = practical collision-free */

typedef uint64_t wubu_prefix_hash_t;

/* Forward declaration for paged KV (full definition in wubu_paged_kv.h) */
typedef struct wubu_paged_kv wubu_paged_kv_t;

typedef struct wubu_prefix_node {
    wubu_prefix_hash_t hash;
    int parent;              /* index in nodes array, -1 for root */
    int children[256];       /* token -> child node index (-1 if none) */
    int ref_count;           /* number of sequences using this prefix */
    int kv_blocks[WUBU_PREFIX_MAX_LEN / 16];  /* physical block IDs */
    int n_kv_blocks;
    uint64_t last_access;    /* for LRU */
    bool is_terminal;        /* marks end of a registered prefix */
} wubu_prefix_node_t;

typedef struct wubu_prefix_cache {
    wubu_prefix_node_t nodes[WUBU_PREFIX_MAX_NODES];
    int n_nodes;
    int free_list[WUBU_PREFIX_MAX_NODES];
    int free_top;
    uint64_t access_counter;
    /* Stats */
    size_t hits;
    size_t misses;
    size_t evictions;
} wubu_prefix_cache_t;

/* Create/destroy */
wubu_prefix_cache_t *wubu_prefix_cache_create(void);
void wubu_prefix_cache_free(wubu_prefix_cache_t *cache);

/* Core operations */
/* Register a prefix (token sequence) -> returns hash, allocates KV blocks via paged KV */
wubu_prefix_hash_t wubu_prefix_cache_register(wubu_prefix_cache_t *cache,
                                               const int *token_ids, int n_tokens,
                                               wubu_paged_kv_t *paged_kv,
                                               int block_size);

/* Lookup: returns number of matching prefix tokens (0 = no match).
 * If match > 0, populates out_kv_blocks with physical block IDs. */
int wubu_prefix_cache_match(wubu_prefix_cache_t *cache,
                            const int *token_ids, int n_tokens,
                            int *out_kv_blocks, int max_blocks);

/* Release a sequence's reference to prefix nodes (called on sequence completion) */
void wubu_prefix_cache_release(wubu_prefix_cache_t *cache,
                               const int *token_ids, int n_tokens);

/* Eviction: LRU evict least-recently-used prefix subtree */
void wubu_prefix_cache_evict_lru(wubu_prefix_cache_t *cache, int count);

/* Stats */
void wubu_prefix_cache_stats(const wubu_prefix_cache_t *cache,
                             size_t *hits, size_t *misses, size_t *evictions,
                             size_t *nodes_used);

/* Hash utility: compute 64-bit hash of token sequence */
wubu_prefix_hash_t wubu_prefix_hash_compute(const int *token_ids, int n_tokens);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PREFIX_CACHE_H */
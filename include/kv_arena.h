/**
 * kv_arena.h — Game Engine Style KV Cache Arena Allocator
 * 
 * Single contiguous arena per GQA layer with block-table indirection.
 * Eliminates VRAM fragmentation by pre-allocating full max_ctx at init.
 */

#ifndef KV_ARENA_H
#define KV_ARENA_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KV_BLOCK_TOKENS 16
#define KV_HEADS 2
#define KV_HEAD_DIM 256
#define KV_DIM (KV_HEADS * KV_HEAD_DIM)  // 512
#define Q4_0_BLOCK_BYTES 18
#define Q4_0_ELEMS_PER_BLOCK 32

// Full arena structure (not opaque - members accessed by host code)
typedef struct {
    // Arena: single contiguous allocation for K and V
    uint8_t* d_k_arena;
    uint8_t* d_v_arena;
    
    // Block table: logical token index → physical block index
    // Size: max_ctx / KV_BLOCK_TOKENS entries
    int* d_block_table;
    
    // Free list: stack of available block indices
    int* d_free_list;
    int free_list_top;  // atomic index for allocation (host-tracked for simplicity)
    
    // LRU list: doubly-linked list of active blocks (by block index) — HOST SIDE
    // For eviction: we track LRU on host to avoid device sync
    int* h_lru_prev;
    int* h_lru_next;
    int lru_head;  // most recently used
    int lru_tail;  // least recently used
    
    // Arena metadata
    int max_ctx;           // maximum context tokens
    int max_blocks;        // max_ctx / KV_BLOCK_TOKENS
    int arena_blocks;      // total Q4_0 blocks in arena
    size_t arena_bytes_k;  // bytes for K arena
    size_t arena_bytes_v;  // bytes for V arena
    
    // Runtime state
    int cache_len;         // current fill (tokens)
    
    // Compaction state
    bool compaction_pending;
    float fragmentation_ratio;
} kv_arena_t;

/**
 * Initialize KV arena for one GQA layer.
 * Pre-allocates full max_ctx capacity — single cudaMalloc per K/V.
 * 
 * @param max_ctx Maximum context tokens (e.g., 524288 for 512k)
 * @param stream CUDA stream for async initialization
 * @return Arena handle, or NULL on failure
 */
kv_arena_t* kv_arena_init(int max_ctx, cudaStream_t stream);

/**
 * Free all arena resources.
 */
void kv_arena_free(kv_arena_t* arena);

/**
 * Append C tokens to the arena (allocate blocks, map tokens).
 * After calling, use kv_arena_get_token_ptrs to get K/V pointers for each token.
 * 
 * @param arena Arena handle
 * @param C Number of tokens to append
 * @param stream CUDA stream
 * @return 1 on success, 0 on failure
 */
int kv_arena_append(kv_arena_t* arena, int C, cudaStream_t stream);

/**
 * Get device pointers for newly appended tokens' K and V data.
 * Call after kv_arena_append to get write pointers for the C tokens just added.
 * 
 * @param arena Arena handle
 * @param C Number of tokens that were just appended
 * @param k_ptrs Output: array of C device pointers to K data (Q4_0) - caller must free
 * @param v_ptrs Output: array of C device pointers to V data (Q4_0) - caller must free
 * @param stream CUDA stream
 */
void kv_arena_get_append_ptrs(kv_arena_t* arena, int C, 
                              const uint8_t*** k_ptrs, const uint8_t*** v_ptrs,
                              cudaStream_t stream);

/**
 * Get device pointers for a token's K and V data.
 * Uses block table indirection to find physical location in arena.
 * 
 * @param arena Arena handle
 * @param token_idx Logical token index
 * @param k_ptr Output: device pointer to K data (Q4_0)
 * @param v_ptr Output: device pointer to V data (Q4_0)
 * @param stream CUDA stream
 * @return 1 on success, 0 if token not mapped
 */
int kv_arena_get_token_ptrs(kv_arena_t* arena, int token_idx,
                            const uint8_t** k_ptr, const uint8_t** v_ptr,
                            cudaStream_t stream);

/**
 * Current fill level (tokens).
 */
static inline int kv_arena_len(const kv_arena_t* arena) {
    return arena->cache_len;
}

/**
 * Maximum capacity (tokens).
 */
static inline int kv_arena_cap(const kv_arena_t* arena) {
    return arena->max_ctx;
}

#ifdef __cplusplus
}
#endif

#endif // KV_ARENA_H

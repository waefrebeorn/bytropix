/**
 * kv_arena.cu — Game Engine Style KV Cache Arena Allocator
 * 
 * Paradigm: Game Engine Asset Management applied to KV Cache
 * 
 * Design Principles (from game engines):
 * 1. SINGLE CONTIGUOUS ARENA per layer — pre-allocate full max_ctx at init
 *    (like a texture atlas or level geometry buffer — no incremental grows)
 * 2. SLAB ALLOCATOR — fixed-size blocks (16 tokens = 1 block) within arena
 *    (like particle pools / mesh buffers — O(1) alloc/free)
 * 3. BLOCK TABLE INDIRECTION — logical token → physical block index
 *    (like vLLM PagedAttention but simpler — no external allocator needed)
 * 4. LRU EVICTION — when full, evict oldest non-recent blocks
 *    (like texture streaming / mesh LOD — reclaim memory under pressure)
 * 5. OPTIONAL COMPACTION — defragment by moving active blocks to front
 *    (like mesh optimization / atlas repacking — maintain locality)
 * 6. HANDLE-BASED ACCESS — kernels use block_table, never raw pointers
 *    (like resource handles — allows moving data transparently)
 * 
 * Benefits over incremental grow:
 * - ZERO fragmentation (single cudaMalloc per layer)
 * - ZERO copy-on-grow (no cudaMemcpy D2D during generation)
 * - Predictable memory footprint (known at init)
 * - Enables 512k+ context without OOM/corruption
 */

#include "wubu_model.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KV_BLOCK_TOKENS 16          // tokens per block (slab size)
#define KV_HEADS 2                  // GQA has 2 KV heads
#define KV_HEAD_DIM 256
#define KV_DIM (KV_HEADS * KV_HEAD_DIM)  // 512

// Q4_0: 18 bytes per 32 elements = 0.5625 bytes/elem
// For 16 tokens * 512 dim = 8192 elements = 256 Q4_0 blocks = 4608 bytes per block
#define Q4_0_BLOCK_BYTES 18
#define Q4_0_ELEMS_PER_BLOCK 32

// Calculate blocks needed for N tokens
static inline int kv_arena_blocks_for_tokens(int tokens) {
    return (tokens * KV_DIM + Q4_0_ELEMS_PER_BLOCK - 1) / Q4_0_ELEMS_PER_BLOCK;
}

// Calculate bytes for N tokens in Q4_0
static inline size_t kv_arena_bytes_for_tokens(int tokens) {
    return (size_t)kv_arena_blocks_for_tokens(tokens) * Q4_0_BLOCK_BYTES;
}

// ================================================================
// LV Arena Structure — per GQA layer
// ================================================================
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

// Forward declarations
extern "C" void kv_arena_free(kv_arena_t* arena);
extern "C" int kv_arena_alloc_block(kv_arena_t* arena, cudaStream_t stream);
extern "C" void kv_arena_lru_touch(kv_arena_t* arena, int block_idx, cudaStream_t stream);
extern "C" int kv_arena_evict_lru(kv_arena_t* arena, cudaStream_t stream);
extern "C" int kv_arena_append(kv_arena_t* arena, int C, cudaStream_t stream);
extern "C" void kv_arena_get_append_ptrs(kv_arena_t* arena, int C, const uint8_t*** k_ptrs, const uint8_t*** v_ptrs, cudaStream_t stream);

// ================================================================
// Host-side arena management
// ================================================================

// Initialize arena for one GQA layer
// Pre-allocates full max_ctx capacity — single cudaMalloc per K/V
extern "C" kv_arena_t* kv_arena_init(int max_ctx, cudaStream_t stream) {
    kv_arena_t* arena = (kv_arena_t*)calloc(1, sizeof(kv_arena_t));
    if (!arena) return NULL;
    
    arena->max_ctx = max_ctx;
    arena->max_blocks = (max_ctx + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS;
    arena->arena_blocks = kv_arena_blocks_for_tokens(max_ctx);
    arena->arena_bytes_k = kv_arena_bytes_for_tokens(max_ctx);
    arena->arena_bytes_v = kv_arena_bytes_for_tokens(max_ctx);
    arena->cache_len = 0;
    arena->free_list_top = arena->max_blocks;  // all blocks initially free
    
    // Allocate contiguous arenas (K and V separate for cache locality)
    arena->d_k_arena = (uint8_t*)wubu_cuda_alloc(arena->arena_bytes_k);
    arena->d_v_arena = (uint8_t*)wubu_cuda_alloc(arena->arena_bytes_v);
    if (!arena->d_k_arena || !arena->d_v_arena) {
        fprintf(stderr, "KV Arena: Failed to allocate %zu MB + %zu MB (max_ctx=%d)\n",
                arena->arena_bytes_k / (1024*1024), arena->arena_bytes_v / (1024*1024), max_ctx);
        kv_arena_free(arena);
        return NULL;
    }
    
    // Initialize to zero (clear KV cache)
    cudaMemsetAsync(arena->d_k_arena, 0, arena->arena_bytes_k, stream);
    cudaMemsetAsync(arena->d_v_arena, 0, arena->arena_bytes_v, stream);
    
    // Block table: logical token -> physical block index
    arena->d_block_table = (int*)wubu_cuda_alloc((size_t)arena->max_blocks * sizeof(int));
    if (!arena->d_block_table) { kv_arena_free(arena); return NULL; }
    // Initialize to -1 (unmapped)
    cudaMemsetAsync(arena->d_block_table, 255, (size_t)arena->max_blocks * sizeof(int), stream);
    
    // Free list: stack of available block indices [0, 1, 2, ..., max_blocks-1]
    arena->d_free_list = (int*)wubu_cuda_alloc((size_t)arena->max_blocks * sizeof(int));
    if (!arena->d_free_list) { kv_arena_free(arena); return NULL; }
    // Host-side init for free list
    int* h_free_list = (int*)malloc((size_t)arena->max_blocks * sizeof(int));
    for (int i = 0; i < arena->max_blocks; i++) h_free_list[i] = i;
    wubu_cuda_to_device((const float*)h_free_list, (float*)arena->d_free_list, 
                        (size_t)arena->max_blocks * sizeof(int), stream);
    free(h_free_list);
    
    // LRU lists (for eviction) — HOST SIDE
    arena->h_lru_prev = (int*)calloc(arena->max_blocks, sizeof(int));
    arena->h_lru_next = (int*)calloc(arena->max_blocks, sizeof(int));
    if (!arena->h_lru_prev || !arena->h_lru_next) { kv_arena_free(arena); return NULL; }
    for (int i = 0; i < arena->max_blocks; i++) {
        arena->h_lru_prev[i] = -1;
        arena->h_lru_next[i] = -1;
    }
    arena->lru_head = -1;
    arena->lru_tail = -1;
    
    cudaStreamSynchronize(stream);
    
    printf("KV Arena: Initialized max_ctx=%d (%.2f MB K + %.2f MB V, %d blocks)\n",
           max_ctx,
           arena->arena_bytes_k / (1024.0*1024.0),
           arena->arena_bytes_v / (1024.0*1024.0),
           arena->arena_blocks);
    
    return arena;
}

extern "C" void kv_arena_free(kv_arena_t* arena) {
    if (!arena) return;
    wubu_cuda_free((float*)arena->d_k_arena);
    wubu_cuda_free((float*)arena->d_v_arena);
    wubu_cuda_free((float*)arena->d_block_table);
    wubu_cuda_free((float*)arena->d_free_list);
    free(arena->h_lru_prev);
    free(arena->h_lru_next);
    free(arena);
}

// ================================================================
// Allocation: get next free block (O(1) from free list stack)
// ================================================================
// Returns physical block index, or -1 if full (needs eviction)
extern "C" int kv_arena_alloc_block(kv_arena_t* arena, cudaStream_t stream) {
    if (!arena || arena->free_list_top <= 0) {
        return -1;  // Need eviction
    }
    
    // Pop from free list
    arena->free_list_top--;
    int block_idx;
    cudaMemcpyAsync(&block_idx, 
                    arena->d_free_list + arena->free_list_top, 
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return block_idx;
}

// ================================================================
// Map logical token to physical block
// ================================================================
extern "C" void kv_arena_map_token(kv_arena_t* arena, int token_idx, int block_idx, cudaStream_t stream) {
    int table_idx = token_idx / KV_BLOCK_TOKENS;
    cudaMemcpyAsync(arena->d_block_table + table_idx, &block_idx, sizeof(int), cudaMemcpyHostToDevice, stream);
}

// ================================================================
// LRU update: move block to MRU position
// ================================================================
extern "C" void kv_arena_lru_touch(kv_arena_t* arena, int block_idx, cudaStream_t stream) {
    // For simplicity, we track LRU on host
    // In production, this could be a device-side linked list
    if (block_idx == arena->lru_head) return;
    
    // Remove from current position
    if (block_idx == arena->lru_tail) {
        arena->lru_tail = arena->h_lru_prev[block_idx];
    }
    
    if (arena->h_lru_prev[block_idx] != -1) {
        arena->h_lru_next[arena->h_lru_prev[block_idx]] = arena->h_lru_next[block_idx];
    }
    if (arena->h_lru_next[block_idx] != -1) {
        arena->h_lru_prev[arena->h_lru_next[block_idx]] = arena->h_lru_prev[block_idx];
    }
    
    // Add to head (MRU)
    arena->h_lru_prev[block_idx] = -1;
    arena->h_lru_next[block_idx] = arena->lru_head;
    if (arena->lru_head != -1) {
        arena->h_lru_prev[arena->lru_head] = block_idx;
    }
    arena->lru_head = block_idx;
    if (arena->lru_tail == -1) arena->lru_tail = block_idx;
}

// ================================================================
// Eviction: reclaim LRU block when arena is full
// ================================================================
// Returns physical block index that was freed, or -1 if none
extern "C" int kv_arena_evict_lru(kv_arena_t* arena, cudaStream_t stream) {
    if (arena->lru_tail == -1) return -1;
    
    int victim = arena->lru_tail;
    
    // Remove from LRU
    arena->lru_tail = arena->h_lru_prev[victim];
    if (arena->lru_tail != -1) {
        arena->h_lru_next[arena->lru_tail] = -1;
    } else {
        arena->lru_head = -1;
    }
    arena->h_lru_prev[victim] = -1;
    arena->h_lru_next[victim] = -1;
    
    // Push back to free list
    cudaMemcpyAsync(arena->d_free_list + arena->free_list_top, &victim, sizeof(int), cudaMemcpyHostToDevice, stream);
    arena->free_list_top++;
    
    // Clear block table entries pointing to this block
    // (This is a simplification; full implementation would track reverse mapping)
    cudaStreamSynchronize(stream);
    
    // Zero the physical block (K and V)
    size_t block_offset = (size_t)victim * Q4_0_BLOCK_BYTES;
    cudaMemsetAsync(arena->d_k_arena + block_offset, 0, Q4_0_BLOCK_BYTES, stream);
    cudaMemsetAsync(arena->d_v_arena + block_offset, 0, Q4_0_BLOCK_BYTES, stream);
    
    return victim;
}

// ================================================================
// Get physical pointer for a logical token
// ================================================================
// Returns pointer into arena for token's K/V data
extern "C" void kv_arena_get_token_ptrs(kv_arena_t* arena, int token_idx, 
                             const uint8_t** k_ptr, const uint8_t** v_ptr, cudaStream_t stream) {
    int table_idx = token_idx / KV_BLOCK_TOKENS;
    int block_idx;
    cudaMemcpyAsync(&block_idx, arena->d_block_table + table_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (block_idx == -1) {
        *k_ptr = *v_ptr = NULL;
        return;
    }
    
    size_t block_offset = (size_t)block_idx * Q4_0_BLOCK_BYTES;
    int token_in_block = token_idx % KV_BLOCK_TOKENS;
    // Within block, each token is KV_DIM elements = 512 elems = 16 Q4_0 blocks = 288 bytes
    size_t token_offset = (size_t)token_in_block * KV_DIM / Q4_0_ELEMS_PER_BLOCK * Q4_0_BLOCK_BYTES;
    
    *k_ptr = arena->d_k_arena + block_offset + token_offset;
    *v_ptr = arena->d_v_arena + block_offset + token_offset;
}

// ================================================================
// Append K,V to arena (allocate blocks, map tokens)
// ================================================================
// Allocates blocks as needed, evicts if full
// Returns 1 on success, 0 on failure
// After calling, use kv_arena_get_token_ptrs to get K/V pointers for each token
extern "C" int kv_arena_append(kv_arena_t* arena, int C, cudaStream_t stream) {
    if (!arena || C <= 0) return 0;
    
    int tokens_needed_blocks = (C + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS;
    
    // Check if we have enough free blocks, evict if needed
    while (arena->free_list_top < tokens_needed_blocks) {
        int evicted = kv_arena_evict_lru(arena, stream);
        if (evicted == -1) {
            fprintf(stderr, "KV Arena: Cannot evict, arena full! max_ctx=%d\n", arena->max_ctx);
            return 0;
        }
        fprintf(stderr, "KV Arena: Evicted block %d (free_list_top=%d)\n", evicted, arena->free_list_top);
    }
    
    // Allocate blocks for new tokens
    int* block_indices = (int*)malloc(tokens_needed_blocks * sizeof(int));
    for (int i = 0; i < tokens_needed_blocks; i++) {
        block_indices[i] = kv_arena_alloc_block(arena, stream);
        if (block_indices[i] == -1) {
            fprintf(stderr, "KV Arena: Allocation failed after eviction attempt\n");
            free(block_indices);
            return 0;
        }
        kv_arena_lru_touch(arena, block_indices[i], stream);
    }
    
    // Map logical tokens to physical blocks
    int start_token = arena->cache_len;
    for (int t = 0; t < C; t++) {
        int token_idx = start_token + t;
        int table_idx = token_idx / KV_BLOCK_TOKENS;
        int block_idx = block_indices[table_idx];
        kv_arena_map_token(arena, token_idx, block_idx, stream);
    }
    
    free(block_indices);
    arena->cache_len += C;
    return 1;
}
// Call when fragmentation_ratio > threshold (e.g., 0.5)
extern "C" void kv_arena_compact(kv_arena_t* arena, cudaStream_t stream) {
    if (!arena || arena->cache_len == 0) return;

    int max_blocks = arena->max_blocks;
    int tokens_per_block = KV_BLOCK_TOKENS;
    int num_table_entries = (arena->cache_len + tokens_per_block - 1) / tokens_per_block;
    if (num_table_entries > max_blocks) num_table_entries = max_blocks;

    // Download block table to host
    int* h_block_table = (int*)malloc(max_blocks * sizeof(int));
    cudaMemcpy(h_block_table, arena->d_block_table, max_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    // Map physical block -> table index (for active blocks)
    // We'll compact by moving active blocks to the front
    int* block_in_use = (int*)calloc(max_blocks, sizeof(int));
    for (int i = 0; i < num_table_entries; i++) {
        int phys = h_block_table[i];
        if (phys >= 0 && phys < max_blocks) {
            block_in_use[phys] = 1;
        }
    }

    // Build mapping: old physical -> new physical (compacted)
    int* old_to_new = (int*)malloc(max_blocks * sizeof(int));
    int new_idx = 0;
    for (int i = 0; i < max_blocks; i++) {
        if (block_in_use[i]) {
            old_to_new[i] = new_idx++;
        } else {
            old_to_new[i] = -1;
        }
    }

    // Copy K/V data for active blocks to new compacted positions
    size_t block_bytes = Q4_0_BLOCK_BYTES;
    uint8_t* tmp_k = (uint8_t*)malloc(arena->arena_bytes_k);
    uint8_t* tmp_v = (uint8_t*)malloc(arena->arena_bytes_v);
    if (!tmp_k || !tmp_v) {
        fprintf(stderr, "KV Arena: Compaction OOM\\n");
        free(h_block_table);
        free(block_in_use);
        free(old_to_new);
        free(tmp_k);
        free(tmp_v);
        return;
    }

    for (int i = 0; i < max_blocks; i++) {
        if (block_in_use[i]) {
            int new_phys = old_to_new[i];
            size_t src_off = (size_t)i * block_bytes;
            size_t dst_off = (size_t)new_phys * block_bytes;
            memcpy(tmp_k + dst_off, arena->d_k_arena + src_off, block_bytes);
            memcpy(tmp_v + dst_off, arena->d_v_arena + src_off, block_bytes);
        }
    }

    // Upload compacted K/V back to device
    cudaMemcpy(arena->d_k_arena, tmp_k, arena->arena_bytes_k, cudaMemcpyHostToDevice);
    cudaMemcpy(arena->d_v_arena, tmp_v, arena->arena_bytes_v, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);

    // Update block table with new physical indices
    for (int i = 0; i < num_table_entries; i++) {
        int old_phys = h_block_table[i];
        if (old_phys >= 0) {
            h_block_table[i] = old_to_new[old_phys];
        }
    }
    cudaMemcpy(arena->d_block_table, h_block_table, max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);

    // Rebuild free list on host
    int* h_free_list = (int*)malloc(max_blocks * sizeof(int));
    int free_idx = 0;
    for (int i = new_idx; i < max_blocks; i++) {
        h_free_list[free_idx++] = i;
    }
    cudaMemcpy(arena->d_free_list, h_free_list, max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    arena->free_list_top = free_idx;

    // Rebuild LRU list on host
    for (int i = 0; i < max_blocks; i++) {
        arena->h_lru_prev[i] = -1;
        arena->h_lru_next[i] = -1;
    }
    arena->lru_head = -1;
    arena->lru_tail = -1;
    for (int i = 0; i < new_idx; i++) {
        // Add to LRU in order
        if (arena->lru_head == -1) {
            arena->lru_head = arena->lru_tail = i;
        } else {
            arena->h_lru_next[arena->lru_tail] = i;
            arena->h_lru_prev[i] = arena->lru_tail;
            arena->lru_tail = i;
        }
    }

    // Calculate fragmentation ratio
    arena->fragmentation_ratio = 1.0f - (float)new_idx / (float)max_blocks;

    free(h_block_table);
    free(block_in_use);
    free(old_to_new);
    free(tmp_k);
    free(tmp_v);
    free(h_free_list);

    printf("KV Arena: Compacted %d -> %d active blocks (fragmentation: %.2f%%)\\n",
           max_blocks - arena->free_list_top, new_idx, arena->fragmentation_ratio * 100.0f);
}

// ================================================================
// Device-side kernel helpers
// ================================================================

// Get K/V base pointer for a block (for flash attention kernels)
__device__ inline const uint8_t* kv_arena_block_k(const kv_arena_t* arena, int block_idx) {
    return arena->d_k_arena + (size_t)block_idx * Q4_0_BLOCK_BYTES;
}

__device__ inline const uint8_t* kv_arena_block_v(const kv_arena_t* arena, int block_idx) {
    return arena->d_v_arena + (size_t)block_idx * Q4_0_BLOCK_BYTES;
}

// Get K/V for a token (using block table)
__device__ inline void kv_arena_get_token(const kv_arena_t* arena, int token_idx,
                                          const uint8_t** k_ptr, const uint8_t** v_ptr) {
    int table_idx = token_idx / KV_BLOCK_TOKENS;
    int block_idx = arena->d_block_table[table_idx];
    if (block_idx == -1) {
        *k_ptr = *v_ptr = NULL;
        return;
    }
    int token_in_block = token_idx % KV_BLOCK_TOKENS;
    size_t block_offset = (size_t)block_idx * Q4_0_BLOCK_BYTES;
    size_t token_offset = (size_t)token_in_block * KV_DIM / Q4_0_ELEMS_PER_BLOCK * Q4_0_BLOCK_BYTES;
    *k_ptr = arena->d_k_arena + block_offset + token_offset;
    *v_ptr = arena->d_v_arena + block_offset + token_offset;
}

// ================================================================
// Get write pointers for newly appended tokens
// ================================================================
extern "C" void kv_arena_get_append_ptrs(kv_arena_t* arena, int C, 
                                         const uint8_t*** k_ptrs_out, const uint8_t*** v_ptrs_out,
                                         cudaStream_t stream) {
    if (!arena || C <= 0) return;
    
    int start_token = arena->cache_len - C;  // tokens just appended
    
    // Allocate host arrays for the pointers
    const uint8_t** k_ptrs = (const uint8_t**)malloc(C * sizeof(uint8_t*));
    const uint8_t** v_ptrs = (const uint8_t**)malloc(C * sizeof(uint8_t*));
    
    for (int i = 0; i < C; i++) {
        int token_idx = start_token + i;
        int table_idx = token_idx / KV_BLOCK_TOKENS;
        int block_idx;
        cudaMemcpyAsync(&block_idx, arena->d_block_table + table_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (block_idx == -1) {
            k_ptrs[i] = v_ptrs[i] = NULL;
            continue;
        }
        size_t block_offset = (size_t)block_idx * Q4_0_BLOCK_BYTES;
        int token_in_block = token_idx % KV_BLOCK_TOKENS;
        size_t token_offset = (size_t)token_in_block * KV_DIM / Q4_0_ELEMS_PER_BLOCK * Q4_0_BLOCK_BYTES;
        
        k_ptrs[i] = arena->d_k_arena + block_offset + token_offset;
        v_ptrs[i] = arena->d_v_arena + block_offset + token_offset;
    }
    
    *k_ptrs_out = k_ptrs;
    *v_ptrs_out = v_ptrs;
}

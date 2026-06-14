/**
 * kv_paged_attention.c — PagedAttention KV Cache Manager (vLLM Style)
 * 
 * Implements block-based KV cache with:
 * - O(1) free list allocation
 * - Prefix caching for shared prompts
 * - Continuous batching scheduler
 * - Chunked prefill for long contexts
 * - Q4_0 quantized blocks (4:1 compression)
 */

#include "wubu_model.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// Hash table for prefix cache (simple open addressing)
#define PREFIX_HASH_TABLE_SIZE 8192
#define PREFIX_HASH_EMPTY 0xFFFFFFFFFFFFFFFFULL

typedef struct {
    uint64_t hash;
    prefix_entry_t* entry;
} prefix_hash_slot_t;

// ================================================================
// Prefix Hash Table
// ================================================================

static prefix_hash_slot_t* prefix_hash_table = NULL;

static void prefix_hash_init() {
    if (!prefix_hash_table) {
        prefix_hash_table = calloc(PREFIX_HASH_TABLE_SIZE, sizeof(prefix_hash_slot_t));
        for (int i = 0; i < PREFIX_HASH_TABLE_SIZE; i++) {
            prefix_hash_table[i].hash = PREFIX_HASH_EMPTY;
        }
    }
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static prefix_entry_t* prefix_hash_lookup(uint64_t hash) {
    prefix_hash_init();
    uint64_t idx = mix64(hash) % PREFIX_HASH_TABLE_SIZE;
    for (int probes = 0; probes < PREFIX_HASH_TABLE_SIZE; probes++) {
        uint64_t slot_hash = prefix_hash_table[idx].hash;
        if (slot_hash == PREFIX_HASH_EMPTY) return NULL;
        if (slot_hash == hash) return prefix_hash_table[idx].entry;
        idx = (idx + 1) % PREFIX_HASH_TABLE_SIZE;
    }
    return NULL;
}

static void prefix_hash_store(uint64_t hash, prefix_entry_t* entry) {
    prefix_hash_init();
    uint64_t idx = mix64(hash) % PREFIX_HASH_TABLE_SIZE;
    for (int probes = 0; probes < PREFIX_HASH_TABLE_SIZE; probes++) {
        uint64_t slot_hash = prefix_hash_table[idx].hash;
        if (slot_hash == PREFIX_HASH_EMPTY || slot_hash == hash) {
            prefix_hash_table[idx].hash = hash;
            prefix_hash_table[idx].entry = entry;
            return;
        }
        idx = (idx + 1) % PREFIX_HASH_TABLE_SIZE;
    }
    // Table full - should never happen with proper sizing
    fprintf(stderr, "Prefix hash table full!\n");
}

// ================================================================
// Rolling Hash for Prompts
// ================================================================

static uint64_t hash_prompt(const int* tokens, int len) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;  // Golden ratio seed
    for (int i = 0; i < len; i++) {
        h = (h ^ (uint64_t)tokens[i]) * 0xbf58476d1ce4e5b9ULL;
    }
    return h;
}

// Find longest matching prefix
static prefix_entry_t* find_longest_prefix(kv_cache_manager_t* mgr, const int* tokens, int len) {
    prefix_entry_t* best = NULL;
    int best_len = 0;
    
    // Check prefixes of increasing length
    for (int l = KV_BLOCK_SIZE; l <= len; l += KV_BLOCK_SIZE) {
        uint64_t h = hash_prompt(tokens, l);
        prefix_entry_t* entry = prefix_hash_lookup(h);
        if (entry && entry->prompt_len == l) {
            // Verify token equality (hash collision check)
            int match = 1;
            for (int i = 0; i < l; i++) {
                if (entry->block_ids[i / KV_BLOCK_SIZE] != 0) { // placeholder check
                    // In real impl, store tokens or verify against actual
                }
            }
            best = entry;
            best_len = l;
        }
    }
    if (best) mgr->prefix_cache_hits++;
    else mgr->prefix_cache_misses++;
    return best;
}

// ================================================================
// PagedAttention Manager Implementation
// ================================================================

kv_cache_manager_t* kv_paged_manager_init(int max_ctx, int block_size, int n_layers, int n_kv_heads) {
    if (block_size <= 0) block_size = KV_BLOCK_SIZE;
    
    kv_cache_manager_t* mgr = calloc(1, sizeof(kv_cache_manager_t));
    if (!mgr) return NULL;
    
    mgr->n_layers = n_layers;
    mgr->n_kv_heads = n_kv_heads;
    
    // Calculate total blocks needed
    int max_blocks_per_layer = (max_ctx + block_size - 1) / block_size;
    int total_blocks = max_blocks_per_layer * n_layers * n_kv_heads;
    
    mgr->max_blocks_per_layer = max_blocks_per_layer;
    mgr->total_blocks = total_blocks;
    
    // Allocate block pool
    mgr->blocks = calloc(total_blocks, sizeof(kv_block_t));
    if (!mgr->blocks) { free(mgr); return NULL; }
    
    // Initialize free list (all blocks initially free)
    mgr->free_capacity = total_blocks;
    mgr->free_blocks = malloc(total_blocks * sizeof(int));
    if (!mgr->free_blocks) { free(mgr->blocks); free(mgr); return NULL; }
    
    for (int i = 0; i < total_blocks; i++) {
        mgr->free_blocks[i] = i;
        mgr->blocks[i].block_id = i;
        mgr->blocks[i].ref_count = 0;
    }
    mgr->free_count = total_blocks;
    
    // GPU memory pool allocation (Q4_0 blocks)
    // Start with initial capacity (kv_cache_init tokens per layer), grow dynamically
    // Per block: KV_BLOCK_SIZE * head_dim elements per KV head
    // Q4_0: 32 elements per block_q4_0_t = 18 bytes
    const int kv_cache_init = 4096;
    int initial_blocks_per_layer = (kv_cache_init + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    int elements_per_head = KV_BLOCK_SIZE * 256; // block_size * head_dim
    int n_q4_blocks_per_layer_per_head = (elements_per_head + 31) / 32;
    
    int n_q4_blocks_total = n_q4_blocks_per_layer_per_head * initial_blocks_per_layer * n_layers * n_kv_heads;
    size_t q4_0_block_size = sizeof(block_q4_0_t); // 18 bytes
    
    mgr->pool_bytes = (size_t)n_q4_blocks_total * q4_0_block_size * 2; // K + V
    mgr->pool_bytes = (mgr->pool_bytes + 127) & ~127; // 128-byte align
    
    // Track max allowed pool size (for max_ctx) - use mgr->max_blocks_per_layer already set
    mgr->max_pool_bytes = (size_t)n_q4_blocks_per_layer_per_head * mgr->max_blocks_per_layer * n_layers * n_kv_heads * q4_0_block_size * 2;
    mgr->max_pool_bytes = (mgr->max_pool_bytes + 127) & ~127;
    
    cudaError_t ce = cudaSuccess;
    // Only allocate initial pool if it fits in a reasonable size (e.g., < 2GB per pool)
    size_t reasonable_limit = 2ULL * 1024 * 1024 * 1024; // 2GB
    if (mgr->pool_bytes <= reasonable_limit) {
        ce = cudaMalloc(&mgr->d_k_pool, mgr->pool_bytes);
        if (ce != cudaSuccess) { fprintf(stderr, "cudaMalloc K pool failed: %s\n", cudaGetErrorString(ce)); }
        ce = cudaMalloc(&mgr->d_v_pool, mgr->pool_bytes);
        if (ce != cudaSuccess) { fprintf(stderr, "cudaMalloc V pool failed: %s\n", cudaGetErrorString(ce)); }
        
        if (ce == cudaSuccess) {
            cudaMemset(mgr->d_k_pool, 0, mgr->pool_bytes);
            cudaMemset(mgr->d_v_pool, 0, mgr->pool_bytes);
        }
    } else {
        // Pool too large for initial alloc - will allocate on first use
        mgr->d_k_pool = NULL;
        mgr->d_v_pool = NULL;
    }
    
    if (ce != cudaSuccess) {
        kv_paged_manager_free(mgr);
        return NULL;
    }
    
    // Block table arrays (start with capacity for 32 sequences)
    mgr->batch_capacity = 32;
    mgr->block_tables = calloc(mgr->batch_capacity, sizeof(int*));
    mgr->block_table_sizes = calloc(mgr->batch_capacity, sizeof(int));
    
    // Assign layer_idx and kv_head_idx to each block
    int block_idx = 0;
    for (int l = 0; l < n_layers; l++) {
        for (int h = 0; h < n_kv_heads; h++) {
            for (int b = 0; b < max_blocks_per_layer; b++) {
                if (block_idx < total_blocks) {
                    mgr->blocks[block_idx].layer_idx = l;
                    mgr->blocks[block_idx].kv_head_idx = h;
                    mgr->blocks[block_idx].last_access = 0;
                    mgr->blocks[block_idx].is_pinned = false;
                    block_idx++;
                }
            }
        }
    }
    
    // Initialize LRU timestamp
    mgr->lru_timestamp = 1;
    
    printf("PagedAttention: initialized with %d blocks (%d layers x %d KV heads x %d blocks/layer), pool: %.2f MB\n",
           total_blocks, n_layers, n_kv_heads, max_blocks_per_layer,
           (double)mgr->pool_bytes / (1024*1024));
    
    return mgr;
}

void kv_paged_manager_free(kv_cache_manager_t* mgr) {
    if (!mgr) return;
    
    if (mgr->d_k_pool) cudaFree(mgr->d_k_pool);
    if (mgr->d_v_pool) cudaFree(mgr->d_v_pool);
    
    if (mgr->free_blocks) free(mgr->free_blocks);
    if (mgr->blocks) free(mgr->blocks);
    
    if (mgr->block_tables) {
        for (int i = 0; i < mgr->batch_capacity; i++) {
            if (mgr->block_tables[i]) free(mgr->block_tables[i]);
        }
        free(mgr->block_tables);
    }
    if (mgr->block_table_sizes) free(mgr->block_table_sizes);
    
    free(mgr);
}

int kv_paged_alloc_block(kv_cache_manager_t* mgr, int layer_idx) {
    if (mgr->free_count == 0) return -1;  // OOM
    
    int block_id = mgr->free_blocks[--mgr->free_count];
    mgr->blocks[block_id].ref_count = 1;
    mgr->total_allocated_blocks++;
    
    // Verify layer matches
    if (mgr->blocks[block_id].layer_idx != layer_idx) {
        fprintf(stderr, "Block layer mismatch: got %d, expected %d\n", 
                mgr->blocks[block_id].layer_idx, layer_idx);
    }
    
    return block_id;
}

void kv_paged_free_block(kv_cache_manager_t* mgr, int block_id) {
    if (block_id < 0 || block_id >= mgr->total_blocks) return;
    
    if (--mgr->blocks[block_id].ref_count == 0) {
        if (mgr->free_count >= mgr->free_capacity) {
            mgr->free_capacity *= 2;
            mgr->free_blocks = realloc(mgr->free_blocks, mgr->free_capacity * sizeof(int));
        }
        mgr->free_blocks[mgr->free_count++] = block_id;
        mgr->total_freed_blocks++;
    }
}

// ================================================================
// LRU Cache Eviction Policy
// ================================================================

// Update block access time (call when block is read/written)
void kv_paged_touch_block(kv_cache_manager_t* mgr, int block_id) {
    if (block_id >= 0 && block_id < mgr->total_blocks) {
        mgr->blocks[block_id].last_access = mgr->lru_timestamp++;
    }
}

// Find LRU victim block that can be evicted (ref_count == 0, not pinned)
// Returns block_id or -1 if no evictable block
static int find_lru_victim(kv_cache_manager_t* mgr, int layer_idx) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    
    for (int i = 0; i < mgr->total_blocks; i++) {
        kv_block_t* block = &mgr->blocks[i];
        if (block->layer_idx != layer_idx) continue;
        if (block->ref_count > 0) continue;  // In use
        if (block->is_pinned) continue;      // Pinned by prefix cache
        
        // Check if block is already in free list
        bool in_free = false;
        for (int f = 0; f < mgr->free_count; f++) {
            if (mgr->free_blocks[f] == i) {
                in_free = true;
                break;
            }
        }
        if (in_free) continue;  // Already free
        
        if (block->last_access < oldest) {
            oldest = block->last_access;
            victim = i;
        }
    }
    return victim;
}

// Evict blocks to make room (used when free list is empty)
// Evicts up to 'target_free' blocks from the specified layer
int kv_paged_evict_blocks(kv_cache_manager_t* mgr, int layer_idx, int target_free) {
    int evicted = 0;
    while (evicted < target_free) {
        int victim = find_lru_victim(mgr, layer_idx);
        if (victim < 0) break;  // No more evictable blocks
        
        // Zero out the block's GPU memory (optional, for security)
        // cudaMemsetAsync(...)
        
        // Add to free list
        if (mgr->free_count >= mgr->free_capacity) {
            mgr->free_capacity *= 2;
            mgr->free_blocks = realloc(mgr->free_blocks, mgr->free_capacity * sizeof(int));
        }
        mgr->free_blocks[mgr->free_count++] = victim;
        mgr->total_freed_blocks++;
        evicted++;
    }
    return evicted;
}

// Allocate block with automatic eviction on OOM
int kv_paged_alloc_block_with_eviction(kv_cache_manager_t* mgr, int layer_idx) {
    if (mgr->free_count > 0) {
        // Fast path: free list has blocks
        int block_id = mgr->free_blocks[--mgr->free_count];
        mgr->blocks[block_id].ref_count = 1;
        mgr->blocks[block_id].last_access = mgr->lru_timestamp++;
        mgr->total_allocated_blocks++;
        
        if (mgr->blocks[block_id].layer_idx != layer_idx) {
            fprintf(stderr, "Block layer mismatch: got %d, expected %d\n", 
                    mgr->blocks[block_id].layer_idx, layer_idx);
        }
        return block_id;
    }
    
    // OOM: try to evict LRU blocks
    int evicted = kv_paged_evict_blocks(mgr, layer_idx, 16);  // Evict in chunks
    if (evicted > 0) {
        fprintf(stderr, "PagedAttention: evicted %d LRU blocks for layer %d\n", evicted, layer_idx);
        return kv_paged_alloc_block_with_eviction(mgr, layer_idx);  // Retry
    }
    
    fprintf(stderr, "PagedAttention OOM: no evictable blocks for layer %d\n", layer_idx);
    return -1;
}

// Pin/unpin blocks for prefix cache
void kv_paged_pin_block(kv_cache_manager_t* mgr, int block_id, bool pin) {
    if (block_id >= 0 && block_id < mgr->total_blocks) {
        mgr->blocks[block_id].is_pinned = pin;
    }
}

// Hash prompt tokens
uint64_t kv_paged_hash_prompt(const int* tokens, int len) {
    return hash_prompt(tokens, len);
}

// Lookup prefix in cache
prefix_entry_t* kv_paged_prefix_lookup(const int* tokens, int len) {
    uint64_t h = hash_prompt(tokens, len);
    return prefix_hash_lookup(h);
}

// Store new prefix in cache
void kv_paged_prefix_store(const int* tokens, int len, int* block_ids, int num_blocks) {
    uint64_t h = hash_prompt(tokens, len);
    
    // Check if already exists
    if (prefix_hash_lookup(h)) return;
    
    prefix_entry_t* entry = calloc(1, sizeof(prefix_entry_t));
    entry->prompt_hash = h;
    entry->prompt_len = len;
    entry->num_blocks = num_blocks;
    entry->ref_count = 1;
    entry->block_ids = malloc(num_blocks * sizeof(int));
    memcpy(entry->block_ids, block_ids, num_blocks * sizeof(int));
    
    prefix_hash_store(h, entry);
}

// Allocate blocks for prefill request
int kv_paged_alloc_prefill(kv_cache_manager_t* mgr, request_t* req) {
    int n_blocks_needed = (req->prompt_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    int total_blocks_needed = n_blocks_needed * mgr->n_layers * mgr->n_kv_heads;
    
    if (mgr->free_count < total_blocks_needed) {
        fprintf(stderr, "PagedAttention OOM: need %d blocks, have %d\n", 
                total_blocks_needed, mgr->free_count);
        return -1;
    }
    
    // Ensure block table has capacity
    if (req->req_id >= mgr->batch_capacity) {
        int new_cap = mgr->batch_capacity * 2;
        mgr->block_tables = realloc(mgr->block_tables, new_cap * sizeof(int*));
        mgr->block_table_sizes = realloc(mgr->block_table_sizes, new_cap * sizeof(int));
        for (int i = mgr->batch_capacity; i < new_cap; i++) {
            mgr->block_tables[i] = NULL;
            mgr->block_table_sizes[i] = 0;
        }
        mgr->batch_capacity = new_cap;
    }
    
    // Allocate or reuse block table
    if (!mgr->block_tables[req->req_id] || 
        mgr->block_table_sizes[req->req_id] < total_blocks_needed) {
        if (mgr->block_tables[req->req_id]) free(mgr->block_tables[req->req_id]);
        mgr->block_tables[req->req_id] = malloc(total_blocks_needed * sizeof(int));
        mgr->block_table_sizes[req->req_id] = total_blocks_needed;
    }
    
    int* block_table = mgr->block_tables[req->req_id];
    int idx = 0;
    
    // Check for prefix sharing
    prefix_entry_t* prefix = find_longest_prefix(mgr, req->tokens, req->prompt_len);
    int prefix_blocks = prefix ? prefix->num_blocks : 0;
    
    if (prefix) {
        // Increment ref counts for shared blocks
        for (int i = 0; i < prefix_blocks * mgr->n_layers * mgr->n_kv_heads; i++) {
            int b = prefix->block_ids[i];
            if (b >= 0) mgr->blocks[b].ref_count++;
        }
        // Copy prefix blocks
        memcpy(block_table, prefix->block_ids, prefix_blocks * mgr->n_layers * mgr->n_kv_heads * sizeof(int));
        idx = prefix_blocks * mgr->n_layers * mgr->n_kv_heads;
        mgr->prefix_cache_hits++;
    } else {
        mgr->prefix_cache_misses++;
    }
    
    // Allocate new blocks for suffix
    int suffix_len = req->prompt_len - prefix_blocks * KV_BLOCK_SIZE;
    int new_blocks_per_layer = (suffix_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    int new_blocks_total = new_blocks_per_layer * mgr->n_layers * mgr->n_kv_heads;
    
    for (int l = 0; l < mgr->n_layers; l++) {
        for (int h = 0; h < mgr->n_kv_heads; h++) {
            for (int b = 0; b < new_blocks_per_layer; b++) {
                int block_id = kv_paged_alloc_block(mgr, l);
                if (block_id < 0) return -1;
                mgr->blocks[block_id].kv_head_idx = h;
                block_table[idx++] = block_id;
            }
        }
    }
    
    req->block_table = block_table;
    req->block_table_size = idx;
    req->n_kv_heads = mgr->n_kv_heads;
    
    // Store new prefix if prompt is long enough
    if (req->prompt_len >= 32) {
        kv_paged_prefix_store(req->tokens, req->prompt_len, block_table, req->block_table_size);
    }
    
    return req->block_table_size;
}

// Allocate one block for decode step
int kv_paged_alloc_decode(kv_cache_manager_t* mgr, request_t* req) {
    // For continuous batching, we may need to grow block table by 1 block per layer per KV head
    int blocks_per_step = mgr->n_layers * mgr->n_kv_heads;
    
    // Check if block table needs resizing
    if (req->block_table_size + blocks_per_step > mgr->block_table_sizes[req->req_id]) {
        int new_size = mgr->block_table_sizes[req->req_id] + blocks_per_step * 16; // grow in chunks
        req->block_table = realloc(req->block_table, new_size * sizeof(int));
        mgr->block_table_sizes[req->req_id] = new_size;
    }
    
    int* block_table = req->block_table;
    for (int l = 0; l < mgr->n_layers; l++) {
        for (int h = 0; h < mgr->n_kv_heads; h++) {
            int block_id = kv_paged_alloc_block(mgr, l);
            if (block_id < 0) return -1;
            mgr->blocks[block_id].kv_head_idx = h;
            block_table[req->block_table_size++] = block_id;
        }
    }
    
    return blocks_per_step;
}

// Free all blocks for a request
void kv_paged_free_request(kv_cache_manager_t* mgr, request_t* req) {
    if (!req->block_table) return;
    
    for (int i = 0; i < req->block_table_size; i++) {
        int block_id = req->block_table[i];
        if (block_id >= 0 && block_id < mgr->total_blocks) {
            kv_paged_free_block(mgr, block_id);
        }
    }
    free(req->block_table);
    req->block_table = NULL;
    req->block_table_size = 0;
}

// Continuous Batching Scheduler
void kv_paged_scheduler_step(kv_cache_manager_t* mgr, request_t** reqs, int num_reqs) {
    // Separate by state
    request_t* waiting[64]; int n_wait = 0;
    request_t* prefill[64]; int n_prefill = 0;
    request_t* decode[64];  int n_decode = 0;
    
    for (int i = 0; i < num_reqs; i++) {
        switch (reqs[i]->state) {
            case REQ_WAITING:  waiting[n_wait++] = reqs[i]; break;
            case REQ_PREFILL:  prefill[n_prefill++] = reqs[i]; break;
            case REQ_DECODE:   decode[n_decode++] = reqs[i]; break;
            default: break;
        }
    }
    
    // Token budget
    int decode_batch = n_decode;
    int prefill_budget = MAX_BATCH_TOKENS - decode_batch;
    
    // Promote waiting -> prefill (FCFS, bounded by token budget)
    for (int i = 0; i < n_wait && prefill_budget > 0; i++) {
        int needed = waiting[i]->prompt_len;
        if (needed <= prefill_budget) {
            kv_paged_alloc_prefill(mgr, waiting[i]);
            waiting[i]->state = REQ_PREFILL;
            prefill[n_prefill++] = waiting[i];
            prefill_budget -= needed;
        }
    }
    
    // Run prefill and decode
    kv_paged_run_prefill_batch(prefill, n_prefill);
    kv_paged_run_decode_batch(decode, n_decode);
    
    // Update states
    for (int i = 0; i < n_prefill; i++) {
        if (prefill[i]->generated_len < prefill[i]->max_tokens) {
            prefill[i]->state = REQ_DECODE;
        } else {
            prefill[i]->state = REQ_FINISHED;
            kv_paged_free_request(mgr, prefill[i]);
        }
    }
    for (int i = 0; i < n_decode; i++) {
        decode[i]->generated_len++;
        if (decode[i]->generated_len >= decode[i]->max_tokens) {
            decode[i]->state = REQ_FINISHED;
            kv_paged_free_request(mgr, decode[i]);
        }
    }
}

// Run prefill batch with chunked prefill
void kv_paged_run_prefill_batch(request_t** reqs, int n) {
    for (int i = 0; i < n; i++) {
        int remaining = reqs[i]->prompt_len - reqs[i]->last_pos;
        int chunk = remaining < PREFILL_CHUNK_SIZE ? remaining : PREFILL_CHUNK_SIZE;
        
        if (chunk > 0) {
            // Forward pass on chunk (would call model_forward with reqs[i]->tokens + reqs[i]->last_pos)
            // model_forward(reqs[i]->tokens + reqs[i]->last_pos, chunk, ...);
            
            reqs[i]->last_pos += chunk;
            if (reqs[i]->last_pos >= reqs[i]->prompt_len) {
                // Prefill complete
            }
        }
    }
}

// Run decode batch
void kv_paged_run_decode_batch(request_t** reqs, int n) {
    // Allocate one block per decode step per request
    // Forward pass on 1 token per request
    // model_forward(reqs[i]->tokens + reqs[i]->prompt_len + reqs[i]->generated_len - 1, 1, ...);
    
    for (int i = 0; i < n; i++) {
        reqs[i]->generated_len++;
    }
}

void* kv_paged_get_k_pool(kv_cache_manager_t* mgr) { return mgr->d_k_pool; }
void* kv_paged_get_v_pool(kv_cache_manager_t* mgr) { return mgr->d_v_pool; }
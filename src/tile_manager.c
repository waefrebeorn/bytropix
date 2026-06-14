/**
 * tile_manager.c — Framebuffer-Style KV Tile Manager
 * 
 * Treats KV cache like GPU framebuffer:
 * - Tiles = 64×64 "pixel" blocks (64 tokens each)
 * - Resident set = active tiles in VRAM
 * - Frame planning = prefetch next, evict LRU
 * - Contiguous tile access = coalesced memory
 */

#include "wubu_turboquant.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// ================================================================
// Tile Manager Implementation
// ================================================================

tile_manager_t* tile_manager_init(int max_ctx, int block_size, int n_layers, int n_kv_heads) {
    (void)block_size;  // Use TILE_SIZE = 64
    
    tile_manager_t* mgr = calloc(1, sizeof(tile_manager_t));
    if (!mgr) return NULL;
    
    int n_tiles = max_ctx / TILE_SIZE;
    if (n_tiles > MAX_TILES) n_tiles = MAX_TILES;
    
    mgr->free_capacity = n_tiles;
    mgr->free_blocks = malloc(n_tiles * sizeof(int));
    mgr->free_count = n_tiles;
    
    for (int i = 0; i < n_tiles; i++) {
        mgr->free_blocks[i] = i;
        mgr->tiles[i].tile_id = i;
        mgr->tiles[i].is_resident = false;
        mgr->tiles[i].last_access_frame = -1;
        mgr->tiles[i].layer = i / (n_kv_heads * (n_tiles / n_layers));
    }
    
    // GPU tile pools
    // Per tile: TILE_SIZE * 256 elements (K: Q4_0, V: Q2_0)
    // K: 64 * 256 / 32 * 18 = 9216 bytes per tile per KV head
    // V: 64 * 256 / 32 * 10 = 5120 bytes per tile per KV head
    int tiles_per_layer = n_tiles / n_layers;
    int tiles_per_kv_head = tiles_per_layer / n_kv_heads;
    
    size_t k_tile_bytes = (size_t)TILE_SIZE * 256 / 32 * sizeof(block_q4_0_t);
    size_t v_tile_bytes = (size_t)TILE_SIZE * 256 / 32 * sizeof(block_q2_0_t);
    size_t total_k = (size_t)n_layers * 2 * tiles_per_kv_head * k_tile_bytes;
    size_t total_v = (size_t)n_layers * 2 * tiles_per_kv_head * v_tile_bytes;
    
    mgr->tile_pool_bytes = total_k + total_v;
    cudaMalloc(&mgr->d_k_tile_pool, total_k);
    cudaMalloc(&mgr->d_v_tile_pool, total_v);
    
    cudaMemset(mgr->d_k_tile_pool, 0, total_k);
    cudaMemset(mgr->d_v_tile_pool, 0, total_v);
    
    // Assign tile pointers
    size_t k_offset = 0, v_offset = 0;
    for (int l = 0; l < 40; l++) {
        for (int h = 0; h < 2; h++) {
            for (int t = 0; t < tiles_per_kv_head; t++) {
                int tile_id = l * 2 * tiles_per_kv_head + h * tiles_per_kv_head + t;
                if (tile_id < MAX_TILES) {
                    mgr->tiles[tile_id].layer = l;
                    mgr->tiles[tile_id].kv_head = h;
                    mgr->tiles[tile_id].block_start = -1;
                    mgr->tiles[tile_id].k_tile_ptr = (char*)mgr->d_k_tile_pool + k_offset;
                    mgr->tiles[tile_id].v_tile_ptr = (char*)mgr->d_v_tile_pool + v_offset;
                    k_offset += k_tile_bytes;
                    v_offset += v_tile_bytes;
                }
            }
        }
    }
    
    // Batch capacity
    mgr->batch_capacity = 32;
    mgr->tile_tables = calloc(mgr->batch_capacity, sizeof(int*));
    mgr->tile_table_sizes = calloc(mgr->batch_capacity, sizeof(int));
    
    mgr->current_frame = 0;
    mgr->window_tiles = 8192 / TILE_SIZE;  // 128 tiles for 8k window
    
    printf("TileManager: %d tiles (%d tiles/frame), K pool: %.2f MB, V pool: %.2f MB\n",
           n_tiles, mgr->window_tiles,
           (double)total_k / (1024*1024),
           (double)total_v / (1024*1024));
    
    return mgr;
}

void tile_manager_free(tile_manager_t* mgr) {
    if (!mgr) return;
    if (mgr->d_k_tile_pool) cudaFree(mgr->d_k_tile_pool);
    if (mgr->d_v_tile_pool) cudaFree(mgr->d_v_tile_pool);
    if (mgr->free_blocks) free(mgr->free_blocks);
    if (mgr->tile_tables) {
        for (int i = 0; i < mgr->batch_capacity; i++) {
            if (mgr->tile_tables[i]) free(mgr->tile_tables[i]);
        }
        free(mgr->tile_tables);
    }
    if (mgr->tile_table_sizes) free(mgr->tile_table_sizes);
    free(mgr);
}

// Allocate a new tile (O(1) free list)
int tile_manager_alloc_tile(tile_manager_t* mgr, int layer, int kv_head) {
    if (mgr->free_count == 0) return -1;
    
    int tile_id = mgr->free_blocks[--mgr->free_count];
    mgr->tiles[tile_id].layer = layer;
    mgr->tiles[tile_id].kv_head = kv_head;
    mgr->tiles[tile_id].is_resident = true;
    mgr->tiles[tile_id].last_access_frame = mgr->current_frame;
    
    return tile_id;
}

void tile_manager_free_tile(tile_manager_t* mgr, int tile_id) {
    if (tile_id < 0 || tile_id >= MAX_TILES) return;
    
    mgr->tiles[tile_id].is_resident = false;
    mgr->tiles[tile_id].layer = -1;
    mgr->tiles[tile_id].kv_head = -1;
    mgr->tiles[tile_id].block_start = -1;
    
    if (mgr->free_count >= mgr->free_capacity) {
        mgr->free_capacity *= 2;
        mgr->free_blocks = realloc(mgr->free_blocks, mgr->free_capacity * sizeof(int));
    }
    mgr->free_blocks[mgr->free_count++] = tile_id;
}

// Frame planning: prefetch active tiles, evict old
void tile_manager_plan_frame(tile_manager_t* mgr, int req_id, int current_pos, int window_size) {
    mgr->current_frame++;
    
    // Active tiles: current position window
    int window_tiles = (window_size + TILE_SIZE - 1) / TILE_SIZE;
    int start_tile = (current_pos - window_size) / TILE_SIZE;
    if (start_tile < 0) start_tile = 0;
    int end_tile = current_pos / TILE_SIZE;
    
    // Prefetch next tiles (beyond current window for next step)
    int prefetch_start = end_tile + 1;
    int prefetch_end = min(prefetch_start + mgr->window_tiles / 4, end_tile + mgr->window_tiles);
    
    // Mark tiles as accessed this frame
    for (int t = start_tile; t <= end_tile; t++) {
        if (t < MAX_TILES) {
            mgr->tiles[t].last_access_frame = mgr->current_frame;
        }
    }
    // Prefetch
    for (int t = prefetch_start; t <= prefetch_end; t++) {
        if (t < MAX_TILES && !mgr->tiles[t].is_resident) {
            // Async copy from host/swap to tile pool
            // tile_manager_prefetch_async(mgr, t);
        }
    }
    
    // Evict LRU tiles outside active window
    int evict_threshold = mgr->current_frame - mgr->window_tiles * 2;
    for (int t = 0; t < MAX_TILES; t++) {
        if (mgr->tiles[t].is_resident && mgr->tiles[t].last_access_frame < evict_threshold) {
            // Check if tile is not in active window
            if (t < start_tile - 10 || t > end_tile + 10) {
                tile_manager_free_tile(mgr, t);
            }
        }
    }
}

// Async prefetch from host/swap
void tile_manager_prefetch_async(tile_manager_t* mgr, int tile_id) {
    if (tile_id < 0 || tile_id >= MAX_TILES) return;
    mgr->tiles[tile_id].is_resident = true;
    // cudaMemcpyAsync from staging buffer to mgr->tiles[tile_id].k_tile_ptr
}

// Get device pointers for active tiles (for kernel launch)
void tile_manager_get_active_tile_ptrs(tile_manager_t* mgr, int req_id, 
                                       void** d_k_tiles, void** d_v_tiles, 
                                       int* n_tiles) {
    int current_pos = 0;  // Would come from request
    int end_tile = current_pos / TILE_SIZE;
    int start_tile = max(0, end_tile - mgr->window_tiles);
    
    *n_tiles = 0;
    for (int t = start_tile; t <= end_tile; t++) {
        if (t < MAX_TILES && mgr->tiles[t].is_resident) {
            d_k_tiles[*n_tiles] = mgr->tiles[t].k_tile_ptr;
            d_v_tiles[*n_tiles] = mgr->tiles[t].v_tile_ptr;
            (*n_tiles)++;
        }
    }
}
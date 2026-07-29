/*
 * wubu_turboquant.c — TurboQuant tile manager implementation
 *
 * Backend for wubu_turboquant.h: manages KV-cache tile allocation,
 * frame-based planning, and LRU eviction for the TurboQuant+/RotorQuant
 * quantization scheme (Q2_0 for V cache, Q4_0 for K cache).
 */

#include "wubu_turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

tile_manager_t *tile_manager_init(int max_ctx, int block_size, int n_layers, int n_kv_heads)
{
    tile_manager_t *mgr = (tile_manager_t *)calloc(1, sizeof(tile_manager_t));
    if (!mgr) return NULL;

    mgr->max_ctx = max_ctx;
    mgr->block_size = block_size;
    mgr->n_layers = n_layers;
    mgr->n_kv_heads = n_kv_heads;
    mgr->free_capacity = MAX_TILES;
    mgr->free_count = MAX_TILES;

    for (int i = 0; i < MAX_TILES; i++) {
        mgr->free_list[i] = i;
        mgr->tiles[i].tile_id = i;
        mgr->tiles[i].is_resident = false;
        mgr->tiles[i].last_access_frame = 0;
    }

    mgr->current_frame = 0;
    return mgr;
}

void tile_manager_free(tile_manager_t *mgr)
{
    if (!mgr) return;
    if (mgr->d_k_tile_pool) free(mgr->d_k_tile_pool);
    if (mgr->d_v_tile_pool) free(mgr->d_v_tile_pool);
    if (mgr->tile_tables) {
        for (int i = 0; i < mgr->batch_capacity; i++)
            free(mgr->tile_tables[i]);
        free(mgr->tile_tables);
    }
    free(mgr->tile_table_sizes);
    free(mgr);
}

int tile_manager_alloc_tile(tile_manager_t *mgr, int layer, int kv_head)
{
    if (mgr->free_count <= 0) return -1;

    int tile_id = mgr->free_list[--mgr->free_count];
    mgr->tiles[tile_id].layer = layer;
    mgr->tiles[tile_id].kv_head = kv_head;
    mgr->tiles[tile_id].is_resident = true;
    mgr->tiles[tile_id].last_access_frame = mgr->current_frame;
    mgr->tiles[tile_id].block_start = tile_id * TILES_PER_WINDOW;
    return tile_id;
}

void tile_manager_free_tile(tile_manager_t *mgr, int tile_id)
{
    if (tile_id < 0 || tile_id >= MAX_TILES) return;
    if (!mgr->tiles[tile_id].is_resident) return;

    mgr->tiles[tile_id].is_resident = false;
    mgr->free_list[mgr->free_count++] = tile_id;
}

void tile_manager_plan_frame(tile_manager_t *mgr, int req_id, int current_pos, int window_size)
{
    mgr->current_frame++;
    mgr->window_tiles = (window_size + mgr->block_size - 1) / mgr->block_size;

    /* Evict tiles outside the current window using LRU */
    for (int i = 0; i < MAX_TILES; i++) {
        if (!mgr->tiles[i].is_resident) continue;
        if (mgr->tiles[i].last_access_frame < mgr->current_frame - mgr->window_tiles) {
            tile_manager_free_tile(mgr, i);
        }
    }
}
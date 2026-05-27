#ifndef MTP_Q8_CACHE_H
#define MTP_Q8_CACHE_H

#include "gguf_reader.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Simple 8-slot direct-mapped cache: stores Q8_0-dequantized expert weights.
// Key = expert_index, Value = 3 Q8_0 weight blobs (gate, up, down) + LRU timestamp.
// Each Q8_0 blob = Q8_0-block-encoded weights (2048×512 → 1.1MB).
#define MTP_Q8_CACHE_SLOTS 12  // 8 routed + 1 shared + 3 spare

// Size of one Q8_0-encoded weight matrix: D_MODEL × D_FF as block_q8_K blocks
#define MTP_Q8_WEIGHT_BYTES ((int)(((int64_t)D_MODEL * D_FF + 255) / 256 * (int64_t)sizeof(block_q8_K)))

typedef struct {
    int   key;               // expert index (-1 = empty)
    int   lru_tick;           // last access tick
    uint8_t gate_q8[MTP_Q8_WEIGHT_BYTES];  // Q8_0-coded gate weight [D_MODEL, D_FF]
    uint8_t up_q8[MTP_Q8_WEIGHT_BYTES];    // Q8_0-coded up weight [D_MODEL, D_FF]
    uint8_t down_q8[MTP_Q8_WEIGHT_BYTES];  // Q8_0-coded down weight [D_FF, D_MODEL]
} mtp_q8_cache_slot_t;

typedef struct {
    mtp_q8_cache_slot_t slots[MTP_Q8_CACHE_SLOTS];
    int tick;
} mtp_q8_cache_t;

// Initialize cache
static inline void mtp_q8_cache_init(mtp_q8_cache_t *cache) {
    memset(cache, 0, sizeof(*cache));
    for (int i = 0; i < MTP_Q8_CACHE_SLOTS; i++)
        cache->slots[i].key = -1;
    cache->tick = 0;
}

// Find slot for given key, or evict LRU
// was_miss: set to true if a new slot was allocated (caller must fill)
static inline mtp_q8_cache_slot_t *mtp_q8_cache_get_slot(mtp_q8_cache_t *cache, int key, bool *was_miss) {
    cache->tick++;
    
    // Check for existing entry
    for (int i = 0; i < MTP_Q8_CACHE_SLOTS; i++) {
        if (cache->slots[i].key == key) {
            cache->slots[i].lru_tick = cache->tick;
            *was_miss = false;
            return &cache->slots[i];
        }
    }
    
    // Find LRU slot (empty slot = lru_tick==0)
    int lru_idx = 0;
    int min_tick = cache->slots[0].lru_tick;
    for (int i = 1; i < MTP_Q8_CACHE_SLOTS; i++) {
        if (cache->slots[i].lru_tick < min_tick) {
            min_tick = cache->slots[i].lru_tick;
            lru_idx = i;
        }
    }
    
    mtp_q8_cache_slot_t *slot = &cache->slots[lru_idx];
    slot->key = key;
    slot->lru_tick = cache->tick;
    *was_miss = true;
    return slot;
}

// Dequantize one expert weight block from source quant type to Q8_0 in slot
static inline void mtp_q8_cache_fill(mtp_q8_cache_slot_t *slot,
                                      const moe_weights_t *moe,
                                      int e, int load_type_gate, int load_type_up, int load_type_down) {
    // Gate: IQ2_XXS → Q8_0 (dequant to F32 → requant to Q8_0)
    if (moe->ffn_gate_exps_q) {
        int64_t src_bytes = gguf_raw_size(load_type_gate, D_MODEL * D_FF);
        const uint8_t *src_gate = moe->ffn_gate_exps_q + (int64_t)e * src_bytes;
        float *f32_buf = (float *)malloc((size_t)D_MODEL * D_FF * sizeof(float));
        gguf_dequantize(src_gate, load_type_gate, D_MODEL * D_FF, f32_buf);
        quantize_row_q8_K(f32_buf, (block_q8_K *)slot->gate_q8, D_MODEL * D_FF);
        free(f32_buf);
    }
    
    // Up: same
    if (moe->ffn_up_exps_q) {
        int64_t src_bytes = gguf_raw_size(load_type_up, D_MODEL * D_FF);
        const uint8_t *src_up = moe->ffn_up_exps_q + (int64_t)e * src_bytes;
        float *f32_buf = (float *)malloc((size_t)D_MODEL * D_FF * sizeof(float));
        gguf_dequantize(src_up, load_type_up, D_MODEL * D_FF, f32_buf);
        quantize_row_q8_K(f32_buf, (block_q8_K *)slot->up_q8, D_MODEL * D_FF);
        free(f32_buf);
    }
    
    // Down: [D_FF, D_MODEL]
    if (moe->ffn_down_exps_q) {
        int64_t src_bytes = gguf_raw_size(load_type_down, D_FF * D_MODEL);
        const uint8_t *src_down = moe->ffn_down_exps_q + (int64_t)e * src_bytes;
        float *f32_buf = (float *)malloc((size_t)D_FF * D_MODEL * sizeof(float));
        gguf_dequantize(src_down, load_type_down, D_FF * D_MODEL, f32_buf);
        quantize_row_q8_K(f32_buf, (block_q8_K *)slot->down_q8, D_FF * D_MODEL);
        free(f32_buf);
    }
}

#endif // MTP_Q8_CACHE_H

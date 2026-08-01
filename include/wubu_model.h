#ifndef WUBU_MODEL_H
#define WUBU_MODEL_H

#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_safetensors_shard.h"
#include "wubu_kvcache_quant.h"
#include "wubu_kv_select.h"
#include "wubu_kv_runtime.h"
#include "wubu_kvvq.h"  /* KB2: VQ codebook for KV compression */
#include "wubu_arena.h" /* C01: arena allocator for forward-buffer OOM safety */
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Layer configuration
typedef struct {
    int layer_idx;
    bool is_ssm;  // false = GQA
    
    // Weights (loaded from GGUF)
    ssm_layer_weights ssm;      // valid if is_ssm
    gqa_layer_weights gqa;      // valid if !is_ssm
    
    // Layer norm (pre-attention for all layers)
    float *attn_norm_weight;    // [D_MODEL], RMSNorm
    
    // Post-attention norm
    float *post_attn_norm_weight; // [D_MODEL], RMSNorm
    
    // MoE (FFN) weights
    moe_weights_t moe;
} wubu_layer_t;

// Complete model
#define GQA_MAX_CTX 8192  // default max cached positions (overridden by WUBU_MAX_CTX env)
// At 512K context, SWA (sliding window) handles long-range attention.
// KV cache is pre-allocated to this size; larger contexts use auto-eviction.
// Set WUBU_MAX_CTX=524288 to enable full 512K pre-allocation (needs ~64GB RAM).
/* GQA_KV_DIM is provided by wubu_dims.h (WUBU_DIMS.gqa_kv_dim) so it
 * resolves to the model's real kv dim at load time. */

// KV cache format: 0=F32, 1=F16 (halves memory at cost of conversion)
#ifndef KV_CACHE_F16
#define KV_CACHE_F16 1  // default to F16 for memory efficiency
#endif

// F16 <-> F32 conversion helpers (used by KV cache)
static inline float fp16_to_fp32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp  = (v >> 10) & 0x1F;
    int mant =  v        & 0x03FF;
    if (exp == 0) return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
}
static inline uint16_t fp32_to_fp16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    int sign = (bits >> 31) & 1;
    int exp  = (bits >> 23) & 0xFF;
    int mant = bits & 0x7FFFFF;
    uint16_t fp16;
    if (exp == 0) { fp16 = (sign << 15) | (0) | (mant >> 13); }
    else if (exp == 0xFF) { fp16 = (sign << 15) | (31 << 10) | (mant >> 13); }
    else {
        int newexp = exp - 127 + 15;
        if (newexp >= 31) fp16 = (sign << 15) | (31 << 10);
        else if (newexp <= 0) fp16 = (sign << 15);
        else fp16 = (sign << 15) | (newexp << 10) | (mant >> 13);
    }
    return fp16;
}

// KV cache access helpers
static inline float kv_cache_read_elem(const void *cache, int64_t idx) {
#if KV_CACHE_F16
    return fp16_to_fp32(((const uint16_t *)cache)[idx]);
#else
    return ((const float *)cache)[idx];
#endif
}
// KV cache quantization options
// KV_CACHE_F16: half-precision (default, 2 bytes/elem)
// KV_CACHE_F32: full precision (4 bytes/elem, fallback)
// KV_CACHE_Q4_0: 4-bit quantized (0.5 bytes/elem for payload, ~0.56 bytes with scale)

#ifndef KV_CACHE_Q4_0
#define KV_CACHE_Q4_0 1  // Q4_0 format for KV cache (4:1 compression vs F16)
#endif

// Q4_0 block: 32 elements, 4-bit each + fp16 scale
typedef struct {
    uint16_t d;    // scale factor (fp16)
    uint8_t qs[16];  // 32 × 4-bit nibbles
} block_q4_0_cache;

#define QK4_CACHE 32

// Our own Q8_0 KV-cache block (8-bit, block-32 absmax symmetric) -- routes to
// the tested wubu_kvcache_quant module. 4:1 vs F16, near-lossless (Roofline +
// llama.cpp + KIVI convergence: decode is BW-bound, halving KV bytes = faster).
typedef struct {
    int8_t qs[32];   // 32 int8 values
    float  d;        // absmax scale (fp32)
} block_q8_0_cache;
#define QK8_CACHE 32

// KB1: Adaptive KV block (doc 001, Ecco). Variable bit-width per block:
// 2-bit, 4-bit, or 8-bit stored as width_bits + scale + packed bytes.
// Layout: 16 bytes (width_bits + scale + 14 bytes packed) = same as Q4_0 block.
typedef struct {
    uint8_t width_bits;   // 2, 4, or 8
    uint8_t _pad[3];      // alignment padding
    float   scale;        // absmax scale
    uint8_t qs[24];       // packed values (32 × 8-bit max, fewer for 2/4-bit)
} block_adaptive_cache;
#define ADAPTIVE_CACHE 32

// KIVI per-token V block: head_dim used at alloc time to size the per-token
// fp32 scales. Must match the model's attention head_dim (Qwen-class = 128).
#ifndef KV_KIVI_HEADDIM
#define KV_KIVI_HEADDIM 128
#endif

// KIVI layout note: one fp32 scale per token's head_dim int8 values.
// (KIVI paper: V per-token, K per-channel. We store K as Q8_0-block which is
// per-block absmax -- near the per-channel intent at block granularity, and is
// computed streamingly; V per-token is exact KIVI since each write is 1 token.)
// Storage per token = head_dim int8 + 1 fp32 scale.
// Quantize 32 floats to Q4_0 block (symmetric, signed)
static inline void quantize_q4_0_cache_block(const float *x, block_q4_0_cache *b) {
    float amax = 0.0f;
    for (int i = 0; i < QK4_CACHE; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0f) {
        b->d = 0;
        memset(b->qs, 0, sizeof(b->qs));
        return;
    }
    const float d = amax / 7.0f;  // symmetric signed: [-7, 7] → [1, 15]
    const float id = 1.0f / d;
    b->d = fp32_to_fp16(d);
    for (int i = 0; i < QK4_CACHE; i++) {
        int q = (int)(x[i] * id + 8.0f);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        b->qs[i / 2] |= (uint8_t)(q << (4 * (i % 2)));
    }
}

// Dequantize one Q4_0 block
static inline void dequantize_q4_0_cache_block(const block_q4_0_cache *b, float *x) {
    const float d = fp16_to_fp32(b->d);
    for (int i = 0; i < QK4_CACHE; i++) {
        int q = (b->qs[i / 2] >> (4 * (i % 2))) & 0xF;
        x[i] = ((float)q - 8.0f) * d;
    }
}

// KV cache read: one head (n floats) from Q4_0 cache
// KV cache read: one head (n floats). DISPATCHES on the runtime g_kv_scheme
// (set at model load by the Roofline auto-selector) instead of a compile-time
// #if, so the engine picks precision per-model. Per-scheme bodies below.
extern int g_kv_scheme;            /* defined in wubu_kv_runtime.c */
extern void wubu_kv_set_scheme(int);
static inline void kv_cache_read_head_q4(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_q8(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_kivi(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_adaptive(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_f16(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_f32(const void *cache, int64_t offset, float *buf, int n);

static inline void kv_cache_read_head(const void *cache, int64_t offset,
                                       float *buf, int n) {
    switch (g_kv_scheme) {
        case WUBU_KV_Q4_0: kv_cache_read_head_q4(cache, offset, buf, n); break;
        case WUBU_KV_Q8:   kv_cache_read_head_q8(cache, offset, buf, n); break;
        case WUBU_KV_KIVI: kv_cache_read_head_kivi(cache, offset, buf, n); break;
        case WUBU_KV_ADAPTIVE: kv_cache_read_head_adaptive(cache, offset, buf, n); break;
        case WUBU_KV_F16:  kv_cache_read_head_f16(cache, offset, buf, n); break;
        case WUBU_KV_4KV:  kv_cache_read_head_f32(cache, offset, buf, n); break; /* 4KV uses F32+quant at layer level */
        case WUBU_KV_3BIT: kv_cache_read_head_f32(cache, offset, buf, n); break; /* 3BIT uses F32+quant at layer level */
        default:           kv_cache_read_head_f32(cache, offset, buf, n); break;
    }
}

static inline void kv_cache_read_head_q4(const void *cache, int64_t offset,
                                         float *buf, int n) {
    // Q4_0: offset is in float indices, convert to block index
    const int block_n = QK4_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_q4_0_cache *blocks = (const block_q4_0_cache *)cache;
    int done = 0;
    while (done < n) {
        float tmp[QK4_CACHE];
        dequantize_q4_0_cache_block(&blocks[start_block + (start_elem + done) / block_n], tmp);
        int blk_off = (start_elem + done) % block_n;
        int to_copy = n - done;
        if (to_copy > block_n - blk_off) to_copy = block_n - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
}

static inline void kv_cache_read_head_q8(const void *cache, int64_t offset,
                                         float *buf, int n) {
    // Our Q8_0 block-32 (near-lossless, routes to wubu_kvcache_quant).
    const int block_n = QK8_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_q8_0_cache *blocks = (const block_q8_0_cache *)cache;
    int done = 0;
    while (done < n) {
        float tmp[QK8_CACHE];
        wubu_kvq_q8_dequant(blocks[start_block + (start_elem + done) / block_n].qs,
                             blocks[start_block + (start_elem + done) / block_n].d,
                             tmp, block_n);
        int blk_off = (start_elem + done) % block_n;
        int to_copy = n - done;
        if (to_copy > block_n - blk_off) to_copy = block_n - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
}

static inline void kv_cache_read_head_kivi(const void *cache, int64_t offset,
                                           float *buf, int n) {
    int hd = g_kv_head_dim > 0 ? g_kv_head_dim : KV_KIVI_HEADDIM;
    if (n == hd) {
        // Fast path: single token
        int t0 = (int)(offset / hd), p0 = (int)(offset % hd);
        const uint8_t *base = (const uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
            const uint8_t *q = base;
            float scale = *(const float *)(base + hd);
            float tmp[512];
            int work_hd = hd > 512 ? 512 : hd;
            wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
            for (int i = 0; i < n; i++) buf[i] = tmp[p0 + i];
        } else {
        // Batch path: read multiple tokens
        int tokens = n / hd;
        for (int t = 0; t < tokens; t++) {
            int token_offset = offset + t * hd;
            int t0 = (int)(token_offset / hd), p0 = (int)(token_offset % hd);
            const uint8_t *base = (const uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
            const uint8_t *q = base;
            float scale = *(const float *)(base + hd);
            float tmp[512];
            int work_hd = hd > 512 ? 512 : hd;
            wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
            for (int i = 0; i < hd; i++) buf[t * hd + i] = tmp[p0 + i];
        }
    }
}

static inline void kv_cache_read_head_f16(const void *cache, int64_t offset,
                                          float *buf, int n) {
    const uint16_t *src = (const uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
}

/* KB1: Adaptive KV read (doc 001). Uses wubu_kvq_adaptive_dequant
 * for per-block variable bit-width dequantization. */
static inline void kv_cache_read_head_adaptive(const void *cache, int64_t offset,
                                                  float *buf, int n) {
    const int block_n = ADAPTIVE_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_adaptive_cache *blocks = (const block_adaptive_cache *)cache;
    int done = 0;
    while (done < n) {
        float tmp[ADAPTIVE_CACHE];
        const block_adaptive_cache *blk = &blocks[start_block + (start_elem + done) / block_n];
        /* Dequantize: unpack based on width_bits */
        int bits = blk->width_bits;
        float scale = blk->scale;
        int vals_per_byte = (bits == 2) ? 4 : (bits == 4) ? 2 : 1;
        for (int i = 0; i < block_n; i++) {
            int byte_idx = i / vals_per_byte;
            int pos = i % vals_per_byte;
            uint8_t raw;
            if (bits == 2) {
                raw = (blk->qs[byte_idx] >> (2 * pos)) & 0x3;
            } else if (bits == 4) {
                raw = (blk->qs[byte_idx] >> (4 * pos)) & 0xF;
            } else {
                raw = blk->qs[byte_idx];
            }
            /* Convert to signed: unsigned [0, qmax] → signed [-half, half-1] */
            int half = (1 << (bits - 1));
            int sq = (int)raw - half;
            tmp[i] = (float)sq * scale;
        }
        int blk_off = (start_elem + done) % block_n;
        int to_copy = n - done;
        if (to_copy > block_n - blk_off) to_copy = block_n - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
}

static inline void kv_cache_read_head_f32(const void *cache, int64_t offset,
                                          float *buf, int n) {
    memcpy(buf, (const float *)cache + offset, n * sizeof(float));
}

// Batch write one head to KV cache (runtime dispatch).
static inline void kv_cache_write_head_q4(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_q8(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_kivi(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_adaptive(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_f16(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_f32(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head(void *cache, int64_t offset,
                                        const float *buf, int n) {
    switch (g_kv_scheme) {
        case WUBU_KV_Q4_0: kv_cache_write_head_q4(cache, offset, buf, n); break;
        case WUBU_KV_Q8:   kv_cache_write_head_q8(cache, offset, buf, n); break;
        case WUBU_KV_KIVI: kv_cache_write_head_kivi(cache, offset, buf, n); break;
        case WUBU_KV_ADAPTIVE: kv_cache_write_head_adaptive(cache, offset, buf, n); break;
        case WUBU_KV_F16:  kv_cache_write_head_f16(cache, offset, buf, n); break;
        case WUBU_KV_4KV:  kv_cache_write_head_f32(cache, offset, buf, n); break;
        case WUBU_KV_3BIT: kv_cache_write_head_f32(cache, offset, buf, n); break;
        default:           kv_cache_write_head_f32(cache, offset, buf, n); break;
    }
}

static inline void kv_cache_write_head_q4(void *cache, int64_t offset,
                                           const float *buf, int n) {
    const int block_n = QK4_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    int end_elem = start_elem + n;
    block_q4_0_cache *blocks = (block_q4_0_cache *)cache;
    if (start_elem == 0) {
        int n_aligned = n - (end_elem % block_n);
        if (n_aligned < 0) n_aligned = 0;
        for (int bi = 0; bi < n_aligned / block_n; bi++) {
            quantize_q4_0_cache_block(buf + bi * block_n, &blocks[start_block + bi]);
        }
        int rem = n - n_aligned;
        if (rem > 0) {
            int bi = n_aligned / block_n;
            float tmp[QK4_CACHE];
            dequantize_q4_0_cache_block(&blocks[start_block + bi], tmp);
            for (int i = 0; i < rem; i++) tmp[i] = buf[n_aligned + i];
            quantize_q4_0_cache_block(tmp, &blocks[start_block + bi]);
        }
    } else {
        int first_rem = block_n - start_elem;
        if (first_rem > n) first_rem = n;
        {
            float tmp[QK4_CACHE];
            dequantize_q4_0_cache_block(&blocks[start_block], tmp);
            for (int i = 0; i < first_rem; i++) tmp[start_elem + i] = buf[i];
            quantize_q4_0_cache_block(tmp, &blocks[start_block]);
        }
        int remaining = n - first_rem;
        if (remaining > 0) {
            kv_cache_write_head_q4(cache, offset + first_rem, buf + first_rem, remaining);
        }
    }
}

static inline void kv_cache_write_head_q8(void *cache, int64_t offset,
                                          const float *buf, int n) {
    const int block_n = QK8_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    block_q8_0_cache *blocks = (block_q8_0_cache *)cache;
    int done = 0;
    while (done < n) {
        int bn = block_n - ((start_elem + done) % block_n);
        if (bn > n - done) bn = n - done;
        int blk = start_block + (start_elem + done) / block_n;
        if (bn == block_n) {
            wubu_kvq_q8_quant(buf + done, blocks[blk].qs, &blocks[blk].d, block_n);
        } else {
            float tmp[QK8_CACHE];
            wubu_kvq_q8_dequant(blocks[blk].qs, blocks[blk].d, tmp, block_n);
            int blk_off = (start_elem + done) % block_n;
            for (int i = 0; i < bn; i++) tmp[blk_off + i] = buf[done + i];
            wubu_kvq_q8_quant(tmp, blocks[blk].qs, &blocks[blk].d, block_n);
        }
        done += bn;
    }
}

static inline void kv_cache_write_head_kivi(void *cache, int64_t offset,
                                            const float *buf, int n) {
    int hd = g_kv_head_dim > 0 ? g_kv_head_dim : KV_KIVI_HEADDIM;
    if (n == hd) {
        // Fast path: single token
        int t0 = (int)(offset / hd), p0 = (int)(offset % hd);
        uint8_t *base = (uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
        uint8_t *q = base;
        float scale = *(const float *)(base + hd);
        float tmp[512];
        int work_hd = hd > 512 ? 512 : hd;
        wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
        for (int i = 0; i < n; i++) tmp[p0 + i] = buf[i];
        wubu_kvq_kivi_quant_V(tmp, q, &scale, 1, work_hd);
        *(float *)(base + hd) = scale;
    } else {
        // Batch path: write multiple tokens
        int tokens = n / hd;
        for (int t = 0; t < tokens; t++) {
            int token_offset = offset + t * hd;
            int t0 = (int)(token_offset / hd), p0 = (int)(token_offset % hd);
            uint8_t *base = (uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
            uint8_t *q = base;
            float scale = *(const float *)(base + hd);
            float tmp[512];
            int work_hd = hd > 512 ? 512 : hd;
            wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
            for (int i = 0; i < hd; i++) tmp[p0 + i] = buf[t * hd + i];
            wubu_kvq_kivi_quant_V(tmp, q, &scale, 1, work_hd);
            *(float *)(base + hd) = scale;
        }
    }
}

static inline void kv_cache_write_head_f16(void *cache, int64_t offset,
                                           const float *buf, int n) {
    uint16_t *dst = (uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) dst[i] = fp32_to_fp16(buf[i]);
}

static inline void kv_cache_write_head_f32(void *cache, int64_t offset,
                                           const float *buf, int n) {
    memcpy((float *)cache + offset, buf, n * sizeof(float));
}

/* KB1: Adaptive KV write (doc 001). Uses Ecco entropy-aware bit-width
 * selection: low-variance blocks → 2-bit, high-variance → 8-bit. */
static inline void kv_cache_write_head_adaptive(void *cache, int64_t offset,
                                                 const float *buf, int n) {
    const int block_n = ADAPTIVE_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    block_adaptive_cache *blocks = (block_adaptive_cache *)cache;
    int done = 0;
    while (done < n) {
        int bn = block_n - ((start_elem + done) % block_n);
        if (bn > n - done) bn = n - done;
        int blk_idx = start_block + (start_elem + done) / block_n;
        int blk_off = (start_elem + done) % block_n;

        if (bn == block_n) {
            /* Full block: quantize directly */
            float tmp[block_n];
            for (int i = 0; i < block_n; i++) tmp[i] = buf[done + i];
            /* Compute absmax and variance for bit-width selection */
            float amax = 0.0f, mean = 0.0f;
            for (int i = 0; i < block_n; i++) {
                float ax = fabsf(tmp[i]);
                if (ax > amax) amax = ax;
                mean += tmp[i];
            }
            mean /= (float)block_n;
            float var = 0.0f;
            for (int i = 0; i < block_n; i++) {
                float d = tmp[i] - mean;
                var += d * d;
            }
            var /= (float)block_n;
            /* Select bit-width: low variance → fewer bits */
            int bits = (var < 0.01f) ? 2 : (var < 0.1f) ? 4 : 8;
            int half = (1 << (bits - 1));
            /* Symmetric signed: [-amax, +amax] → [-half, half-1] */
            /* scale maps amax to half-1 so the full range is representable */
            float scale = (amax > 1e-8f) ? amax / (float)(half - 1) : 0.0f;
            if (scale == 0.0f) scale = 1e-8f;
            blocks[blk_idx].width_bits = (uint8_t)bits;
            blocks[blk_idx].scale = scale;
            /* Pack values — symmetric signed quantization */
            int vals_per_byte = (bits == 2) ? 4 : (bits == 4) ? 2 : 1;
            memset(blocks[blk_idx].qs, 0, sizeof(blocks[blk_idx].qs));
            for (int i = 0; i < block_n; i++) {
                int q = (int)roundf(tmp[i] / scale);
                /* Clamp to signed range [-half, half-1] */
                if (q >= half) q = half - 1;
                if (q < -half) q = -half;
                /* Convert to unsigned: [-half, half-1] → [0, qmax] */
                q += half;
                int byte_idx = i / vals_per_byte;
                int pos = i % vals_per_byte;
                if (bits == 2) {
                    blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (2 * pos));
                } else if (bits == 4) {
                    blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (4 * pos));
                } else {
                    blocks[blk_idx].qs[byte_idx] = (uint8_t)q;
                }
            }
        } else {
            /* Partial block: read existing, merge, re-quantize */
            float tmp[block_n];
            /* Dequantize existing block */
            const block_adaptive_cache *blk = &blocks[blk_idx];
            int bits = blk->width_bits;
            float scale = blk->scale;
            int vals_per_byte = (bits == 2) ? 4 : (bits == 4) ? 2 : 1;
            int qmax = (1 << bits) - 1;
            int half = (1 << (bits - 1));
            for (int i = 0; i < block_n; i++) {
                int byte_idx = i / vals_per_byte;
                int pos = i % vals_per_byte;
                uint8_t raw;
                if (bits == 2) raw = (blk->qs[byte_idx] >> (2 * pos)) & 0x3;
                else if (bits == 4) raw = (blk->qs[byte_idx] >> (4 * pos)) & 0xF;
                else raw = blk->qs[byte_idx];
                int sq = (int)raw;
                if (sq >= half) sq -= (qmax + 1);
                tmp[i] = (float)sq * scale;
            }
            /* Overwrite the new elements */
            for (int i = 0; i < bn; i++) tmp[blk_off + i] = buf[done + i];
            /* Re-quantize the whole block */
            float amax = 0.0f, mean = 0.0f;
            for (int i = 0; i < block_n; i++) {
                float ax = fabsf(tmp[i]);
                if (ax > amax) amax = ax;
                mean += tmp[i];
            }
            mean /= (float)block_n;
            float var = 0.0f;
            for (int i = 0; i < block_n; i++) { float d = tmp[i] - mean; var += d * d; }
            var /= (float)block_n;
            int new_bits = (var < 0.01f) ? 2 : (var < 0.1f) ? 4 : 8;
            int nhalf = (1 << (new_bits - 1));
            /* Symmetric signed: [-amax, +amax] → [-nhalf, nhalf-1] */
            float new_scale = (amax > 1e-8f) ? amax / (float)(nhalf - 1) : 1e-8f;
            blocks[blk_idx].width_bits = (uint8_t)new_bits;
            blocks[blk_idx].scale = new_scale;
            int nvpb = (new_bits == 2) ? 4 : (new_bits == 4) ? 2 : 1;
            memset(blocks[blk_idx].qs, 0, sizeof(blocks[blk_idx].qs));
            for (int i = 0; i < block_n; i++) {
                int q = (int)roundf(tmp[i] / new_scale);
                if (q >= nhalf) q = nhalf - 1;
                if (q < -nhalf) q = -nhalf;
                q += nhalf;
                int byte_idx = i / nvpb;
                int pos = i % nvpb;
                if (new_bits == 2) blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (2 * pos));
                else if (new_bits == 4) blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (4 * pos));
                else blocks[blk_idx].qs[byte_idx] = (uint8_t)q;
            }
        }
        done += bn;
    }
}

// KV cache allocation: returns number of bytes needed for n_elems (runtime dispatch).
// Uses model's actual head_dim via g_kv_head_dim (set by wubu_kv_autoselect).
static inline int64_t kv_cache_alloc_size(int64_t n_elems) {
    switch (g_kv_scheme) {
        case WUBU_KV_Q4_0: {
            int64_t n_blocks = (n_elems + QK4_CACHE - 1) / QK4_CACHE;
            return n_blocks * (int64_t)sizeof(block_q4_0_cache);
        }
        case WUBU_KV_Q8: {
            int64_t n_blocks = (n_elems + QK8_CACHE - 1) / QK8_CACHE;
            return n_blocks * (int64_t)sizeof(block_q8_0_cache);
        }
        case WUBU_KV_KIVI: {
            int64_t hd = g_kv_head_dim > 0 ? g_kv_head_dim : KV_KIVI_HEADDIM;
            int64_t tokens = (n_elems + hd - 1) / hd;
            return n_elems * (int64_t)sizeof(int8_t) + tokens * (int64_t)sizeof(float);
        }
        case WUBU_KV_ADAPTIVE: {
            int64_t n_blocks = (n_elems + ADAPTIVE_CACHE - 1) / ADAPTIVE_CACHE;
            return n_blocks * (int64_t)sizeof(block_adaptive_cache);
        }
        case WUBU_KV_F16:
            return n_elems * (int64_t)sizeof(uint16_t);
        case WUBU_KV_4KV:
            /* SAW-INT4: 0.5 bytes/elem (4-bit) + per-block-16 scales */
            return (n_elems / 2) + (n_elems / 16) * (int64_t)sizeof(float) + 8;
        case WUBU_KV_3BIT:
            /* TurboQuant INT3: ~0.375 bytes/elem (3-bit) + per-token scales */
            return (n_elems * 3 + 7) / 8 + (n_elems / 64) * (int64_t)sizeof(float) + 8;
        default:
            return n_elems * (int64_t)sizeof(float);
    }
}

// MTP (Multi-Token Prediction) head for speculative decode
// Architecture: h_39 → hnorm → concat(hnorm, enorm(embd)) → eh_proj → blk.40 → shared_head_norm → output
typedef struct {
    bool loaded;
    
    // Nextn norms (F32, all [D_MODEL])
    float *nextn_hnorm;             // [2048] — hidden state norm
    float *nextn_enorm;             // [2048] — token embedding norm
    float *nextn_shared_head_norm;  // [2048] — output norm
    
    // eh_proj weight (F32 dequantized): concat([h_norm | e_norm], dim=4096) → [2048]
    float *nextn_eh_proj_f32;   // [4096, 2048] F32
    int64_t nextn_eh_proj_dim;         // 4096 (concat dim)
    
    // Blk.40 (a full GQA+MoE layer)
    wubu_layer_t blk40;
    
    // KV cache for blk.40's GQA attention
    float *k_cache;  // [GQA_MAX_CTX * kv_dim]
    float *v_cache;  // [GQA_MAX_CTX * kv_dim]
    int kv_dim;      // per-layer KV dim (kv_heads * head_dim)
    int cache_len;
} mtp_head_t;
typedef struct {
    int n_layers;
    wubu_layer_t *layers;
    
    // Token embedding
    float *token_embd;       // [vocab_size, D_MODEL] or NULL if using embedding file
    float *output_weight;    // [D_MODEL, vocab_size] or NULL
    const uint8_t *output_weight_q;   // raw Q4_K quantized
    int output_weight_type;
    bool tied_output;                // true if output_weight_q points to token_embd
    const uint8_t *token_embd_q;     // raw Q4_K quantized token embeddings (large vocab, mmap'd)
    int token_embd_type;             // GGML type of token_embd (usually Q4_K)
    int rotate_P;                    // doc 013: Hadamard prefix fused into output_weight (>1 if WUBU_ROTATE_W)
    // Lazy, zero-copy embedding / lm_head (safetensors BF16 path). When set,
    // token_embd / output_weight are NOT copied to F32; the row is dequantized
    // on demand from the mmap'd shard. Saves ~10 GB for 27B-class models.
    const uint8_t *lazy_embd_raw;     // raw bytes of embed_tokens (row-major)
    int            lazy_embd_dtype;   // ST_DTYPE_*
    int64_t        lazy_embd_row;     // elems per embedding row (= D_MODEL)
    const uint8_t *lazy_lmhead_raw;   // raw bytes of lm_head.weight [vocab, D]
    int            lazy_lmhead_dtype;
    int64_t        lazy_lmhead_row;   // elems per lm_head row (= D_MODEL)

    /* Persistent shard context for lazy embed/lm_head mmap access.
     * Must stay alive for the lifetime of the model. */
    struct wubu_shard_ctx *shard_ctx;

    // Embedding file (from Phase 1)
    bool use_embedding_file;
    int vocab_size;
    
    // Norms
    float *norm_weight;  // final RMSNorm [D_MODEL]
    
    // State buffers (reused across calls)
    float *ssm_states;    // [max_layers, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    float *conv_states;   // [max_layers, B, CONV_KERNEL-1, CONV_DIM]
    size_t ssm_state_total;  // bytes allocated for ssm_states (incl. conv_states)    
    // GQA KV cache (10 GQA layers, max 256k context)
    void *gqa_k_cache;  // [n_gqa_layers * runtime_max_ctx * kv_dim] F32 or F16
    void *gqa_v_cache;  // [n_gqa_layers * runtime_max_ctx * kv_dim]
    int gqa_cache_len;   // how many tokens cached per layer (all layers same len)
    int gqa_max_ctx;     // runtime max ctx (GQA_MAX_CTX or WUBU_MAX_CTX env)

    // GGUF context (for per-layer MoE lazy loading)
    // Model state save/restore (for speculative decode rollback)
    float *ssm_states_saved;    // same size as ssm_states
    float *conv_states_saved;   // same size as conv_states
    int gqa_cache_len_saved;
    int mtp_cache_len_saved;

    // GGUF context (for per-layer MoE lazy loading)
    gguf_ctx *gguf_ctx;
    size_t    data_blob_size;  // size of GGUF data blob (for budget calc)
    
    // Enable MoE during forward (default: false for memory reasons)
    bool enable_moe;
    
    // Skip output projection in forward (for GPU offload)
    bool skip_output_proj;
    
    // MoE test: only load MoE for first N layers (0 = all)
    int moe_max_layers;

    // ds4-ssd slot-bank: when non-NULL, routed MoE experts are paged from a
    // sidecar on disk (LRU) instead of held resident. Set by the bridge when
    // a sidecar directory is supplied. Forward uses wubu_moe_forward_ssd.
    wubu_ssd_moe_t *ssd_moe;
    
    // MTP (Multi-Token Prediction) head for speculative decode
    mtp_head_t mtp;
    
    // Last hidden state capture (for MTP: set to a [D_MODEL] buffer before forward)
    float *save_last_hidden;

    // GPU acceleration context (opaque pointer, managed by wubu_model_gpu.cu)
    void *gpu_ctx;

    // OOM-safe forward arena: allocated once at model init, reset per forward.
    // All temporary buffers (x, normed, attn_out, normed2, ffn_out, prev_experts)
    // come from here — no per-token malloc churn. Sized to B*T*d_model*4.
    wubu_arena_t fwd_arena;
    wubu_sub_arena_t fwd_sub;

    // Number of GQA layers (for KV cache sizing)
    int n_gqa_layers;

    // Dynamic model dimensions (extracted from GGUF, model-adapter aware)
    int d_model;          // hidden dimension (2048 for Qwen, 2816 for DiffusionGemma)
    int d_ff;             // expert intermediate dim
    int n_experts;        // total experts
    int n_active_experts; // top-k
    int shared_expert_ff; // shared expert intermediate dim (MoE); 0 if none
    int tensor_naming;    // 0=blk.Qwen 1=model.layers.Gemma 2=pure-GQA

    // Additional dynamic dimensions for GPU and other code
    int d_inner;          // SSM inner dimension / VALUE_DIM
    int key_dim;          // KEY_DIM = SSM_D_STATE * SSM_K_HEADS
    int conv_dim;         // CONV_DIM
    int conv_kernel;      // CONV_KERNEL (conv1d kernel size, default 4)
    int dt_rank;          // DT_RANK
    int ssm_k_heads;      // SSM_K_HEADS
    int ssm_v_heads;      // SSM_V_HEADS
    int ssm_d_state;      // SSM_D_STATE
    int gqa_q_heads;      // GQA_Q_HEADS
    int gqa_kv_heads;     // GQA_KV_HEADS
    int gqa_head_dim;     // GQA_HEAD_DIM
    int rotary_dim;       // ROTARY_DIM (RoPE rotation dim)
} wubu_model_t;

// Create model, load from GGUF
bool wubu_model_init(wubu_model_t *model, const char *gguf_path);

// Free model resources
void wubu_model_free(wubu_model_t *model);

// Forward pass through all layers
// Input: token_ids [B, T], Output: logits [B, T, vocab_size]
void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits);

// Forward pass from embeddings (bypass token lookup)
// Input: embeddings [B, T, D_MODEL], Output: logits [B, T, vocab_size]
void wubu_model_forward_from_embd(wubu_model_t *model,
                                   const float *embeddings, int B, int T,
                                   float *logits);

// Reset persistent SSM/conv recurrence state to zero (start a fresh sequence).
// Required before comparing two independent generations (e.g. plain vs
// speculative decode) so both start from the same zero state.
void wubu_model_reset_state(wubu_model_t *model);

// Chunked forward: process a long [B, T_total] sequence in time-chunks of
// <= chunk_sz tokens, carrying the model's persistent SSM/conv/KV-cache state
// across chunks. Mathematically identical to a single forward, but bounds peak
// memory: each chunk allocates intermediates for chunk_sz tokens only. The
// final chunk's logits (positions [T_total-chunk_sz, T_total)) are written to
// `logits` (sized B*chunk_sz*vocab_size). Use this to run the full 256K context
// on a memory-limited box.
void wubu_model_forward_chunked(wubu_model_t *model,
                                const int *token_ids, int B, int T_total,
                                int chunk_sz, float *logits);

// ================================================================
// GPU-Accelerated Forward Path (wubu_model_gpu.cu)
// ================================================================

// Initialize GPU context: upload GQA weights, allocate KV cache + scratch.
// max_ctx: maximum KV cache positions (e.g. 262144).
// chunk_sz: max tokens per GPU batch (e.g. 512).
// Returns 1 on success, 0 on failure.
// When GPU context is active, wubu_model_forward() automatically uses
// GPU for GQA attention layers.
int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz);

// Run one GQA layer on GPU.
// Internal: called by wubu_model_forward when gpu_ctx != NULL.
int wubu_model_gpu_gqa_forward(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C, float *h_attn);

// Get GPU chunk size (max tokens per batched GPU call).
// Returns 0 if GPU not initialized.
int wubu_model_gpu_chunk_sz(wubu_model_t *model);

// Run SSM projections (qkv, gate) on GPU via quantized matmul kernels.
// h_norm: [C, D_MODEL] input
// C: number of tokens (1 for decode)
// qkv_out: [C, CONV_DIM] output (host)
// z_out: [C, VALUE_DIM] output (host)
// ssm_out_out: unused (future: ssm output projection)
int wubu_model_gpu_ssm_project(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C,
                                float *qkv_out, float *z_out,
                                float *ssm_out_out);

// Run GPU SSM completely on GPU: quantized matmuls → conv1d → SiLU → split
// → L2 norm → recurrence → gated norm → ssm_out projection.
// Returns 1 on success, 0 on fallback to CPU.
int wubu_model_gpu_ssm_forward_full(wubu_model_t *model, int layer_idx,
                                     const float *h_norm, int C,
                                     float *h_attn_out);

// Set SSM layer GPU pointers from gpu_ctx for hybrid (CPU SSM + GPU recurrence).
// Called by wubu_model_forward fallback paths when gpu_ctx exists.
// gpu_ctx is model->gpu_ctx (void*), ssm is layer->ssm to fill.
void wubu_gpu_set_ssm_hybrid(void *gpu_ctx, int layer_idx, ssm_layer_weights *ssm);

// Sync CPU SSM state + conv state to GPU before forward_full decode.
// Call after hybrid prefill path updates CPU state, so subsequent
// forward_full decode uses the correct accumulated state.
void wubu_gpu_sync_ssm_state_to_gpu(void *gpu_ctx, int layer_idx,
                                     const float *cpu_ssm_state,
                                     const float *cpu_conv_state);

// Sync GPU SSM state + conv state back to CPU after forward_full decode.
// Ensures CPU state tracks GPU state for next hybrid prefill.
void wubu_gpu_sync_ssm_state_to_cpu(void *gpu_ctx, int layer_idx,
                                     float *cpu_ssm_state,
                                     float *cpu_conv_state);

// Run MoE experts via GPU kernel, replacing CPU quantized matmul loop.
// Shared expert and router remain on CPU.
// Called per-token from wubu_moe_forward's expert loop.
void wubu_model_gpu_moe_experts(const moe_weights_t *w,
    const float *x_s,
    const int *indices_s, const float *weights_s,
    float *expert_contribs,
    void *model_ptr);

// Free all GPU resources and reset gpu_ctx to NULL.
void wubu_model_gpu_free(wubu_model_t *model);

// Model-level backward pass
// Requires saved layer outputs from forward (normed, attn_out, normed2, ffn_out arrays)
// All arrays are [n_layers * B * T * D_MODEL] flattened
// SSM/GQA intermediates arrays (per layer) — see wubu_ssm_backward / wubu_gqa_backward
// For MoE: gradient passes through (identity backward)
// Backward pass from embeddings
void wubu_model_backward_from_embd(
    const wubu_model_t *model,
    const float *embeddings,
    const float *logits, const float *d_logits,
    const float *saved_normed,     // [n_layers * N * D_MODEL]
    const float *saved_attn_out,   // [n_layers * N * D_MODEL]
    const float *saved_normed2,    // [n_layers * N * D_MODEL]
    const float *saved_ffn_out,    // [n_layers * N * D_MODEL]
    float *d_embeddings,
    int B, int T);

// MTP: Load MTP head from a separate GGUF model file
// Must be called AFTER wubu_model_init on the main model
// Pass the MTP GGUF model path
bool wubu_mtp_load(mtp_head_t *mtp, const char *mtp_gguf_path,
                   gguf_ctx *main_ctx, const uint8_t *main_blob,
                   int gqa_max_ctx);

// MTP: Draft forward — predict next tokens from last hidden state
// x: [D_MODEL] — last hidden state from main model (layer 39 output, post-residual)
// token_embd: [B, D_MODEL] — embeddings of candidate continuation tokens
// B: number of draft candidates to evaluate
// logits_out: [B, vocab_size] — output logits for each candidate
// Returns: number of tokens consumed from token_embd (for KV cache tracking)
int wubu_mtp_draft_forward(wubu_model_t *model,
                           const float *x,
                           const float *token_embd, int B,
                           float *logits_out);

// Free MTP head resources
void wubu_mtp_free(mtp_head_t *mtp);

// Model state save/restore for speculative decode rollback
bool wubu_model_checkpoint(wubu_model_t *model);
void wubu_model_rollback(wubu_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MODEL_H

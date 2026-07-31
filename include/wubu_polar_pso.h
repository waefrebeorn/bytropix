/*
 * wubu_polar_pso.h — Meta-compiled PolarQuant decode kernels
 *
 * Three game-hardware-inspired optimizations:
 *
 * 1. PSO CACHING (like compiled shaders):
 *    Pre-compile decode kernels for each (bits, head_dim) config.
 *    At init, select the right function pointer. Decode is then
 *    a single indirect call — no branching on config at runtime.
 *
 * 2. PROCEDURAL SPOOLING (inflated precache as formula):
 *    KV tokens are stored as polar PARAMETERS (radius + angles),
 *    not as full vectors. The "precache" is the formula. At decode
 *    time, we "inflate" the parameters into the full vector using
 *    the recursive polar reconstruction — same as procedural
 *    generation inflates a seed into a world.
 *
 * 3. N64 RAMBUS SERIAL BUS:
 *    The packed bitstream is read serially, bit-by-bit, through
 *    a narrow "channel" (a single uint64_t register that we
 *    refill from the packed byte stream). This avoids random
 *    memory access and keeps the decode in L1 cache. The Rambus
 *    pattern: narrow bus, high clock, serial bursts.
 *
 *    The bit reader loads 8 bytes at a time into a shift register
 *    and pops bits off the end. This is exactly how RDRAM worked:
 *    9-bit serial channel at 500MHz = 4.5 GB/s from 4MB.
 */

#ifndef WUBU_POLAR_PSO_H
#define WUBU_POLAR_PSO_H

#include "wubu_polarquant.h"
#include <stdint.h>
#include <stddef.h>

/* ==========================================================
 * Rambus-style serial bit reader
 *
 * Loads 8 bytes at a time into a 64-bit shift register.
 * Pops N bits off the MSB end. Refills when empty.
 * This keeps the bitstream access pattern sequential
 * and cache-friendly (pure L1 hits).
 * ========================================================== */

typedef struct {
    const uint8_t *data;     /* pointer into packed bitstream */
    const uint8_t *end;      /* end of bitstream */
    uint64_t reg;            /* 64-bit shift register */
    int bits_in_reg;         /* valid bits currently in register */
} wubu_bit_reader_t;

static inline void wubu_bit_reader_init(wubu_bit_reader_t *r,
    const uint8_t *data, int nbytes) {
    r->data = data;
    r->end = data + nbytes;
    r->reg = 0;
    r->bits_in_reg = 0;
}

/* Refill the shift register from the bitstream (max 64 bits) */
static inline void wubu_bit_reader_refill(wubu_bit_reader_t *r) {
    while (r->bits_in_reg <= 56 && r->data < r->end) {
        r->reg |= ((uint64_t)(*r->data++)) << r->bits_in_reg;
        r->bits_in_reg += 8;
    }
}

/* Pop N bits (1-16) from the register, MSB-first → LSB-first */
static inline int wubu_bit_reader_pop(wubu_bit_reader_t *r, int nbits) {
    if (r->bits_in_reg < nbits) {
        wubu_bit_reader_refill(r);
    }
    int val = (int)(r->reg & ((1ULL << nbits) - 1));
    r->reg >>= nbits;
    r->bits_in_reg -= nbits;
    return val;
}

/* ==========================================================
 * PSO: Pre-compiled decode function table
 *
 * Each (bits, head_dim) combo gets a dedicated decode function.
 * At init, we select the right one and store it in a dispatch
 * table. Decode is then a single indirect call — no branching.
 *
 * This mirrors GPU Pipeline State Object caching:
 * - Compile once (at init)
 * - Hot-path is just `table[idx](args)`
 * - No runtime shader compilation stalls
 * ========================================================== */

typedef void (*wubu_polar_decode_fn)(
    const wubu_polarquant_t *pq,
    const uint8_t *packed, int nbytes,
    float *out, int d);

typedef struct {
    wubu_polar_decode_fn decode;
    int bits;
    int d;
    int storage_bytes;
    /* Rambus-style: precomputed angles table for fast cos/sin */
    /* For bits=8: 256 entries, bits=4: 16 entries, etc. */
    float *cos_table;  /* [1<<bits] precomputed cos values */
    float *sin_table;  /* [1<<bits] precomputed sin values */
} wubu_polar_pso_t;

/* Public PSO decode wrapper */
void wubu_pso_decode(const uint8_t *packed, int nbytes, float *out, int d);

/* Initialize PSO for a given (bits, d) config.
 * Pre-compiles the decode kernel and precomputes trig tables. */
int wubu_polar_pso_init(wubu_polar_pso_t *pso,
    wubu_polarquant_t *pq, int bits, int d);

/* Free PSO */
void wubu_polar_pso_free(wubu_polar_pso_t *pso);

/* Decode using PSO — single indirect call */
static inline void wubu_polar_pso_decode(
    const wubu_polar_pso_t *pso,
    const uint8_t *packed, int nbytes,
    float *out, int d) {
    pso->decode(NULL /*unused*/, packed, nbytes, out, d);
}

/* ==========================================================
 * Procedural precache: formula-encoded KV tokens
 *
 * Instead of storing dequantized vectors, we store the
 * polar PARAMETERS (radius + angle indices) as the "seed".
 * The full vector is "procedurally generated" from the seed
 * at decode time — exactly like procedural spooling in games.
 *
 * Memory cost: ~84 bytes per token at 5-bit (vs 512 F32).
 * Decode cost: O(d) recursive reconstruction.
 *
 * The precache array is a flat buffer of packed seeds,
 * indexed by token position. The "inflation" happens in
 * the decode kernel, which is PSO-compiled for the exact
 * bit-width.
 * ========================================================== */

typedef struct {
    wubu_polar_pso_t pso;       /* compiled decode kernel */
    wubu_polarquant_t *pq;      /* quantizer for encoding */
    int d;                      /* head_dim */
    int max_tokens;             /* capacity */
    int n_tokens;               /* current count */
    int bytes_per_token;        /* packed bytes per token */
    uint8_t *k_seed;            /* [max_tokens * bytes_per_token] K seeds */
    uint8_t *v_seed;            /* [max_tokens * bytes_per_token] V seeds */
    int *seed_bytes;            /* [max_tokens] actual bytes per K seed */
} wubu_polar_precache_t;

/* Initialize procedural precache */
int wubu_polar_precache_init(wubu_polar_precache_t *pc,
    wubu_polarquant_t *pq, int bits, int d, int max_tokens);

/* Free precache */
void wubu_polar_precache_free(wubu_polar_precache_t *pc);

/* Push a K,V pair as a procedural seed */
int wubu_polar_precache_push(wubu_polar_precache_t *pc,
    const float *k, const float *v);

/* Decode token i's K vector (procedural inflation) */
int wubu_polar_precache_decode_k(wubu_polar_precache_t *pc,
    int token_idx, float *k_out);

/* Decode token i's V vector */
int wubu_polar_precache_decode_v(wubu_polar_precache_t *pc,
    int token_idx, float *v_out);

/* Compute attention using PSO-decode + online softmax */
int wubu_polar_precache_attention(wubu_polar_precache_t *pc,
    const float *q, float *out, float temperature,
    int n_recent_f32, const float *recent_k, const float *recent_v);

/* ==========================================================
 * Rambus-style fused decode + dot via serial bit reader
 *
 * Reads the packed bitstream serially (8 bytes at a time
 * into a 64-bit shift register), decodes K inline, and
 * accumulates Q·K. No intermediate buffer for K.
 * ========================================================== */

/* Fused: decode K from seed + dot with Q.
 * Uses wubu_bit_reader_t for serial bitstream access.
 * Returns Q·K. */
float wubu_polar_rambus_fused_dot(
    const wubu_polar_pso_t *pso,
    const float *q,           /* [d] query */
    const uint8_t *k_seed,    /* packed K bitstream */
    int k_bytes);

#endif /* WUBU_POLAR_PSO_H */

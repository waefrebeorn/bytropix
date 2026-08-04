/*
 * wubu_polar_pso.c — Meta-compiled PolarQuant decode kernels
 *
 * Implements PSO-cached decode, procedural precache, and Rambus-style
 * serial bit reading for PolarQuant KV cache.
 */

#include "wubu_polar_pso.h"
#include "wubu_polarquant.h"
#include "wubu_mobius.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ==========================================================
 * FWHT — same as wubu_polarquant.c
 * ========================================================== */

static void fwht(float *v, int d) {
    for (int h = 1; h < d; h <<= 1) {
        for (int i = 0; i < d; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                float a = v[j], b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
        }
    }
    float norm = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) v[i] *= norm;
}

/* ==========================================================
 * PSO: Pre-compiled decode kernels
 *
 * Each kernel is specialized for a (bits, d) config.
 * Uses precomputed cos/sin tables for angle reconstruction.
 * ========================================================== */

/* ==========================================================
 * Rambus-style serial bit reader decode
 *
 * Reads the packed bitstream through a 64-bit shift register.
 * Decodes the polar angles serially, reconstructing K inline.
 * ========================================================== */

/* Fast recursive polar decode using serial bit reader.
 * NO MALLOC — uses thread-local scratch arena.
 * Recursion levels reuse pre-allocated buffers. */
static void rambus_decode_recursive_nomalloc(
    wubu_bit_reader_t *br,
    int n, int d_orig, const float *cos_tbl, const float *sin_tbl,
    int bits, float final_r, float *x_out,
    int *scratch_angles, float *scratch_radii) {
    if (n <= 1) {
        if (n == 1) x_out[0] = final_r;
        return;
    }

    int n_pairs = n / 2;
    int rem = n - n_pairs * 2;
    int levels = 1 << bits;
    int is_level0 = (n == d_orig); /* Level 0: full circle; deeper: half range */

    /* Read angles from serial bitstream — no malloc */
    for (int p = 0; p < n_pairs; p++) {
        scratch_angles[p] = wubu_bit_reader_pop(br, bits);
    }

    /* Recursive: decode the radii using remaining scratch */
    float *sub_radii = scratch_radii + n_pairs;  /* nested space */
    if (n_pairs > 1) {
        rambus_decode_recursive_nomalloc(br, n_pairs, d_orig, cos_tbl, sin_tbl,
                                         bits, final_r, scratch_radii,
                                         scratch_angles + n_pairs, sub_radii);
    } else {
        scratch_radii[0] = final_r;
    }

    /* Reconstruct (x, y) using precomputed trig tables.
     * Level 0: theta = idx/levels * 2π − π  (full circle [-π,π])
     * Levels ≥1: theta = idx/levels * π − π/2  (half range [-π/2,π/2])
     * Match the angle normalization used by polar_encode_recursive. */
    for (int p = 0; p < n_pairs; p++) {
        int idx = scratch_angles[p];
        if (idx >= levels) idx = levels - 1;
        if (idx < 0) idx = 0;
        float c, s;
        if (cos_tbl && is_level0) {
            /* Full-circle table valid only at level 0 */
            c = cos_tbl[idx];
            s = sin_tbl[idx];
        } else {
            float norm_a = (float)idx / (float)levels;
            float theta = is_level0
                ? (norm_a * 2.0f * (float)M_PI - (float)M_PI)
                : (norm_a * (float)M_PI - (float)M_PI / 2.0f);
            c = cosf(theta);
            s = sinf(theta);
        }
        float r = scratch_radii[p];
        x_out[2*p]   = r * c;
        x_out[2*p+1] = r * s;
    }

    if (rem > 0) {
        x_out[n - 1] = final_r;
    }
}

/* Forward declarations */
static void pso_decode_fast(const wubu_polarquant_t *pq,
    const uint8_t *packed, int nbytes, float *out, int d);

/* ==========================================================
 * Scratch arena — eliminates malloc from decode hot path.
 *
 * Pre-allocate enough scratch for d=1024 (max head_dim).
 * The recursive decode uses this instead of per-call malloc.
 * This is the game-dev "pre-allocated decompression buffer"
 * pattern (same as DirectStorage's fixed working buffer).
 * ========================================================== */

#define WUBU_POLAR_MAX_D 1024
#define WUBU_POLAR_MAX_LEVELS 10  /* log2(1024) */

typedef struct {
    float  radii_buf[WUBU_POLAR_MAX_D];      /* max level 0 radii */
    int    angles_buf[WUBU_POLAR_MAX_D];      /* max level 0 angles */
    float  x_buf[WUBU_POLAR_MAX_D];           /* decode output */
    float  sub_radii[WUBU_POLAR_MAX_D / 2];   /* recursive radii */
} wubu_polar_scratch_t;

/* Thread-local scratch arena (avoids malloc per decode) */
static __thread wubu_polar_scratch_t g_scratch;

/* Thread-local PSO pointer — set before batch decode, read by decode kernel */
static __thread const wubu_polar_pso_t *tl_pso = NULL;

/* Public wrapper for PSO decode */
void wubu_pso_decode(const uint8_t *packed, int nbytes, float *out, int d) {
    pso_decode_fast(NULL, packed, nbytes, out, d);
}

/* Set thread-local PSO context (enables trig tables for decode) */
void wubu_pso_set_context(const wubu_polar_pso_t *pso) {
    tl_pso = pso;
}

/* PSO decode function: serial bit reader + scratch arena + trig tables.
 * Thread-local tl_pso provides trig tables and bit-width. */
static void pso_decode_fast(
    const wubu_polarquant_t *pq_unused,
    const uint8_t *packed, int nbytes,
    float *out, int d) {
    (void)pq_unused;

    /* Read the final radius (first 4 bytes) */
    float final_r;
    memcpy(&final_r, packed, sizeof(float));

    /* Initialize serial bit reader for angle data */
    wubu_bit_reader_t br;
    wubu_bit_reader_init(&br, packed + sizeof(float),
                         nbytes - (int)sizeof(float));

    /* Get bits from thread-local PSO, or default to 8 */
    int bits = tl_pso ? tl_pso->bits : 8;
    const float *cos_tbl = tl_pso ? tl_pso->cos_table : NULL;
    const float *sin_tbl = tl_pso ? tl_pso->sin_table : NULL;

    /* Decode using thread-local scratch (NO MALLOC) */
    rambus_decode_recursive_nomalloc(&br, d, d, cos_tbl, sin_tbl,
        bits, final_r, out,
        g_scratch.angles_buf, g_scratch.radii_buf);

    /* Undo Hadamard rotation */
    if ((d & (d - 1)) == 0) {
        fwht(out, d);
    }
}

/* ==========================================================
 * PSO Lifecycle
 * ========================================================== */

int wubu_polar_pso_init(wubu_polar_pso_t *pso,
    wubu_polarquant_t *pq, int bits, int d) {
    if (!pso || !pq || bits < 1 || bits > 16 || d <= 0) return -1;

    pso->bits = bits;
    pso->d = d;
    pso->storage_bytes = wubu_polarquant_storage_bytes(pq, d);
    pso->decode = pso_decode_fast;

    /* Precompute cos/sin tables for this bit-width */
    int levels = 1 << bits;
    pso->cos_table = (float *)malloc((size_t)levels * sizeof(float));
    pso->sin_table = (float *)malloc((size_t)levels * sizeof(float));

    if (!pso->cos_table || !pso->sin_table) {
        free(pso->cos_table); free(pso->sin_table);
        return -1;
    }

    for (int i = 0; i < levels; i++) {
        float theta = ((float)i / (float)levels) * 2.0f * (float)M_PI
                      - (float)M_PI;
        pso->cos_table[i] = cosf(theta);
        pso->sin_table[i] = sinf(theta);
    }

    return 0;
}

void wubu_polar_pso_free(wubu_polar_pso_t *pso) {
    if (!pso) return;
    free(pso->cos_table);
    free(pso->sin_table);
    memset(pso, 0, sizeof(*pso));
}

/* ==========================================================
 * Procedural precache
 * ========================================================== */

int wubu_polar_precache_init(wubu_polar_precache_t *pc,
    wubu_polarquant_t *pq, int bits, int d, int max_tokens) {
    if (!pc || !pq || d <= 0 || max_tokens <= 0) return -1;

    if (wubu_polar_pso_init(&pc->pso, pq, bits, d) != 0) return -1;

    pc->pq = pq;
    pc->d = d;
    pc->max_tokens = max_tokens;
    pc->n_tokens = 0;
    pc->bytes_per_token = pc->pso.storage_bytes + 16;

    pc->k_seed = (uint8_t *)malloc((size_t)max_tokens * pc->bytes_per_token);
    pc->v_seed = (uint8_t *)malloc((size_t)max_tokens * pc->bytes_per_token);
    pc->seed_bytes = (int *)malloc((size_t)max_tokens * sizeof(int));

    if (!pc->k_seed || !pc->v_seed || !pc->seed_bytes) {
        free(pc->k_seed); free(pc->v_seed); free(pc->seed_bytes);
        wubu_polar_pso_free(&pc->pso);
        return -1;
    }

    return 0;
}

void wubu_polar_precache_free(wubu_polar_precache_t *pc) {
    if (!pc) return;
    wubu_polar_pso_free(&pc->pso);
    free(pc->k_seed);
    free(pc->v_seed);
    free(pc->seed_bytes);
    memset(pc, 0, sizeof(*pc));
}

int wubu_polar_precache_push(wubu_polar_precache_t *pc,
    const float *k, const float *v) {
    if (!pc || !k || !v) return -1;
    if (pc->n_tokens >= pc->max_tokens) return -1;

    int idx = pc->n_tokens++;
    int ob = pc->bytes_per_token;

    /* Pack K using wubu_polarquant API */
    float *k_dst = (float *)&pc->k_seed[idx * pc->bytes_per_token];
    wubu_polarquant_quantize_kv(pc->pq, k, k_dst, &ob);
    pc->seed_bytes[idx] = ob;

    /* Pack V */
    ob = pc->bytes_per_token;
    float *v_dst = (float *)&pc->v_seed[idx * pc->bytes_per_token];
    wubu_polarquant_quantize_kv(pc->pq, v, v_dst, &ob);

    return 0;
}

int wubu_polar_precache_decode_k(wubu_polar_precache_t *pc,
    int token_idx, float *k_out) {
    if (!pc || token_idx < 0 || token_idx >= pc->n_tokens) return -1;

    const uint8_t *seed = &pc->k_seed[token_idx * pc->bytes_per_token];
    int nbytes = pc->seed_bytes[token_idx];

    tl_pso = &pc->pso;  /* enable trig tables */
    pso_decode_fast(NULL, seed, nbytes, k_out, pc->d);
    tl_pso = NULL;
    return 0;
}

int wubu_polar_precache_decode_v(wubu_polar_precache_t *pc,
    int token_idx, float *v_out) {
    if (!pc || token_idx < 0 || token_idx >= pc->n_tokens) return -1;

    const uint8_t *seed = &pc->v_seed[token_idx * pc->bytes_per_token];
    int nbytes = pc->seed_bytes[token_idx]; /* V same size as K */

    tl_pso = &pc->pso;
    pso_decode_fast(NULL, seed, nbytes, v_out, pc->d);
    tl_pso = NULL;
    return 0;
}

/* ==========================================================
 * Attention with online softmax + PSO decode
 * ========================================================== */

int wubu_polar_precache_attention(wubu_polar_precache_t *pc,
    const float *q, float *out, float temperature,
    int n_recent_f32, const float *recent_k, const float *recent_v) {
    if (!pc || !q || !out) return -1;
    int d = pc->d;

    float max_s = -1e30f;
    float sum_e = 0.0f;
    float *p_out = (float *)alloca((size_t)d * sizeof(float));
    if (!p_out) return -1;
    memset(p_out, 0, (size_t)d * sizeof(float));

    /* F32 recent tokens (high accuracy for most-attended) */
    int n_recent = n_recent_f32 < pc->n_tokens ? n_recent_f32 : pc->n_tokens;
    for (int i = 0; i < n_recent; i++) {
        const float *k = &recent_k[i * d];
        const float *v = &recent_v[i * d];
        float s = 0.0f;
        for (int j = 0; j < d; j++) s += q[j] * k[j];
        s /= temperature;
        if (s > max_s) {
            float om = max_s; max_s = s;
            sum_e = sum_e * expf(om - max_s) + 1.0f;
            float sc = expf(om - max_s);
            for (int j = 0; j < d; j++) { p_out[j] *= sc; p_out[j] += v[j]; }
        } else {
            float e = expf(s - max_s); sum_e += e;
            for (int j = 0; j < d; j++) p_out[j] += e * v[j];
        }
    }

    /* Quantized tokens — PSO decode with serial bit reader + trig tables.
     * Set tl_pso ONCE for the batch (avoids per-token overhead). */
    tl_pso = &pc->pso;
    for (int i = n_recent; i < pc->n_tokens; i++) {
        /* Decode K — uses thread-local scratch, no malloc */
        float *k_dec = g_scratch.x_buf;  /* reuse scratch buffer */
        const uint8_t *k_seed = &pc->k_seed[i * pc->bytes_per_token];
        pso_decode_fast(NULL, k_seed, pc->seed_bytes[i], k_dec, d);

        float s = 0.0f;
        for (int j = 0; j < d; j++) s += q[j] * k_dec[j];
        s /= temperature;

        /* Decode V — needs separate buffer since K is still in use */
        float v_dec[WUBU_POLAR_MAX_D];
        const uint8_t *v_seed = &pc->v_seed[i * pc->bytes_per_token];
        pso_decode_fast(NULL, v_seed, pc->seed_bytes[i], v_dec, d);

        if (s > max_s) {
            float om = max_s; max_s = s;
            sum_e = sum_e * expf(om - max_s) + 1.0f;
            float sc = expf(om - max_s);
            for (int j = 0; j < d; j++) { p_out[j] *= sc; p_out[j] += v_dec[j]; }
        } else {
            float e = expf(s - max_s); sum_e += e;
            for (int j = 0; j < d; j++) p_out[j] += e * v_dec[j];
        }
    }
    tl_pso = NULL;  /* restore */

    for (int j = 0; j < d; j++) out[j] = p_out[j] / (sum_e + 1e-10f);
    return 0;
}

/* ==========================================================
 * Rambus-style fused decode + dot
 * ========================================================== */

float wubu_polar_rambus_fused_dot(
    const wubu_polar_pso_t *pso,
    const float *q,
    const uint8_t *k_seed,
    int k_bytes) {
    int d = pso->d;
    
    /* Decode K inline using serial bit reader — stack buffer, no malloc */
    float k_buf[WUBU_POLAR_MAX_D];
    pso_decode_fast(NULL, k_seed, k_bytes, k_buf, d);

    /* Dot product */
    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += q[i] * k_buf[i];

    return dot;
}

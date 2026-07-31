/*
 * wubu_polarquant.c — PolarQuant recursive polar decomposition
 *
 * Implements the recursive pairwise polar transform from the PolarQuant
 * paper (arXiv:2502.02617). Pairs of coordinates are transformed to
 * (radius, angle); radii are recursively paired and transformed again
 * until one final radius remains. The angles concentrate after random
 * preconditioning, allowing sub-4-bit quantization with no per-block
 * scale overhead.
 *
 * The recursion tree IS fractal stacking on the Poincaré sphere:
 * each level's radii map into progressively smaller Poincaré balls.
 *
 * Uses wubu_mobius.h for exp_map/log_map at each fractal level.
 */

#include "wubu_polarquant.h"
#include "wubu_mobius.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ==========================================================
 * Forward declarations
 * ========================================================== */
static int level_from_n(int d_orig, int n);

/* ==========================================================
 * Preconditioning: Hadamard rotation
 * ========================================================== */

/* Fast Walsh-Hadamard Transform (FWHT) — O(d log d) in-place.
 * This is the optimal orthogonal rotation for outlier suppression
 * (MixQuant 2026: full-vector Hadamard outperforms block rotations
 * when δ < 1/√d, which is the common case for KV cache vectors).
 *
 * For d=128 (power of 2): 7 levels of butterfly operations.
 * Each level: d/2 add/subtract pairs = 64 ops → 7*64 = 448 total.
 * Inverse FWHT is identical (H * H = I for normalized Hadamard). */
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
    /* Normalize by 1/√d for unitary transform */
    float norm = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) v[i] *= norm;
}

static void apply_precondition(const wubu_polarquant_t *pq,
                               const float *x, float *x_out) {
    int d = pq->d;
    memcpy(x_out, x, (size_t)d * sizeof(float));
    /* Only works for power-of-2 dimensions (which head_dim always is) */
    if ((d & (d - 1)) == 0) {
        fwht(x_out, d);
    }
    /* For non-power-of-2: fall back to identity (no rotation) */
}

static void undo_precondition(const wubu_polarquant_t *pq,
                              const float *x, float *x_out) {
    int d = pq->d;
    memcpy(x_out, x, (size_t)d * sizeof(float));
    /* FWHT is its own inverse (H * H = I for normalized Hadamard) */
    if ((d & (d - 1)) == 0) {
        fwht(x_out, d);
    }
}

/* Thread-local scratch arenas — zero malloc on encode/decode hot path.
 * Sized for d <= 1024 (max head_dim).
 * g_pq_radii: used for radii during encode recursion + decode output
 * g_pq_rot:   used for preconditioned input during encode + inter/radii during decode
 * g_pq_angles: used for angle indices during encode/dequantize
 * NOTE: decode needs x_out (caller buffer) + intermediate radii + x_unrot.
 *   g_pq_rot serves double duty: radii_scratch OR x_unrot (never both). */
#define WUBU_PQ_MAX_D 1024
static __thread float g_pq_radii_scratch[WUBU_PQ_MAX_D];
static __thread float g_pq_rot_scratch[WUBU_PQ_MAX_D];
static __thread int   g_pq_angles_scratch[WUBU_PQ_MAX_D];

/* ==========================================================
 * Recursive polar encode: d-dim vector → (d-1) angle indices + 1 radius
 *
 * NO MALLOC — uses thread-local scratch buffers.
 * ========================================================== */

static void polar_encode_recursive(const float *x, int n, int d_orig,
    const wubu_polarquant_t *pq,
    int *angle_buf, int *angle_pos,
    int *bits_buf, float *out_final_r,
    float *radii_scratch) {
    if (n <= 1) {
        if (n == 1) *out_final_r = x[0];
        else *out_final_r = 0.0f;
        return;
    }

    int n_pairs = n / 2;
    int rem = n - n_pairs * 2;

    int level = level_from_n(d_orig, n);
    int bits = wubu_polarquant_bits_at(pq, level);
    int levels = 1 << bits;

    /* Use caller-provided scratch instead of malloc */
    for (int p = 0; p < n_pairs; p++) {
        float rx = x[2*p], ry = x[2*p+1];
        float r = sqrtf(rx*rx + ry*ry);
        float theta = atan2f(ry, rx);
        radii_scratch[p] = r;

        /* Angle normalization per PolarQuant paper (arXiv:2502.02617):
         * Level 0 (n == d_orig): angles uniform in [0, 2π) → map [−π,π]→[0,1]
         * Levels ≥1: angles concentrate in [0, π/2] → map [−π/2,π/2]→[0,1]
         * Deeper levels have tighter concentration → utilize full codebook. */
        float norm_a;
        if (n == d_orig) {
            /* Level 0: full circle */
            norm_a = (theta + (float)M_PI) / (2.0f * (float)M_PI);
        } else {
            /* Deeper levels: half range [−π/2, π/2] */
            float theta_clamped = theta;
            if (theta_clamped > (float)M_PI / 2.0f) theta_clamped -= (float)M_PI;
            if (theta_clamped < -(float)M_PI / 2.0f) theta_clamped += (float)M_PI;
            norm_a = (theta_clamped + (float)M_PI / 2.0f) / (float)M_PI;
        }
        int idx = (int)(norm_a * levels);
        if (idx >= levels) idx = levels - 1;
        if (idx < 0) idx = 0;
        angle_buf[(*angle_pos)++] = idx;
        bits_buf[(*angle_pos) - 1] = bits;
    }

    float sub_r;
    /* Nested scratch: level 0 uses [0, d/2), level 1 uses [d/2, 3d/4), etc. */
    polar_encode_recursive(radii_scratch, n_pairs, d_orig, pq,
        angle_buf, angle_pos, bits_buf, &sub_r,
        radii_scratch + n_pairs);

    if (rem > 0) {
        *out_final_r = x[n - 1];
    } else {
        *out_final_r = sub_r;
    }
}

/* ==========================================================
 * Recursive polar decode: (d-1) angle indices + 1 radius → d-dim vector
 *
 * NO MALLOC — uses thread-local scratch buffers.
 * ========================================================== */

static void polar_decode_recursive(int n, int d_orig,
    const wubu_polarquant_t *pq,
    const int *angle_buf, int *angle_pos,
    const int *bits_buf,
    float final_r, float *x_out,
    int *angles_scratch, float *radii_scratch) {
    if (n <= 1) {
        if (n == 1) x_out[0] = final_r;
        return;
    }

    int n_pairs = n / 2;
    int rem = n - n_pairs * 2;

    int level = level_from_n(d_orig, n);
    int bits = wubu_polarquant_bits_at(pq, level);
    int levels = 1 << bits;

    /* Read angles into scratch — no malloc */
    for (int p = 0; p < n_pairs; p++) {
        angles_scratch[p] = angle_buf[(*angle_pos)++];
    }

    /* Recursive: decode the radii using nested scratch */
    if (n_pairs > 1) {
        polar_decode_recursive(n_pairs, d_orig, pq, angle_buf, angle_pos,
                               bits_buf, final_r, radii_scratch,
                               angles_scratch + n_pairs, radii_scratch + n_pairs);
    } else {
        radii_scratch[0] = final_r;
    }

    for (int p = 0; p < n_pairs; p++) {
        int idx = angles_scratch[p];
        float norm_a = (float)idx / (float)levels;
        /* Decode must match encode normalization:
         * Level 0 (n == d_orig): [0,1] → [−π, π] (full circle)
         * Levels ≥1: [0,1] → [−π/2, π/2] (half range, paper: [0, π/2]) */
        float theta;
        if (n == d_orig) {
            theta = norm_a * 2.0f * (float)M_PI - (float)M_PI;
        } else {
            theta = norm_a * (float)M_PI - (float)M_PI / 2.0f;
        }
        float r = radii_scratch[p];
        x_out[2*p]   = r * cosf(theta);
        x_out[2*p+1] = r * sinf(theta);
    }

    if (rem > 0) {
        x_out[n - 1] = final_r;
    }
}

/* ==========================================================
 * Public lifecycle
 * ========================================================== */

/* Compute recursion level for a given sub-vector size n.
 * level 0: d, level 1: d/2, level 2: d/4, ... */
static int level_from_n(int d_orig, int n) {
    int level = 0;
    int cur = d_orig;
    while (cur > n && cur > 1) { cur >>= 1; level++; }
    return level;
}

int wubu_polarquant_init(wubu_polarquant_t *pq, int d, int depth,
                                       float R_max, float bits_per_coord) {
    return wubu_polarquant_init_perlevel(pq, d, R_max, bits_per_coord, NULL, 0);
}

int wubu_polarquant_init_perlevel(wubu_polarquant_t *pq, int d,
    float R_max, float default_bits,
    const int *bits_array, int n) {
    if (!pq || d <= 0) return -1;
    if (default_bits < 1 || default_bits > 16) return -1;

    memset(pq, 0, sizeof(*pq));
    pq->d = d;
    pq->depth = 1;
    pq->R_max = R_max;
    pq->bits_per_coord = default_bits;
    pq->n_levels = 0;

    if (bits_array && n > 0) {
        int nmax = n > 16 ? 16 : n;
        for (int i = 0; i < nmax; i++) {
            pq->bits_per_level[i] = bits_array[i];
        }
        pq->n_levels = nmax;
    }

    pq->level_scale = (float *)malloc(sizeof(float));
    pq->rand_precondition = (float *)malloc((size_t)d * sizeof(float));
    pq->bits_store = (int *)malloc((size_t)d * sizeof(int));
    if (!pq->level_scale || !pq->rand_precondition || !pq->bits_store) {
        free(pq->level_scale); free(pq->rand_precondition);
        free(pq->bits_store);
        return -1;
    }

    pq->level_scale[0] = R_max;
    float phi = (sqrtf(5.0f) + 1.0f) / 2.0f;
    for (int i = 0; i < d; i++) {
        pq->rand_precondition[i] = 2.0f * (float)M_PI * (float)i * phi / (float)d;
    }

    return 0;
}

void wubu_polarquant_free(wubu_polarquant_t *pq) {
    if (!pq) return;
    free(pq->level_scale);
    free(pq->rand_precondition);
    free(pq->bits_store);
    memset(pq, 0, sizeof(*pq));
}

/* ==========================================================
 * Encode: Cartesian → recursive polar (angle indices + final radius)
 * ========================================================== */

int wubu_polarquant_encode(const wubu_polarquant_t *pq,
                                  const float *x,
                                  float *level_radius,
                                  int   *level_angle_idx,
                                  int   *n_angles_per_level) {
    if (!pq || !x || !level_radius || !level_angle_idx || !n_angles_per_level) return -1;

    int d = pq->d;

    /* Precondition: Hadamard rotation to spread outliers.
     * Use scratch arena instead of malloc. */
    float *x_rot = g_pq_rot_scratch;
    apply_precondition(pq, x, x_rot);

    /* Recursive polar decomposition with scratch arena (NO MALLOC) */
    int angle_pos = 0;
    float final_r;
    int bits_buf[1024];
    polar_encode_recursive(x_rot, d, d, pq, level_angle_idx, &angle_pos,
                           bits_buf, &final_r, g_pq_radii_scratch);

    level_radius[0] = final_r;
    n_angles_per_level[0] = angle_pos;

    /* Store bit widths for quantize_kv */
    if (pq->bits_store) {
        memcpy(pq->bits_store, bits_buf, (size_t)angle_pos * sizeof(int));
    }

    return 0;
}

/* ==========================================================
 * Decode: recursive polar → Cartesian
 * ========================================================== */

int wubu_polarquant_decode(const wubu_polarquant_t *pq,
                                  const float *level_radius,
                                  const int   *level_angle_idx,
                                  const int   *n_angles_per_level,
                                  float *x_out, int d_out) {
    if (!pq || !x_out) return -1;
    memset(x_out, 0, (size_t)d_out * sizeof(float));

    /* Recursive polar reconstruction with per-level bits */
    int angle_pos = 0;
    int bits_buf[1024];
    /* Reconstruct bits_buf from pq's per-level config */
    int idx = 0;
    int n = d_out;
    while (n > 1) {
        int n_pairs = n / 2;
        int level = level_from_n(pq->d, n);
        int b = wubu_polarquant_bits_at(pq, level);
        for (int p = 0; p < n_pairs; p++) {
            bits_buf[idx++] = b;
        }
        n = n_pairs;
    }

    /* Use g_pq_rot_scratch as intermediate radii buffer during decode.
     * x_out is the output — cannot be the same as radii_scratch,
     * so use the separate rot buffer for intermediates. */
    polar_decode_recursive(d_out, d_out, pq, level_angle_idx, &angle_pos,
                           bits_buf, level_radius[0], x_out,
                           g_pq_angles_scratch, g_pq_rot_scratch);

    (void)bits_buf;

    /* Undo preconditioning.
     * Use scratch arena instead of malloc. */
    float *x_unrot = g_pq_rot_scratch;
    undo_precondition(pq, x_out, x_unrot);
    memcpy(x_out, x_unrot, (size_t)d_out * sizeof(float));

    return 0;
}

/* ==========================================================
 * KV quantize/dequantize (packed bitstream)
 * ========================================================== */

int wubu_polarquant_quantize_kv(const wubu_polarquant_t *pq,
                                           const float *k_col,
                                           float *out_bits, int *out_bytes) {
    if (!pq || !k_col || !out_bits || !out_bytes) return -1;

    int d = pq->d;
    int n_angles = d - 1;

    float level_radius[1];
    int n_per_level[1];

    /* Encode into thread-local scratch — NO MALLOC */
    wubu_polarquant_encode(pq, k_col, level_radius,
                           g_pq_angles_scratch, n_per_level);

    /* Compute total bit width from per-level config */
    int total_bits = 0;
    int n = d;
    while (n > 1) {
        int n_pairs = n / 2;
        int level = level_from_n(d, n);
        int b = wubu_polarquant_bits_at(pq, level);
        total_bits += n_pairs * b;
        n = n_pairs;
    }
    int packed_angle_bytes = (total_bits + 7) / 8;
    int need = (int)sizeof(float) + packed_angle_bytes;
    if (*out_bytes < need) return -1;

    float *fp = (float *)out_bits;
    fp[0] = level_radius[0];

    /* Pack angle indices from thread-local scratch */
    uint8_t *bp = (uint8_t *)(fp + 1);
    memset(bp, 0, (size_t)packed_angle_bytes);
    int bit_pos = 0;
    for (int i = 0; i < n_angles; i++) {
        int bits = pq->bits_store ? pq->bits_store[i] : (int)pq->bits_per_coord;
        int val = g_pq_angles_scratch[i];
        for (int b = 0; b < bits; b++) {
            if (val & (1 << b)) {
                bp[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }

    *out_bytes = need;
    return 0;
}

int wubu_polarquant_dequantize_kv(const wubu_polarquant_t *pq,
                                           const float *in_bits, int in_bytes,
                                           float *k_col_out, int d) {
    if (!pq || !in_bits || !k_col_out) return -1;

    int n_angles = d - 1;

    /* Compute total packed bits and verify size */
    int total_bits = 0;
    {
        int n = d;
        while (n > 1) {
            int n_pairs = n / 2;
            int level = level_from_n(pq->d, n);
            int b = wubu_polarquant_bits_at(pq, level);
            total_bits += n_pairs * b;
            n = n_pairs;
        }
    }
    int packed_angle_bytes = (total_bits + 7) / 8;
    int need = (int)sizeof(float) + packed_angle_bytes;
    if (in_bytes < need) return -1;

    const float *fp = in_bits;
    float level_radius[1];
    level_radius[0] = fp[0];

    int n_per_level[1];

    /* Unpack with per-angle bit widths into thread-local scratch.
     * NO MALLOC — uses g_pq_angles_scratch. */
    const uint8_t *bp = (const uint8_t *)(fp + 1);
    int bit_pos = 0;
    int idx = 0;
    int n = d;
    while (n > 1) {
        int n_pairs = n / 2;
        int level = level_from_n(pq->d, n);
        int bits = wubu_polarquant_bits_at(pq, level);
        for (int p = 0; p < n_pairs; p++) {
            int val = 0;
            for (int b = 0; b < bits; b++) {
                if (bp[bit_pos / 8] & (1 << (bit_pos % 8))) {
                    val |= (1 << b);
                }
                bit_pos++;
            }
            g_pq_angles_scratch[idx++] = val;
        }
        n = n_pairs;
    }
    n_per_level[0] = n_angles;

    int rc = wubu_polarquant_decode(pq, level_radius,
                                    g_pq_angles_scratch, n_per_level,
                                    k_col_out, d);
    return rc;
}

/* ==========================================================
 * Fused decode + attention dot product
 * ========================================================== */

float wubu_polarquant_fused_dot(
        const wubu_polarquant_t *pq,
        const float *q,
        const float *k_packed,
        int k_bytes) {
    /* Decode K (includes inverse Hadamard, so K is in original space).
     * Q is also in original space — no Hadamard needed on Q. */
    int d = pq->d;
    
    /* Decode K inline into thread-local scratch — NO MALLOC */
    float *k = g_pq_radii_scratch;  /* reuse as decode buffer */
    wubu_polarquant_dequantize_kv(pq, k_packed, k_bytes, k, d);

    /* Compute dot product */
    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += q[i] * k[i];
    return dot;
}

/* ==========================================================
 * Mixed-precision KV cache (KIVI/Kitty pattern)
 * ========================================================== */

int wubu_polar_cache_init(wubu_polar_cache_t *c,
    wubu_polarquant_t *pq, int d, int n_recent, int capacity) {
    if (!c || !pq || d <= 0 || n_recent < 0 || capacity <= 0) return -1;
    
    c->pq = pq;
    c->d = d;
    c->n_recent = n_recent;
    c->capacity = capacity;
    c->n_filled = 0;
    c->recent_head = 0;
    /* Pack bytes as floats: each token takes max_storage_bytes bytes,
     * but we index through float*, so allocate in float units */
    int max_bytes = wubu_polarquant_storage_bytes(pq, d) + 16;
    /* Round up to float boundary */
    c->max_bytes_per_token = (max_bytes + (int)sizeof(float) - 1) / (int)sizeof(float);
    
    if (n_recent > 0) {
        c->recent_k = (float *)malloc((size_t)n_recent * d * sizeof(float));
        c->recent_v = (float *)malloc((size_t)n_recent * d * sizeof(float));
    } else {
        c->recent_k = NULL;
        c->recent_v = NULL;
    }
    
    /* quant buffers indexed as float arrays */
    c->quant_k = (float *)malloc((size_t)capacity * c->max_bytes_per_token * sizeof(float));
    c->quant_v = (float *)malloc((size_t)capacity * c->max_bytes_per_token * sizeof(float));
    c->quant_bytes = (int *)malloc((size_t)capacity * sizeof(int));
    
    if ((n_recent > 0 && (!c->recent_k || !c->recent_v)) ||
        !c->quant_k || !c->quant_v || !c->quant_bytes) {
        free(c->recent_k); free(c->recent_v);
        free(c->quant_k); free(c->quant_v); free(c->quant_bytes);
        return -1;
    }
    
    return 0;
}

void wubu_polar_cache_free(wubu_polar_cache_t *c) {
    if (!c) return;
    free(c->recent_k);
    free(c->recent_v);
    free(c->quant_k);
    free(c->quant_v);
    free(c->quant_bytes);
    memset(c, 0, sizeof(*c));
}

int wubu_polar_cache_push(wubu_polar_cache_t *c,
    const float *k, const float *v) {
    if (!c || !k || !v) return -1;
    if (c->n_filled >= c->capacity) return -1;
    
    int idx = c->n_filled++;
    
    /* Store in F32 ring buffer if we have room in recent section */
    if (idx < c->n_recent) {
        memcpy(&c->recent_k[idx * c->d], k, (size_t)c->d * sizeof(float));
        memcpy(&c->recent_v[idx * c->d], v, (size_t)c->d * sizeof(float));
    } else {
        /* Quantize and store */
        int n_quant_idx = idx - c->n_recent;
        int ob = c->max_bytes_per_token * (int)sizeof(float);
        float *k_dst = &c->quant_k[n_quant_idx * c->max_bytes_per_token];
        float *v_dst = &c->quant_v[n_quant_idx * c->max_bytes_per_token];
        wubu_polarquant_quantize_kv(c->pq, k, k_dst, &ob);
        c->quant_bytes[n_quant_idx] = ob;
        ob = c->max_bytes_per_token * (int)sizeof(float);
        wubu_polarquant_quantize_kv(c->pq, v, v_dst, &ob);
        /* V byte count same as K for same config */
    }
    
    return 0;
}

int wubu_polar_cache_attention(wubu_polar_cache_t *c,
    const float *q, float *out, float temperature) {
    if (!c || !q || !out) return -1;
    int d = c->d;
    
    /* Online softmax (FlashDecoding++ pattern) */
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float *partial_out = (float *)alloca((size_t)d * sizeof(float));
    if (!partial_out) return -1;
    memset(partial_out, 0, (size_t)d * sizeof(float));
    
    /* F32 recent tokens */
    int n_recent = c->n_filled < c->n_recent ? c->n_filled : c->n_recent;
    for (int i = 0; i < n_recent; i++) {
        const float *k = &c->recent_k[i * d];
        const float *v = &c->recent_v[i * d];
        float score = 0.0f;
        for (int j = 0; j < d; j++) score += q[j] * k[j];
        score /= temperature;
        if (score > max_score) {
            float old_max = max_score;
            max_score = score;
            sum_exp = sum_exp * expf(old_max - max_score) + 1.0f;
            float scale = expf(old_max - max_score);
            for (int j = 0; j < d; j++) partial_out[j] *= scale;
            for (int j = 0; j < d; j++) partial_out[j] += v[j];
        } else {
            float e = expf(score - max_score);
            sum_exp += e;
            for (int j = 0; j < d; j++) partial_out[j] += e * v[j];
        }
    }
    
    /* Quantized older tokens — fused decode+dot */
    int n_quant = c->n_filled - c->n_recent;
    for (int i = 0; i < n_quant; i++) {
        const float *k_packed = &c->quant_k[i * c->max_bytes_per_token];
        const float *v_packed = &c->quant_v[i * c->max_bytes_per_token];
        int k_bytes = c->quant_bytes[i];
        
        /* Fused dot: Q · decode(K) */
        float score = wubu_polarquant_fused_dot(c->pq, q, k_packed, k_bytes);
        score /= temperature;
        
        /* Decode V for weighted sum — use scratch arena (NO MALLOC).
         * fused_dot already used g_pq_radii_scratch for K decode,
         * so it's safe to reuse for V. */
        float *v = g_pq_radii_scratch;
        wubu_polarquant_dequantize_kv(c->pq, v_packed, k_bytes, v, d);
        
        if (score > max_score) {
            float old_max = max_score;
            max_score = score;
            sum_exp = sum_exp * expf(old_max - max_score) + 1.0f;
            float scale = expf(old_max - max_score);
            for (int j = 0; j < d; j++) partial_out[j] *= scale;
            for (int j = 0; j < d; j++) partial_out[j] += v[j];
        } else {
            float e = expf(score - max_score);
            sum_exp += e;
            for (int j = 0; j < d; j++) partial_out[j] += e * v[j];
        }
    }

    /* Normalize */
    for (int j = 0; j < d; j++) out[j] = partial_out[j] / (sum_exp + 1e-10f);
    return 0;
}

/* ==========================================================
 * Bandwidth helpers
 * ========================================================== */

double wubu_polarquant_theoretical_bandwidth(const wubu_polarquant_t *pq, int d) {
    int storage = wubu_polarquant_storage_bytes(pq, d);
    return (double)storage * 2.0 * 40.0;
}

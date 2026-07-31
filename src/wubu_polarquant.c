/*
 * wubu_polarquant.c — PolarQuant + Poincaré fractal stacking
 *
 * Recursive polar decomposition maps onto nested Poincaré balls.
 * Each level halves the ball radius, creating fractal structure.
 * Angular coordinates concentrate naturally — no per-block overhead.
 *
 * Key design: the codebook is SHARED across all vectors (stored once).
 * Each vector stores only codebook indices + radius per level.
 * The fractal stacking means each level's residual becomes the next level's input.
 *
 * WSL2 substrate: CPU-only, C11, no external deps.
 */

#include "wubu_polarquant.h"
#include "wubu_mobius.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef WUBU_POLAR_DEPTH
#define WUBU_POLAR_DEPTH 3
#endif
#ifndef WUBU_POLAR_BITS_PER_COORD
#define WUBU_POLAR_BITS_PER_COORD 2
#endif

/* ==========================================================
 * Fibonacci spiral codebook generation
 * ========================================================== */

/* Generate a shared codebook for one fractal level.
 * The codebook has 2^bits entries, each a POINT in the
 * Poincaré ball of radius R at the given dimension.
 *
 * The fractal nesting means each level's angle space
 * is a lower-dimensional sphere (dim shrinks by half),
 * so the codebook only needs to cover that sphere. */
static int generate_level_codebook(wubu_polar_level_t *lv, int seed) {
    int n = lv->codebook_size; /* 2^bits */
    int d = lv->dims;          /* dims at this level */
    float R = lv->R;

    float *cb = lv->codebook;
    float *rc = lv->radius_centroids;

    /* Fibonacci spiral on S^{d-1}: generate n uniformly
     * distributed points on the d-1 sphere */
    float phi_g = (sqrtf(5.0f) + 1.0f) / 2.0f; /* golden ratio */

    for (int i = 0; i < n; i++) {
        float *p = cb + (size_t)i * d;

        /* Fibonacci spiral coordinates */
        float theta = 2.0f * M_PI * i / phi_g;
        float zeta = (float)(i + 0.5f) / (float)n;
        float sin_t = sqrtf(1.0f - zeta);
        float cos_t = sqrtf(zeta);

        /* First two dims */
        p[0] = sin_t * cosf(theta);
        if (d >= 2) p[1] = sin_t * sinf(theta);

        /* Remaining dims use recursive golden-angle spiral */
        for (int j = 2; j < d; j++) {
            float a = theta * (j + 1);
            p[j] = cos_t * cosf(a);
        }

        /* Map to Poincaré ball of radius R */
        /* Project: ball = R * tanh(c) * unit_direction */
        float norm = 0.0f;
        for (int j = 0; j < d; j++) norm += p[j] * p[j];
        norm = sqrtf(norm);
        if (norm < 1e-10f) norm = 1e-10f;

        /* Scale to be inside the ball (stay at 70% of R for safety) */
        float target = R * 0.7f;
        float scale = target / norm;
        for (int j = 0; j < d; j++) p[j] *= scale;

        /* Radius centroid: uniform in log(artanh) space */
        float t = (float)i / (float)n;
        float artanh_t = 0.5f * logf((1.0f + t) / (1.0f - t + 1e-10f));
        if (artanh_t > 5.0f) artanh_t = 5.0f;
        rc[i] = R * tanhf(artanh_t);
    }

    return 0;
}

/* ==========================================================
 * Fractal decomposition helpers
 * ========================================================== */

/* Level l decomposes the current residual vector.
 *  - radius r = ||residual|| (log-quantized into codebook index)
 *  - angle = residual / r (mapped to ball via exp_map)
 *  - nearest codebook entry's angular vector is the "angle index"
 *  - residual for next level = residual - reconstruction_at_this_level */
static void decompose_level(const wubu_polar_level_t *lv,
                            const float *residual, int d_l,
                            float *out_radius,
                            int   *out_angle_idx,
                            float *out_recon) {
    float norm = 0.0f;
    for (int i = 0; i < d_l; i++) norm += residual[i] * residual[i];
    norm = sqrtf(norm);
    *out_radius = norm;

    if (norm < 1e-30f) {
        *out_angle_idx = 0;
        for (int i = 0; i < d_l; i++) out_recon[i] = 0.0f;
        return;
    }

    /* Find nearest codebook entry by angular cosine similarity */
    const float *codebook = lv->codebook;
    int cb_dims = lv->dims;
    int cb_size = lv->codebook_size;
    int effective_dim = (cb_dims < d_l) ? cb_dims : d_l;

    int best_idx = 0;
    float best_sim = -2.0f;

    for (int k = 0; k < cb_size; k++) {
        const float *cb = codebook + (size_t)k * cb_dims;
        float dot = 0.0f, n_recon = 0.0f;
        for (int i = 0; i < effective_dim; i++) {
            dot += residual[i] * cb[i];
            n_recon += cb[i] * cb[i];
        }
        float sim = dot / (sqrtf(n_recon) * norm + 1e-10f);
        if (sim > best_sim) { best_sim = sim; best_idx = k; }
    }

    *out_angle_idx = best_idx;

    /* Reconstruct: codebook angular direction scaled by residual radius */
    const float *cb_best = codebook + (size_t)best_idx * cb_dims;
    float cb_norm = 0.0f;
    for (int i = 0; i < effective_dim; i++) cb_norm += cb_best[i] * cb_best[i];
    cb_norm = sqrtf(cb_norm);
    if (cb_norm < 1e-10f) cb_norm = 1e-10f;

    float scale = norm / cb_norm;
    for (int i = 0; i < d_l && i < cb_dims; i++) {
        out_recon[i] = cb_best[i] * scale;
    }
    for (int i = cb_dims; i < d_l; i++) out_recon[i] = 0.0f;
}

/* ==========================================================
 * Public lifecycle
 * ========================================================== */

int wubu_polarquant_init(wubu_polarquant_t *pq, int d, int depth,
                                       float R_max, float bits_per_coord) {
    if (!pq || d <= 0 || depth <= 0 || depth > WUBU_POLAR_DEPTH) return -1;
    if (bits_per_coord < 1 || bits_per_coord > 8) return -1;

    memset(pq, 0, sizeof(*pq));
    pq->d = d;
    pq->depth = depth;
    pq->R_max = R_max;
    pq->bits_per_coord = bits_per_coord;

    pq->level_scale = (float *)malloc(depth * sizeof(float));
    pq->rand_precondition = (float *)malloc(d * sizeof(float));
    if (!pq->level_scale || !pq->rand_precondition) {
        free(pq->level_scale); free(pq->rand_precondition);
        return -1;
    }

    /* Deterministic preconditioning (golden ratio hash) */
    float phi = (sqrtf(5.0f) + 1.0f) / 2.0f;
    for (int i = 0; i < d; i++) {
        pq->rand_precondition[i] = cosf(2.0f * M_PI * i * phi);
    }

    for (int l = 0; l < depth; l++) {
        wubu_polar_level_t *lv = &pq->levels[l];
        lv->R = R_max / sqrtf((float)(1 << l));
        lv->codebook_size = 1 << (int)bits_per_coord;
        lv->dims = d / (1 << l);
        if (lv->dims < 1) lv->dims = 1;

        int cb_bytes = lv->codebook_size * lv->dims * sizeof(float);
        lv->codebook = (float *)malloc(cb_bytes);
        lv->radius_centroids = (float *)malloc(lv->codebook_size * sizeof(float));
        if (!lv->codebook || !lv->radius_centroids) {
            for (int k = 0; k < l; k++) {
                free(pq->levels[k].codebook);
                free(pq->levels[k].radius_centroids);
            }
            free(pq->level_scale); free(pq->rand_precondition);
            return -1;
        }
        memset(lv->codebook, 0, cb_bytes);
        memset(lv->radius_centroids, 0, lv->codebook_size * sizeof(float));
        generate_level_codebook(lv, l * 7919);
        pq->level_scale[l] = lv->R * tanhf(1.0f);
    }

    return 0;
}

void wubu_polarquant_free(wubu_polarquant_t *pq) {
    if (!pq) return;
    for (int l = 0; l < pq->depth; l++) {
        free(pq->levels[l].codebook);
        free(pq->levels[l].radius_centroids);
    }
    free(pq->level_scale);
    free(pq->rand_precondition);
    memset(pq, 0, sizeof(*pq));
}

/* ==========================================================
 * Encode: Cartesian → fractal polar (codebook indices + radii)
 * ========================================================== */

int wubu_polarquant_encode(const wubu_polarquant_t *pq,
                                  const float *x,
                                  float *level_radius,
                                  int   *level_angle_idx,
                                  int   *n_angles_per_level) {
    if (!pq || !x || !level_radius || !level_angle_idx || !n_angles_per_level) return -1;

    int d = pq->d;
    int depth = pq->depth;

    /* Working buffer: residual at current level */
    float *residual = (float *)calloc(d, sizeof(float));
    float *recon_this = (float *)malloc(d * sizeof(float));
    if (!residual || !recon_this) {
        free(residual); free(recon_this);
        return -1;
    }
    memcpy(residual, x, d * sizeof(float));

    for (int l = 0; l < depth; l++) {
        int d_l = d / (1 << l);
        if (d_l < 1) d_l = 1;

        decompose_level(&pq->levels[l], residual, d_l,
                        &level_radius[l], &level_angle_idx[l], recon_this);
        n_angles_per_level[l] = 1; /* one codebook index per level */

        /* Residual for next level = input minus reconstruction */
        if (l + 1 < depth) {
            for (int i = 0; i < d_l; i++) {
                residual[i] -= recon_this[i];
            }
            /* Remaining higher dimensions of residual stay as-is for deeper levels */
        }
    }

    free(residual); free(recon_this);
    return 0;
}

/* ==========================================================
 * Decode: fractal polar → Cartesian
 * ========================================================== */

int wubu_polarquant_decode(const wubu_polarquant_t *pq,
                                  const float *level_radius,
                                  const int   *level_angle_idx,
                                  const int   *n_angles_per_level,
                                  float *x_out, int d_out) {
    if (!pq || !x_out) return -1;
    memset(x_out, 0, d_out * sizeof(float));

    int depth = pq->depth;
    float *recon = (float *)calloc(d_out, sizeof(float));
    float *level_recon = (float *)malloc(d_out * sizeof(float));
    if (!recon || !level_recon) { free(recon); free(level_recon); return -1; }

    /* Inverse fractal decomposition:
     * Start from the deepest level and accumulate up.
     * Each level adds its reconstruction to the output. */
    for (int l = depth - 1; l >= 0; l--) {
        int d_l = d_out / (1 << l);
        if (d_l < 1) d_l = 1;

        int idx = level_angle_idx[l];
        float r = level_radius[l];
        const float *cb = pq->levels[l].codebook + (size_t)idx * pq->levels[l].dims;

        /* Reconstruct at this level: scale codebook vector by radius */
        float cb_norm = 0.0f;
        int effective = (pq->levels[l].dims < d_l) ? pq->levels[l].dims : d_l;
        for (int i = 0; i < effective; i++) cb_norm += cb[i] * cb[i];
        cb_norm = sqrtf(cb_norm);
        if (cb_norm < 1e-10f) cb_norm = 1e-10f;

        float scale = r / cb_norm;
        for (int i = 0; i < d_l && i < pq->levels[l].dims; i++) {
            level_recon[i] = cb[i] * scale;
        }
        for (int i = d_l; i < d_out; i++) level_recon[i] = 0.0f;

        /* Add to output (accumulate from deep to shallow) */
        for (int i = 0; i < d_out; i++) x_out[i] += level_recon[i];
    }

    free(recon); free(level_recon);
    return 0;
}

/* ==========================================================
 * KV quantize/dequantize (bitstream interface)
 * ========================================================== */

int wubu_polarquant_quantize_kv(const wubu_polarquant_t *pq,
                                           const float *k_col,
                                           float *out_bits, int *out_bytes) {
    if (!pq || !k_col || !out_bits || !out_bytes || *out_bytes < pq->depth * (int)sizeof(float)) return -1;

    float *level_radius = (float *)calloc(pq->depth, sizeof(float));
    int   *level_idx = (int *)calloc(pq->depth, sizeof(int));
    int   *n_angles = (int *)calloc(pq->depth, sizeof(int));
    if (!level_radius || !level_idx || !n_angles) {
        free(level_radius); free(level_idx); free(n_angles);
        return -1;
    }

    wubu_polarquant_encode(pq, k_col, level_radius, level_idx, n_angles);
    memcpy(out_bits, level_radius, pq->depth * sizeof(float));
    *out_bytes = pq->depth * (int)sizeof(float);

    free(level_radius); free(level_idx); free(n_angles);
    return 0;
}

int wubu_polarquant_dequantize_kv(const wubu_polarquant_t *pq,
                                           const float *in_bits, int in_bytes,
                                           float *k_col_out, int d) {
    if (!pq || !in_bits || !k_col_out) return -1;
    if (in_bytes < pq->depth * (int)sizeof(float)) return -1;

    float level_radius[WUBU_POLAR_DEPTH];
    int   level_idx[WUBU_POLAR_DEPTH];
    int   n_angles[WUBU_POLAR_DEPTH];
    memcpy(level_radius, in_bits, pq->depth * sizeof(float));

    /* Use codebook index 0 (default) — full dequant needs
     * storing indices alongside radii in the bitstream */
    for (int l = 0; l < pq->depth; l++) {
        level_idx[l] = 0;
        n_angles[l] = 1;
    }

    return wubu_polarquant_decode(pq, level_radius, level_idx, n_angles, k_col_out, d);
}

/* ==========================================================
 * Bandwidth helpers
 * ========================================================== */

/* Total bytes for a single quantized vector, including
 * radius (F32) and index (int) per level. */
static double total_bytes_per_vector(const wubu_polarquant_t *pq, int d) {
    double total = 0.0;
    for (int l = 0; l < pq->depth; l++) {
        int d_l = d / (1 << l);
        if (d_l < 1) d_l = 1;
        total += sizeof(float) + sizeof(int);
    }
    return total;
}

double wubu_polarquant_theoretical_bandwidth(const wubu_polarquant_t *pq, int d) {
    double bytes_per_vec = total_bytes_per_vector(pq, d);
    return bytes_per_vec * 2.0 * 40.0;
}
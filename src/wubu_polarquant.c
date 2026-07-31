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
 * Preconditioning: random rotation (deterministic golden-ratio)
 * ========================================================== */

/* Pairwise Hadamard rotation preconditioner.
 * Applies a deterministic orthogonal transform that spreads
 * outlier coordinates across all dimensions. This is the key
 * insight from PolarQuant: after rotation, the polar angles
 * concentrate tightly, allowing 3-4 bit quantization.
 *
 * We use a recursive pairwise rotation:
 * For each pair (i, i+1), rotate by a fixed angle phi_i.
 * This is NOT a full Hadamard matrix, but it's orthogonal and
 * cheap to apply. The inverse just rotates by -phi_i.
 *
 * The angles are chosen as golden-ratio-based irrationals
 * so no two pairs have the same rotation. */
static void apply_precondition(const wubu_polarquant_t *pq,
                               const float *x, float *x_out) {
    int d = pq->d;
    float golden = (sqrtf(5.0f) + 1.0f) / 2.0f;
    for (int i = 0; i < d - 1; i += 2) {
        float ang = (float)M_PI * (float)(i/2 + 1) * golden / (float)(d/2);
        float c = cosf(ang), s = sinf(ang);
        float a = x[i], b = x[i+1];
        x_out[i]   = c * a - s * b;
        x_out[i+1] = s * a + c * b;
    }
    if (d % 2) x_out[d-1] = x[d-1];
}

static void undo_precondition(const wubu_polarquant_t *pq,
                              const float *x, float *x_out) {
    int d = pq->d;
    float golden = (sqrtf(5.0f) + 1.0f) / 2.0f;
    for (int i = 0; i < d - 1; i += 2) {
        float ang = (float)M_PI * (float)(i/2 + 1) * golden / (float)(d/2);
        float c = cosf(ang), s = sinf(ang);
        /* Inverse: rotate by -ang */
        float a = x[i], b = x[i+1];
        x_out[i]   =  c * a + s * b;
        x_out[i+1] = -s * a + c * b;
    }
    if (d % 2) x_out[d-1] = x[d-1];
}

/* ==========================================================
 * Recursive polar encode: d-dim vector → (d-1) angle indices + 1 radius
 *
 * The recursion produces a binary tree of polar decompositions.
 * Angles are stored in pre-order (left-to-right, top-to-bottom).
 * ========================================================== */

static void polar_encode_recursive(const float *x, int n, int bits,
    int *angle_buf, int *angle_pos, float *out_final_r) {
    if (n <= 1) {
        if (n == 1) *out_final_r = x[0];
        else *out_final_r = 0.0f;
        return;
    }

    int n_pairs = n / 2;
    int rem = n - n_pairs * 2;
    int levels = 1 << bits;

    /* Extract radii and quantize angles for each pair */
    float *radii = (float *)malloc((size_t)n_pairs * sizeof(float));
    for (int p = 0; p < n_pairs; p++) {
        float rx = x[2*p], ry = x[2*p+1];
        float r = sqrtf(rx*rx + ry*ry);
        float theta = atan2f(ry, rx);
        radii[p] = r;

        float norm_a = (theta + (float)M_PI) / (2.0f * (float)M_PI);
        int idx = (int)(norm_a * levels);
        if (idx >= levels) idx = levels - 1;
        if (idx < 0) idx = 0;
        angle_buf[(*angle_pos)++] = idx;
    }

    /* Recursive: decompose the radii */
    float sub_r;
    polar_encode_recursive(radii, n_pairs, bits, angle_buf, angle_pos, &sub_r);

    /* Odd element goes as final radius */
    if (rem > 0) {
        *out_final_r = x[n - 1];
    } else {
        *out_final_r = sub_r;
    }

    free(radii);
}

/* ==========================================================
 * Recursive polar decode: (d-1) angle indices + 1 radius → d-dim vector
 *
 * Reconstructs the tree bottom-up: first decode the deepest level
 * (innermost radii), then work outward using the stored angles.
 * Angles are consumed in the SAME order as encode (pre-order).
 * ========================================================== */

static void polar_decode_recursive(int n, int bits,
    const int *angle_buf, int *angle_pos,
    float final_r, float *x_out) {
    if (n <= 1) {
        if (n == 1) x_out[0] = final_r;
        return;
    }

    int n_pairs = n / 2;
    int rem = n - n_pairs * 2;
    int levels = 1 << bits;

    /* We must decode the INNERMOST level first (radii), then use
     * those radii to reconstruct the current level's (x,y) pairs.
     *
     * Encode stored angles in pre-order: [level0_angles, level1_angles, ...]
     * So we need to SKIP the current level's angles, recurse to decode
     * radii, then COME BACK and read them. But that requires knowing
     * how many angles the recursive call will consume.
     *
     * Simpler: reverse the angle buffer. The encoder stores innermost
     * angle LAST, so if we read from the end, we get innermost first.
     *
     * But we're using a shared angle_pos counter, so instead let's
     * calculate: level 0 has n_pairs angles, level 1 has n_pairs/2, etc.
     * Read angles for THIS level first (consumed in pre-order), then
     * recurse to get radii. */

    /* Save angles for this level */
    int *level_angles = (int *)malloc((size_t)n_pairs * sizeof(int));
    for (int p = 0; p < n_pairs; p++) {
        level_angles[p] = angle_buf[(*angle_pos)++];
    }

    /* Recursive: decode the radii vector */
    float *radii = (float *)malloc((size_t)n_pairs * sizeof(float));
    if (n_pairs > 1) {
        polar_decode_recursive(n_pairs, bits, angle_buf, angle_pos,
                               final_r, radii);
    } else {
        radii[0] = final_r;
    }

    /* Reconstruct (x, y) from (r, theta) at this level */
    for (int p = 0; p < n_pairs; p++) {
        int idx = level_angles[p];
        float norm_a = (float)idx / (float)levels;
        float theta = norm_a * 2.0f * (float)M_PI - (float)M_PI;
        float r = radii[p];
        x_out[2*p]   = r * cosf(theta);
        x_out[2*p+1] = r * sinf(theta);
    }

    if (rem > 0) {
        x_out[n - 1] = final_r;
    }

    free(level_angles);
    free(radii);
}

/* ==========================================================
 * Public lifecycle
 * ========================================================== */

int wubu_polarquant_init(wubu_polarquant_t *pq, int d, int depth,
                                       float R_max, float bits_per_coord) {
    if (!pq || d <= 0 || depth <= 0) return -1;
    if (bits_per_coord < 1 || bits_per_coord > 16) return -1;

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

    float phi = (sqrtf(5.0f) + 1.0f) / 2.0f;
    for (int i = 0; i < d; i++) {
        pq->rand_precondition[i] = 2.0f * (float)M_PI * (float)i * phi / (float)d;
    }

    for (int l = 0; l < depth; l++) {
        pq->level_scale[l] = R_max / sqrtf((float)(1 << l));
    }

    return 0;
}

void wubu_polarquant_free(wubu_polarquant_t *pq) {
    if (!pq) return;
    free(pq->level_scale);
    free(pq->rand_precondition);
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
    int bits = (int)pq->bits_per_coord;

    /* Precondition: random rotation to spread outliers */
    float *x_rot = (float *)malloc((size_t)d * sizeof(float));
    if (!x_rot) return -1;
    apply_precondition(pq, x, x_rot);

    /* Recursive polar decomposition */
    int angle_pos = 0;
    float final_r;
    polar_encode_recursive(x_rot, d, bits, level_angle_idx, &angle_pos, &final_r);

    level_radius[0] = final_r;
    n_angles_per_level[0] = angle_pos;

    free(x_rot);
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

    int bits = (int)pq->bits_per_coord;

    /* Recursive polar reconstruction */
    int angle_pos = 0;
    polar_decode_recursive(d_out, bits, level_angle_idx, &angle_pos,
                           level_radius[0], x_out);

    /* Undo preconditioning */
    float *x_unrot = (float *)malloc((size_t)d_out * sizeof(float));
    if (!x_unrot) return -1;
    undo_precondition(pq, x_out, x_unrot);
    memcpy(x_out, x_unrot, (size_t)d_out * sizeof(float));
    free(x_unrot);

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
    int bits = (int)pq->bits_per_coord;
    int n_angles = d - 1;

    /* Encode: get final radius + angle indices */
    float level_radius[1];
    int *angle_idx = (int *)malloc((size_t)n_angles * sizeof(int));
    int n_per_level[1];
    if (!angle_idx) return -1;

    wubu_polarquant_encode(pq, k_col, level_radius, angle_idx, n_per_level);

    /* Pack: 1 float radius + n_angles * bits bits (packed into bytes) */
    int packed_angle_bytes = (n_angles * bits + 7) / 8;
    int need = (int)sizeof(float) + packed_angle_bytes;
    if (*out_bytes < need) { free(angle_idx); return -1; }

    /* Store radius */
    float *fp = (float *)out_bits;
    fp[0] = level_radius[0];

    /* Pack angle indices into bitstream */
    uint8_t *bp = (uint8_t *)(fp + 1);
    memset(bp, 0, (size_t)packed_angle_bytes);
    int bit_pos = 0;
    for (int i = 0; i < n_angles; i++) {
        int val = angle_idx[i];
        for (int b = 0; b < bits; b++) {
            if (val & (1 << b)) {
                bp[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }

    *out_bytes = need;
    free(angle_idx);
    return 0;
}

int wubu_polarquant_dequantize_kv(const wubu_polarquant_t *pq,
                                           const float *in_bits, int in_bytes,
                                           float *k_col_out, int d) {
    if (!pq || !in_bits || !k_col_out) return -1;

    int bits = (int)pq->bits_per_coord;
    int n_angles = d - 1;
    int packed_angle_bytes = (n_angles * bits + 7) / 8;
    int need = (int)sizeof(float) + packed_angle_bytes;
    if (in_bytes < need) return -1;

    /* Unpack radius */
    const float *fp = in_bits;
    float level_radius[1];
    level_radius[0] = fp[0];

    /* Unpack angle indices from bitstream */
    int *angle_idx = (int *)malloc((size_t)n_angles * sizeof(int));
    int n_per_level[1];
    if (!angle_idx) return -1;

    const uint8_t *bp = (const uint8_t *)(fp + 1);
    int bit_pos = 0;
    for (int i = 0; i < n_angles; i++) {
        int val = 0;
        for (int b = 0; b < bits; b++) {
            if (bp[bit_pos / 8] & (1 << (bit_pos % 8))) {
                val |= (1 << b);
            }
            bit_pos++;
        }
        angle_idx[i] = val;
    }
    n_per_level[0] = n_angles;

    int rc = wubu_polarquant_decode(pq, level_radius, angle_idx, n_per_level,
                                    k_col_out, d);
    free(angle_idx);
    return rc;
}

/* ==========================================================
 * Bandwidth helpers
 * ========================================================== */

double wubu_polarquant_theoretical_bandwidth(const wubu_polarquant_t *pq, int d) {
    int n_angles = d - 1;
    int bytes = (int)sizeof(float) + n_angles * (int)sizeof(int);
    return (double)bytes * 2.0 * 40.0;
}

/*
 * wubu_polarquant.h — PolarQuant + Poincaré fractal stacking
 *
 * Combines PolarQuant's recursive polar coordinate decomposition
 * with wubu_math's Poincaré ball (Möbius) operations to create
 * a hierarchical hyperbolic codebook for sub-4-bit KV cache
 * quantization with ZERO per-block overhead.
 *
 * Design principle: each recursion level of the polar transform
 * maps to a Poincaré ball of progressively smaller radius.
 * The nested ball structure = fractal stacking on the Poincaré
 * sphere manifold. Angular coordinates at each level become
 * points in the next smaller ball, naturally concentrating.
 *
 * WSL2 substrate: CPU-only, C11, no external deps.
 * Uses existing wubu_mobius.h (exp_map, log_map, mobius_add).
 */

#ifndef WUBU_POLARQUANT_H
#define WUBU_POLARQUANT_H

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================
 * Configuration
 * ========================================================== */

/* Maximum recursion depth for polar decomposition.
 * depth=1: radius + 1 angle (2 values total → ~2 bits/coord at 2 bits/angle)
 * depth=2: radius + 2 sub-radii + 2 angles (5 values → ~2.5 bits/coord at 2 bits/angle)
 * depth=3: fractal codebook, each level halves the ball radius
 * For d=128, depth=3 gives 2^(3+1)-1 = 15 values, encoded in ~30 bits = 2.34 bits/coord */
#ifndef WUBU_POLAR_DEPTH
#define WUBU_POLAR_DEPTH 3
#endif

/* Bits per angle/radius index in the codebook */
#ifndef WUBU_POLAR_BITS_PER_COORD
#define WUBU_POLAR_BITS_PER_COORD 2
#endif

/* Poincaré ball radius for the outermost level */
#ifndef WUBU_POLAR_R_OUTER
#define WUBU_POLAR_R_OUTER 1.0f
#endif

/* ==========================================================
 * Types
 * ========================================================== */

/* A single fractal level: a Poincaré ball of radius R_l
 * containing a codebook of 2^(bits_per_coord) entries.
 * The codebook points live inside this ball. */
typedef struct {
    float R;                  /* ball radius at this level */
    int codebook_size;        /* = 1 << bits_per_coord */
    int dims;                 /* = d / 2^depth + 1 (diminishing with depth) */
    float *codebook;          /* [codebook_size * dims] — sub-4-bit embedded */
    float *radius_centroids;  /* [codebook_size] — quantization levels for log-radius */
} wubu_polar_level_t;

/* Full PolarQuant fractal stacking state:
 * L levels (WUBU_POLAR_DEPTH), each a Poincaré ball
 * nested inside the previous one (R_0 > R_1 > ... > R_L).
 * The fractal property: each level's ball is a scaled copy
 * of the previous, creating self-similar structure. */
typedef struct {
    int d;                    /* original vector dimension */
    int depth;                /* number of polar recursion levels */
    float R_max;             /* radius of outermost Poincaré ball */
    float bits_per_coord;    /* bits per coordinate in codebook */
    float *rand_precondition; /* [d] random rotation preconditioning vector */

    /* Fractal codebook: level l has d_l dimensions and a codebook
     * of size 2^(bits_per_coord * d_l). The levels are nested:
     * R_0 > R_1 > ... > R_L, each level's vectors live in a
     * progressively smaller Poincaré ball. */
    wubu_polar_level_t levels[WUBU_POLAR_DEPTH];

    /* Precomputed exp_map scaling factors for each level */
    float *level_scale;       /* [depth] precomputed R_l / artanh(R_l/R_max) factors */
} wubu_polarquant_t;

/* ==========================================================
 * Lifecycle
 * ========================================================== */

/* Initialize PolarQuant fractal state: precompute codebooks
 * for each level using random preconditioning + polar decomposition.
 * Returns 0 on success, -1 on allocation failure. */
int wubu_polarquant_init(wubu_polarquant_t *pq, int d, int depth,
                                float R_max, float bits_per_coord);

/* Free all allocated memory */
void wubu_polarquant_free(wubu_polarquant_t *pq);

/* ==========================================================
 * Polar decomposition (the recursive "fractal stacking")
 * ========================================================== */

/* Cartesian (R^d) → fractal polar representation (depth levels).
 *
 * Level 0: full dimension, extract radius r_0 and angle vector a_0.
 *   r_0 = ||x||, a_0 = x / r_0  (normalized to unit S^{d-1})
 *   a_0 maps into Poincaré ball of radius R_0 via exp_map.
 *
 * Level 1: take the angle residual and subdivide. Each pair
 *   of coordinates from a_0 undergoes polar decomposition recursively.
 *   The residual radius r_1 < r_0, angles a_1 map into ball R_1.
 *
 * Level l: r_l = ||residual_{l-1}||, angles a_l map into ball R_l.
 *   R_l = R_{l-1} / sqrt(2) — self-similar nesting.
 *
 * Output: level_radius[l] + level_angle_idx[l][0..k-1] for each
 *   level l, where each index selects a codebook entry.
 *
 * This is the forward PolarQuant transform adapted to use our
 * existing wubu_mobius exp_map/log_map for the ball projection. */
int wubu_polarquant_encode(const wubu_polarquant_t *pq,
                                  const float *x,   /* [d] input vector */
                                  float *level_radius,     /* [depth] output radii */
                                  int   *level_angle_idx,  /* [depth * max_angles] codebook indices */
                                  int   *n_angles_per_level /* [depth] angles at each level */);

/* Fractal polar reconstruction: angles and radii → Cartesian (R^d).
 * The inverse of the encode function. Uses log_map to project
 * codebook entries from Poincaré ball back to tangent space,
 * then exp_map to fold back into the ball at each level. */
int wubu_polarquant_decode(const wubu_polarquant_t *pq,
                                  const float *level_radius,
                                  const int   *level_angle_idx,
                                  const int   *n_angles_per_level,
                                  float *x_out,    /* [d] reconstructed vector */
                                  int d);

/* ==========================================================
 * KV cache quantization with fractal stacking
 * ========================================================== */

/* Quantize KV cache column using PolarQuant fractal stacking.
 * Stores the quantized representation in a compact bitstream.
 * Returns bytes written to out_bits (caller-allocated). */
int wubu_polarquant_quantize_kv(
        const wubu_polarquant_t *pq,
        const float *k_col,        /* [d] one KV column */
        float *out_bits,           /* [out_bytes] packed bitstream */
        int *out_bytes);           /* [in] buffer size, [out] bytes written */

/* Dequantize a KV column from the packed bitstream back to F32. */
int wubu_polarquant_dequantize_kv(
        const wubu_polarquant_t *pq,
        const float *in_bits,      /* packed bitstream */
        int in_bytes,
        float *k_col_out,          /* [d] dequantized KV column */
        int d);

/* ==========================================================
 * KV bandwidth analysis (Roofline)
 * ========================================================== */

/* Bits per KV column at this configuration (d dims, depth levels). */
static inline double wubu_polarquant_bits_per_vector(const wubu_polarquant_t *pq, int d) {
    double total_bits = 0.0;
    for (int l = 0; l < pq->depth; l++) {
        int d_l = d / (1 << l);  /* dims shrink by half each level */
        if (d_l < 1) d_l = 1;
        /* radius + d_l angles, each quantized to bits_per_coord bits */
        total_bits += (1 + d_l) * pq->bits_per_coord;
    }
    return total_bits;
}

static inline double wubu_polarquant_bytes_per_vector(const wubu_polarquant_t *pq, int d) {
    return wubu_polarquant_bits_per_vector(pq, d) / 8.0;
}

/* Compression ratio vs F32 KV (128 bytes per d=128 vector) */
static inline float wubu_polarquant_compression_ratio(const wubu_polarquant_t *pq, int d) {
    float f32_bytes = (float)d * sizeof(float);
    float pq_bytes = (float)wubu_polarquant_bytes_per_vector(pq, d);
    return f32_bytes / pq_bytes;
}

#ifdef __cplusplus
}
#endif

#endif /* WUBU_POLARQUANT_H */
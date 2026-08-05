/*
 * wubu_gaad.h  --  WuBuOS Golden Aspect Adaptive Decomposition
 *
 * Cell 393: GAAD -- the universal resolution translator.
 *
 * From bytropix THEORY/papers/GAAD-WuBu-ST1.md:
 *   "GAAD provides a multi-scale, aspect-ratio agnostic method
 *    for decomposing frames into geometrically significant
 *    regions based on φ."
 *
 * Recursive Golden Subdivision turns ANY rectangle into squares +
 * golden rectangles. This gives a resolution-independent coordinate
 * system.
 *
 * φ = (1 + √5) / 2 ≈ 1.6180339887
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3 (ported from WuBuOS)
 */
#ifndef WUBU_GAAD_H
#define WUBU_GAAD_H

#include <stdint.h>
#include <stdbool.h>

/* -- Golden Ratio Constants -- */
#define WUBU_PHI       1.6180339887498948482   /* φ = (1+√5)/2 */
#define WUBU_PHI_INV   0.6180339887498948482   /* 1/φ = φ-1 */
#define WUBU_PHI_SQ    2.6180339887498948482   /* φ² = φ+1 */

/* -- GAAD Region -- */
#define WUBU_GAAD_MAX_REGIONS  64
#define WUBU_GAAD_MAX_DEPTH    6

typedef enum {
    WUBU_GAAD_SQUARE    = 0,   /* Square region from golden subdivision */
    WUBU_GAAD_GOLDEN_W  = 1,   /* Golden rect wider than tall (φ:1) */
    WUBU_GAAD_GOLDEN_H  = 2,   /* Golden rect taller than wide (1:φ) */
    WUBU_GAAD_SPIRAL_PT = 3,   /* Φ-spiral sector center point */
} WubuGaadKind;

typedef struct {
    int      x, y, w, h;       /* Region coordinates in parent space */
    int      depth;            /* Subdivision depth (0 = full frame) */
    int      index;            /* Unique region index */
    WubuGaadKind kind;         /* Square, golden rect, spiral point */
    double   phi_scale;        /* φ^n scale factor for this region */
    int      cardinal;         /* 0=N, 1=E, 2=S, 3=W, -1=center */
    bool     is_snap_target;   /* True = window can snap here */
} WubuGaadRegion;

/* -- GAAD Decomposition -- */
typedef struct {
    WubuGaadRegion regions[WUBU_GAAD_MAX_REGIONS];
    int            n_regions;
    int            frame_w, frame_h;
    int            max_depth;
    bool           with_spirals;
    bool           with_cardinals;
} WubuGaadDecomp;

/* -- Feng Shui Cardinal Mirrors -- */
typedef struct {
    WubuGaadRegion north[4];   /* Top regions (φ², φ, 1 vertical) */
    WubuGaadRegion east[4];    /* Right regions (1, φ, φ² horizontal) */
    WubuGaadRegion south[4];   /* Bottom regions (1, φ, φ² vertical) */
    WubuGaadRegion west[4];    /* Left regions (φ², φ, 1 horizontal) */
    WubuGaadRegion center;     /* Golden center region */
} WubuFengShui;

/* -- Resolution Translation -- */
typedef struct {
    int src_w, src_h;          /* Source resolution (e.g., 640×480) */
    int dst_w, dst_h;          /* Target resolution (e.g., 1920×1080) */
    WubuGaadDecomp src_decomp; /* GAAD decomposition of source */
    WubuGaadDecomp dst_decomp; /* GAAD decomposition of target */
} WubuGaadTranslate;

/* ==================================================================
 *  API: Golden Subdivision
 * ================================================================== */

/* Decompose a rectangle into GAAD regions via Recursive Golden Subdivision. */
void wubu_gaad_decompose(int width, int height, int max_depth,
                          WubuGaadDecomp *out);

/* Decompose with feng shui cardinal mirrors. */
void wubu_gaad_decompose_feng_shui(int width, int height, int max_depth,
                                    WubuGaadDecomp *out,
                                    WubuFengShui *fs);

/* Find the nearest GAAD snap region for a window position. */
int wubu_gaad_find_snap(const WubuGaadDecomp *decomp,
                         int win_x, int win_y, int win_w, int win_h,
                         int snap_threshold);

/* Get snap position for a region index. */
void wubu_gaad_snap_pos(const WubuGaadDecomp *decomp, int region_idx,
                          int *out_x, int *out_y, int *out_w, int *out_h);

/* ==================================================================
 *  API: Phi-Spiral Sectoring
 * ================================================================== */

/* Add phi-spiral sector points to an existing decomposition. */
void wubu_gaad_add_spirals(WubuGaadDecomp *decomp,
                            int num_arms, int points_per_arm);

/* ==================================================================
 *  API: Resolution Translation
 * ================================================================== */

/* Create a resolution translator. */
void wubu_gaad_translate_init(int src_w, int src_h,
                               int dst_w, int dst_h,
                               int max_depth,
                               WubuGaadTranslate *out);

/* Translate a pixel coordinate from source → target resolution. */
void wubu_gaad_translate_pixel(const WubuGaadTranslate *t,
                                int src_x, int src_y,
                                int *dst_x, int *dst_y);

/* Translate a pixel coordinate from target → source (inverse). */
void wubu_gaad_translate_inverse(const WubuGaadTranslate *t,
                                  int dst_x, int dst_y,
                                  int *src_x, int *src_y);

/* Translate an entire rectangle (for blitting/rendering). */
void wubu_gaad_translate_rect(const WubuGaadTranslate *t,
                               int src_x, int src_y, int src_w, int src_h,
                               int *dst_x, int *dst_y,
                               int *dst_w, int *dst_h);

/* Get the scale factor for a given GAAD region. */
double wubu_gaad_region_scale(const WubuGaadTranslate *t, int region_idx);

/* ==================================================================
 *  API: Feng Shui Snap Layout
 * ================================================================== */

/* Build the feng shui cardinal mirror snap layout. */
void wubu_gaad_feng_shui_build(int frame_w, int frame_h,
                                WubuFengShui *fs);

/* Find the nearest feng shui snap position for a window. */
bool wubu_gaad_feng_shui_snap(const WubuFengShui *fs,
                               int win_x, int win_y, int win_w, int win_h,
                               int snap_threshold,
                               int *out_x, int *out_y,
                               int *out_w, int *out_h);

/* ==================================================================
 *  API: Pure C Math (no libm)
 * ================================================================== */

int wubu_isqrt(int n);
int wubu_dist(int x1, int y1, int x2, int y2);
double wubu_phi_pow(int n);
int wubu_clamp(int val, int lo, int hi);

#endif /* WUBU_GAAD_H */

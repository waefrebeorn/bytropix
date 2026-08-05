/*
 * wubu_gaad.c  --  WuBuOS Golden Aspect Adaptive Decomposition
 *
 * Cell 393: GAAD -- the universal resolution translator.
 *
 * Port of WuBuOS/src/kernel/wubu_gaad.c. Made self-contained for Windows:
 *   - Uses <math.h> directly (libm available on Windows/MSYS2)
 *   - No bare-metal, no VBE, no kernel dependencies
 *   - C11, opaque-friendly structs, minimal includes
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#include "wubu_gaad.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define WUBU_M_PI 3.14159265358979323846

/* -- Pure C Math Helpers -- */

int wubu_isqrt(int n) {
    if (n <= 0) return 0;
    if (n < 2) return 1;
    int x = n;
    int y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

int wubu_dist(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    return wubu_isqrt(dx * dx + dy * dy);
}

double wubu_phi_pow(int n) {
    if (n == 0) return 1.0;
    if (n == 1) return WUBU_PHI;
    if (n > 0) {
        double a = 1.0, b = WUBU_PHI;
        for (int i = 2; i <= n; i++) {
            double c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
    /* Negative: φ^(-n) = (1/φ)^n */
    double a = 1.0, b = WUBU_PHI_INV;
    for (int i = -2; i >= n; i--) {
        double c = a * WUBU_PHI_INV;
        a = b;
        b = c;
    }
    return b;
}

int wubu_clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/* -- Recursive Golden Subdivision -- */

static void golden_subdivide(int x, int y, int w, int h,
                              int depth, int max_depth,
                              int cardinal_hint,
                              WubuGaadDecomp *out) {
    if (depth >= max_depth || out->n_regions >= WUBU_GAAD_MAX_REGIONS)
        return;
    if (w < 4 || h < 4)
        return;

    WubuGaadRegion *r = &out->regions[out->n_regions];
    r->x = x;
    r->y = y;
    r->w = w;
    r->h = h;
    r->depth = depth;
    r->index = out->n_regions;
    r->cardinal = cardinal_hint;
    r->is_snap_target = (depth <= 2);

    if (w == h) {
        r->kind = WUBU_GAAD_SQUARE;
        r->phi_scale = wubu_phi_pow(-depth);
        out->n_regions++;
        return;
    }

    if (w > h) {
        /* Landscape: cut off a square from the left */
        int sq = h;
        r->kind = WUBU_GAAD_SQUARE;
        r->w = sq;
        r->phi_scale = wubu_phi_pow(-depth);
        out->n_regions++;

        /* Remaining golden rectangle on the right */
        int rem_w = w - sq;
        if (rem_w > 0) {
            int hint = (cardinal_hint >= 0)
                ? cardinal_hint
                : ((x + sq > out->frame_w / 2) ? 1 : 3);
            golden_subdivide(x + sq, y, rem_w, h,
                             depth + 1, max_depth, hint, out);
        }
    } else {
        /* Portrait: cut off a square from the top */
        int sq = w;
        r->kind = WUBU_GAAD_SQUARE;
        r->h = sq;
        r->phi_scale = wubu_phi_pow(-depth);
        out->n_regions++;

        /* Remaining golden rectangle below */
        int rem_h = h - sq;
        if (rem_h > 0) {
            int hint = (cardinal_hint >= 0)
                ? cardinal_hint
                : ((y + sq > out->frame_h / 2) ? 2 : 0);
            golden_subdivide(x, y + sq, w, rem_h,
                             depth + 1, max_depth, hint, out);
        }
    }
}

void wubu_gaad_decompose(int width, int height, int max_depth,
                          WubuGaadDecomp *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->frame_w = width;
    out->frame_h = height;
    out->max_depth = max_depth > 0 ? max_depth : WUBU_GAAD_MAX_DEPTH;
    out->with_spirals = false;
    out->with_cardinals = false;

    golden_subdivide(0, 0, width, height, 0, out->max_depth, -1, out);
}

/* -- Feng Shui Cardinal Mirrors -- */

void wubu_gaad_feng_shui_build(int frame_w, int frame_h, WubuFengShui *fs) {
    if (!fs) return;
    memset(fs, 0, sizeof(*fs));

    double total_v = WUBU_PHI_SQ + WUBU_PHI + 1.0;

    /* Vertical column widths: φ² : φ : 1 */
    int col1_w = (int)(frame_w * WUBU_PHI_SQ / total_v);
    int col2_w = (int)(frame_w * WUBU_PHI / total_v);
    int col3_w = frame_w - col1_w - col2_w;

    /* Horizontal row heights: 1 : φ : φ² */
    int row1_h = (int)(frame_h * 1.0 / total_v);
    int row2_h = (int)(frame_h * WUBU_PHI / total_v);
    int row3_h = frame_h - row1_h - row2_h;

    /* North: top 3 columns (commanding = left heavy: φ²,φ,1) */
    for (int i = 0; i < 3; i++) {
        int cx = (i == 0) ? 0 : (i == 1) ? col1_w : col1_w + col2_w;
        int cw = (i == 0) ? col1_w : (i == 1) ? col2_w : col3_w;
        fs->north[i] = (WubuGaadRegion){
            .x = cx, .y = 0, .w = cw, .h = row1_h,
            .depth = 1, .index = i, .kind = WUBU_GAAD_GOLDEN_W,
            .phi_scale = wubu_phi_pow(-1), .cardinal = 0,
            .is_snap_target = true
        };
    }

    /* South: bottom 3 columns (receptive = right heavy: 1,φ,φ² mirrored) */
    int south_y = row1_h + row2_h;
    for (int i = 0; i < 3; i++) {
        int cx = (i == 0) ? 0 : (i == 1) ? col1_w : col1_w + col2_w;
        int cw = (i == 0) ? col3_w : (i == 1) ? col2_w : col1_w;
        fs->south[i] = (WubuGaadRegion){
            .x = cx, .y = south_y, .w = cw, .h = row3_h,
            .depth = 1, .index = 3 + i, .kind = WUBU_GAAD_GOLDEN_W,
            .phi_scale = wubu_phi_pow(-1), .cardinal = 2,
            .is_snap_target = true
        };
    }

    /* West: left 3 rows (commanding = top heavy: φ²,φ,1) */
    for (int i = 0; i < 3; i++) {
        int cy = (i == 0) ? 0 : (i == 1) ? row1_h : row1_h + row2_h;
        int ch = (i == 0) ? row3_h : (i == 1) ? row2_h : row1_h;
        fs->west[i] = (WubuGaadRegion){
            .x = 0, .y = cy, .w = col1_w, .h = ch,
            .depth = 1, .index = 6 + i, .kind = WUBU_GAAD_GOLDEN_H,
            .phi_scale = wubu_phi_pow(-1), .cardinal = 3,
            .is_snap_target = true
        };
    }

    /* East: right 3 rows (receptive = bottom heavy: 1,φ,φ²) */
    int east_x = col1_w + col2_w;
    for (int i = 0; i < 3; i++) {
        int cy = (i == 0) ? 0 : (i == 1) ? row1_h : row1_h + row2_h;
        int ch = (i == 0) ? row1_h : (i == 1) ? row2_h : row3_h;
        fs->east[i] = (WubuGaadRegion){
            .x = east_x, .y = cy, .w = col3_w, .h = ch,
            .depth = 1, .index = 9 + i, .kind = WUBU_GAAD_GOLDEN_H,
            .phi_scale = wubu_phi_pow(-1), .cardinal = 1,
            .is_snap_target = true
        };
    }

    /* Center: golden rectangle at the intersection */
    fs->center = (WubuGaadRegion){
        .x = col1_w, .y = row1_h,
        .w = col2_w, .h = row2_h,
        .depth = 1, .index = 12, .kind = WUBU_GAAD_SQUARE,
        .phi_scale = wubu_phi_pow(-1), .cardinal = -1,
        .is_snap_target = true
    };
}

void wubu_gaad_decompose_feng_shui(int width, int height, int max_depth,
                                    WubuGaadDecomp *out,
                                    WubuFengShui *fs) {
    wubu_gaad_decompose(width, height, max_depth, out);
    out->with_cardinals = true;

    for (int i = 0; i < out->n_regions; i++) {
        WubuGaadRegion *r = &out->regions[i];
        int cx = r->x + r->w / 2;
        int cy = r->y + r->h / 2;

        if (cy < height / 3)       r->cardinal = 0;   /* N */
        else if (cy > 2*height/3)  r->cardinal = 2;   /* S */
        else if (cx < width / 3)   r->cardinal = 3;   /* W */
        else if (cx > 2*width/3)   r->cardinal = 1;   /* E */
        else                       r->cardinal = -1;  /* center */
    }

    if (fs) wubu_gaad_feng_shui_build(width, height, fs);
}

/* -- Find Nearest Snap -- */

int wubu_gaad_find_snap(const WubuGaadDecomp *decomp,
                         int win_x, int win_y, int win_w, int win_h,
                         int snap_threshold) {
    if (!decomp) return -1;

    int best_idx = -1;
    int best_dist = snap_threshold + 1;

    int wcx = win_x + win_w / 2;
    int wcy = win_y + win_h / 2;

    for (int i = 0; i < decomp->n_regions; i++) {
        const WubuGaadRegion *r = &decomp->regions[i];
        if (!r->is_snap_target) continue;
        int rcx = r->x + r->w / 2;
        int rcy = r->y + r->h / 2;
        int d = wubu_dist(wcx, wcy, rcx, rcy);
        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }

    return (best_dist <= snap_threshold) ? best_idx : -1;
}

void wubu_gaad_snap_pos(const WubuGaadDecomp *decomp, int region_idx,
                          int *out_x, int *out_y, int *out_w, int *out_h) {
    if (!decomp || region_idx < 0 || region_idx >= decomp->n_regions) return;
    const WubuGaadRegion *r = &decomp->regions[region_idx];
    if (out_x) *out_x = r->x;
    if (out_y) *out_y = r->y;
    if (out_w) *out_w = r->w;
    if (out_h) *out_h = r->h;
}

/* -- Feng Shui Snap -- */

bool wubu_gaad_feng_shui_snap(const WubuFengShui *fs,
                               int win_x, int win_y, int win_w, int win_h,
                               int snap_threshold,
                               int *out_x, int *out_y,
                               int *out_w, int *out_h) {
    if (!fs) return false;

    int wcx = win_x + win_w / 2;
    int wcy = win_y + win_h / 2;
    int best_dist = snap_threshold + 1;
    const WubuGaadRegion *best = NULL;

    for (int dir = 0; dir < 4; dir++) {
        const WubuGaadRegion *group;
        switch (dir) {
            case 0: group = fs->north; break;
            case 1: group = fs->east;  break;
            case 2: group = fs->south; break;
            default: group = fs->west; break;
        }
        for (int i = 0; i < 4; i++) {
            int rcx = group[i].x + group[i].w / 2;
            int rcy = group[i].y + group[i].h / 2;
            int d = wubu_dist(wcx, wcy, rcx, rcy);
            if (d < best_dist) {
                best_dist = d;
                best = &group[i];
            }
        }
    }

    /* Check center */
    {
        int rcx = fs->center.x + fs->center.w / 2;
        int rcy = fs->center.y + fs->center.h / 2;
        int d = wubu_dist(wcx, wcy, rcx, rcy);
        if (d < best_dist) {
            best_dist = d;
            best = &fs->center;
        }
    }

    if (best && best_dist <= snap_threshold) {
        if (out_x) *out_x = best->x;
        if (out_y) *out_y = best->y;
        if (out_w) *out_w = best->w;
        if (out_h) *out_h = best->h;
        return true;
    }
    return false;
}

/* -- Phi-Spiral Sectoring -- */

void wubu_gaad_add_spirals(WubuGaadDecomp *decomp,
                            int num_arms, int points_per_arm) {
    if (!decomp) return;
    decomp->with_spirals = true;

    double cx = decomp->frame_w / 2.0;
    double cy = decomp->frame_h / 2.0;
    double min_dim = decomp->frame_w < decomp->frame_h
                     ? (double)decomp->frame_w : (double)decomp->frame_h;
    double initial_r = min_dim * 0.05;
    double max_r = min_dim * 0.45;

    /* b = ln(φ) / (π/2) -- ensures φ growth per 90° */
    double b = log(WUBU_PHI) / (WUBU_M_PI / 2.0);

    for (int arm = 0; arm < num_arms; arm++) {
        double angle_offset = (2.0 * WUBU_M_PI / num_arms) * arm;
        double theta_max = 4.0 * WUBU_M_PI;  /* 2 full revolutions */

        if (initial_r > 0 && max_r > initial_r && b > 1e-6) {
            double tmax = log(max_r / initial_r) / b;
            if (tmax < theta_max) theta_max = tmax;
        }

        for (int pt = 0; pt < points_per_arm; pt++) {
            if (decomp->n_regions >= WUBU_GAAD_MAX_REGIONS) return;

            double theta = theta_max * pt / (points_per_arm - 1);
            double r = initial_r * exp(b * theta);
            if (r > max_r) break;

            double angle = angle_offset + theta;
            int px = (int)(cx + r * cos(angle));
            int py = (int)(cy + r * sin(angle));

            if (px < 0 || px >= decomp->frame_w ||
                py < 0 || py >= decomp->frame_h) continue;

            WubuGaadRegion *reg = &decomp->regions[decomp->n_regions];
            reg->x = px;
            reg->y = py;
            reg->w = 1;
            reg->h = 1;
            reg->depth = pt;
            reg->index = decomp->n_regions;
            reg->kind = WUBU_GAAD_SPIRAL_PT;
            reg->phi_scale = r / min_dim;
            reg->cardinal = -1;
            reg->is_snap_target = false;
            decomp->n_regions++;
        }
    }
}

/* -- Resolution Translation -- */

void wubu_gaad_translate_init(int src_w, int src_h,
                               int dst_w, int dst_h,
                               int max_depth,
                               WubuGaadTranslate *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->src_w = src_w;
    out->src_h = src_h;
    out->dst_w = dst_w;
    out->dst_h = dst_h;

    wubu_gaad_decompose(src_w, src_h, max_depth, &out->src_decomp);
    wubu_gaad_decompose(dst_w, dst_h, max_depth, &out->dst_decomp);
}

void wubu_gaad_translate_pixel(const WubuGaadTranslate *t,
                                int src_x, int src_y,
                                int *dst_x, int *dst_y) {
    if (!t || !dst_x || !dst_y) return;

    double u = (double)src_x / t->src_w;
    double v = (double)src_y / t->src_h;

    if (u < 0.0) u = 0.0;
    if (u >= 1.0) u = 1.0 - 1e-9;
    if (v < 0.0) v = 0.0;
    if (v >= 1.0) v = 1.0 - 1e-9;

    /* Find source GAAD region */
    int src_region = -1;
    for (int i = 0; i < t->src_decomp.n_regions; i++) {
        const WubuGaadRegion *r = &t->src_decomp.regions[i];
        if (src_x >= r->x && src_x < r->x + r->w &&
            src_y >= r->y && src_y < r->y + r->h) {
            src_region = i;
            break;
        }
    }

    /* Compute local (u,v) within source region */
    double local_u = u, local_v = v;
    if (src_region >= 0) {
        const WubuGaadRegion *sr = &t->src_decomp.regions[src_region];
        if (sr->w > 0) local_u = (double)(src_x - sr->x) / sr->w;
        if (sr->h > 0) local_v = (double)(src_y - sr->y) / sr->h;
    }

    /* Map to corresponding target GAAD region */
    int dst_region = -1;
    if (src_region >= 0 && src_region < t->dst_decomp.n_regions) {
        dst_region = src_region;
    } else {
        for (int i = 0; i < t->dst_decomp.n_regions; i++) {
            const WubuGaadRegion *r = &t->dst_decomp.regions[i];
            double ru = (double)(r->x + r->w/2) / t->dst_w;
            double rv = (double)(r->y + r->h/2) / t->dst_h;
            if (fabs(ru - u) < 0.5 && fabs(rv - v) < 0.5) {
                dst_region = i;
                break;
            }
        }
    }

    /* Convert local (u,v) to target pixel */
    if (dst_region >= 0 && dst_region < t->dst_decomp.n_regions) {
        const WubuGaadRegion *dr = &t->dst_decomp.regions[dst_region];
        *dst_x = dr->x + (int)(local_u * dr->w);
        *dst_y = dr->y + (int)(local_v * dr->h);
    } else {
        *dst_x = (int)(u * t->dst_w);
        *dst_y = (int)(v * t->dst_h);
    }
}

void wubu_gaad_translate_inverse(const WubuGaadTranslate *t,
                                  int dst_x, int dst_y,
                                  int *src_x, int *src_y) {
    if (!t || !src_x || !src_y) return;

    double u = (double)dst_x / t->dst_w;
    double v = (double)dst_y / t->dst_h;

    if (u < 0.0) u = 0.0;
    if (u >= 1.0) u = 1.0 - 1e-9;
    if (v < 0.0) v = 0.0;
    if (v >= 1.0) v = 1.0 - 1e-9;

    *src_x = (int)(u * t->src_w);
    *src_y = (int)(v * t->src_h);
}

void wubu_gaad_translate_rect(const WubuGaadTranslate *t,
                               int src_x, int src_y, int src_w, int src_h,
                               int *dst_x, int *dst_y,
                               int *dst_w, int *dst_h) {
    int tl_x, tl_y, br_x, br_y;
    wubu_gaad_translate_pixel(t, src_x, src_y, &tl_x, &tl_y);
    wubu_gaad_translate_pixel(t, src_x + src_w - 1, src_y + src_h - 1, &br_x, &br_y);
    if (dst_x) *dst_x = tl_x;
    if (dst_y) *dst_y = tl_y;
    if (dst_w) *dst_w = br_x - tl_x + 1;
    if (dst_h) *dst_h = br_y - tl_y + 1;
}

double wubu_gaad_region_scale(const WubuGaadTranslate *t, int region_idx) {
    if (!t || region_idx < 0 || region_idx >= t->dst_decomp.n_regions)
        return 1.0;
    if (region_idx >= t->src_decomp.n_regions)
        return 1.0;

    const WubuGaadRegion *sr = &t->src_decomp.regions[region_idx];
    const WubuGaadRegion *dr = &t->dst_decomp.regions[region_idx];

    double src_area = (double)sr->w * sr->h;
    double dst_area = (double)dr->w * dr->h;

    if (src_area > 0.0) return dst_area / src_area;
    return 1.0;
}

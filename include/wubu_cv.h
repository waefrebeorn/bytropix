/*
 * wubu_cv.h -- core image + imgproc for the eyes (solidcv reimplementation).
 *
 * Scope note: this is OUR C11 implementation of the classical pieces solidcv
 * advertises (core / imgproc), not a binding. Integer arithmetic where the
 * operation allows it; float only where a mean or variance is genuinely needed.
 */
#ifndef WUBU_CV_H
#define WUBU_CV_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int w, h, stride;
    uint8_t *data;
} wubu_img;

int  wubu_img_alloc(wubu_img *im, int w, int h);
void wubu_img_free(wubu_img *im);
int  wubu_img_clone(const wubu_img *src, wubu_img *dst);
void wubu_img_fill(wubu_img *im, uint8_t v);
uint8_t wubu_img_at(const wubu_img *im, int x, int y);

/* (w+1)*(h+1) integral images. Caller frees. */
int wubu_integral_u64(const wubu_img *im, uint64_t **sum, uint64_t **sumsq);

/* O(1) box mean/variance from integral images. */
int wubu_box_stats(const uint64_t *sum, const uint64_t *sumsq, int w, int h,
                   int x, int y, int bw, int bh, double *mean, double *var);

/* Fused 3x3 Sobel. gx/gy are w*h int16 buffers (caller-owned). */
int wubu_sobel3(const wubu_img *im, int16_t *gx, int16_t *gy);

/* Otsu threshold over the 256-bin histogram. Returns 0..255, or -1. */
int wubu_otsu(const wubu_img *im);

/* Separable 3x3 Gaussian (1 2 1)/4, edge-clamped. */
int wubu_gauss3(const wubu_img *in, wubu_img *out);

/* --- Canny + contours (wave 2) ---------------------------------------- */

/* Gradient magnitude, non-maximum suppression along the gradient direction,
 * then hysteresis: strong >= high seeds, weak >= low connects. Edges are 0/255.
 * Returns 0, or -1 on bad arguments. */
int wubu_canny(const wubu_img *in, int low, int high, wubu_img *edges);

typedef struct {
    int x, y, w, h;   /* bounding box */
    int area;         /* number of edge pixels in the component */
} wubu_contour;

/* 8-connected components of an edge image, skipping ones below min_area.
 * Returns the number written (<= max_contours), or -1. */
int wubu_find_contours(const wubu_img *edges, wubu_contour *out,
                       int max_contours, int min_area);

/* 3x3 morphology. mode 0 = erode (min), 1 = dilate (max). Edge-clamped. */
int wubu_morph3(const wubu_img *in, wubu_img *out, int mode);

/* --- Foveated ROI selection (wave 7) ---------------------------------- */

typedef struct {
    int x, y, w, h;
    double score;    /* summed |gx|+|gy| over the tile */
} wubu_roi;

/* Pick up to k non-overlapping (in tile space, nms_radius apart) regions with
 * the highest edge energy. Tiles with zero energy are never returned.
 * Returns the count, or -1. */
int wubu_foveated_rois(const wubu_img *im, int tile, int k, int nms_radius,
                       wubu_roi *out, int max_rois);

#endif /* WUBU_CV_H */

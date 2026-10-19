/*
 * wubu_flow.h -- pyramidal-free Lucas-Kanade flow for tracked points.
 *
 * The solidcv `video` module advertises pyramidal LK and Farneback. This is the
 * single-scale core: given a previous frame, a current frame and a sparse point
 * set, solve the 2x2 normal equations per point over a window. A pyramid wrapper
 * is the next rung, and this header says it is not here yet.
 */
#ifndef WUBU_FLOW_H
#define WUBU_FLOW_H

#include "wubu_cv.h"
#include "wubu_features.h"

/* dx/dy receive the displacement (current - previous) per keypoint.
 * win = window edge in pixels (odd, >=3), iters = Gauss-Newton passes.
 * Returns 0, or -1 on bad arguments. */
int wubu_lk_flow(const wubu_img *prev, const wubu_img *curr,
                 const wubu_keypoint *kp, int n, int win, int iters,
                 float *dx, float *dy);

/* Coarse-to-fine variant: build `levels` half-resolution images, solve at the
 * coarsest, double the flow, refine at each finer level. Handles motion beyond
 * the single-scale +/-2 px linearisation limit. levels 1..5. */
int wubu_lk_flow_pyr(const wubu_img *prev, const wubu_img *curr,
                     const wubu_keypoint *kp, int n, int levels,
                     int win, int iters, float *dx, float *dy);

/* Dense block-matching flow: for every `block`-sized cell, search +/-search for
 * the minimum SAD displacement. Writes integer dx/dy per block (grid of
 * ceil(w/block) x ceil(h/block)). This is block matching, not Farneback.
 * Returns the number of blocks written, or -1. */
int wubu_block_flow(const wubu_img *prev, const wubu_img *curr,
                    int block, int search, int16_t *dx, int16_t *dy);

#endif /* WUBU_FLOW_H */
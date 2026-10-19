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

#endif /* WUBU_FLOW_H */
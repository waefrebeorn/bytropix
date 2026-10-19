/*
 * wubu_look.h -- the eyes' single entry point.
 *
 * Given two consecutive frames, decide what the model must look at: the ROIs
 * whose content changed, ranked by edge energy, plus the share of the frame
 * that can be reused untouched. This composes wubu_eye (patch differencing)
 * and wubu_cv's foveated selector so callers do not have to.
 */
#ifndef WUBU_LOOK_H
#define WUBU_LOOK_H

#include "wubu_cv.h"

typedef struct {
    int x, y, w, h;
    double energy;      /* summed |delta| inside the region */
} wubu_look_roi;

/* prev may be NULL for a first frame (everything is new).
 * Returns the number of ROIs (0 okay), or -1. `reuse_ratio` may be NULL. */
int wubu_look(const wubu_img *prev, const wubu_img *curr,
              int patch, int tile, int max_rois,
              wubu_look_roi *out, float *reuse_ratio);

#endif /* WUBU_LOOK_H */

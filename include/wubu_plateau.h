/* wubu_plateau.h -- the amoeba growth trigger: the sliding-window loss
 * trend detector. The growth operator fires when the recent loss
 * improvements fall below a threshold for a sustained window -- the
 * plateau, not a fixed clock (the Bu schedule is the fallback). */
#ifndef WUBU_PLATEAU_H
#define WUBU_PLATEAU_H

/* Detect a plateau in the loss series.
 * losses [n]: the training-loss history (most recent last).
 * window: the trailing window over which the trend is measured.
 * min_improve: the minimum per-step improvement (negative slope) that
 *   still counts as progress; a slope shallower than -min_improve over
 *   the full window = plateau.
 * Returns 1 on a plateau, 0 on progress (or when n < window). */
int wubu_plateau_detect(const float *losses, int n, int window,
                        float min_improve);

/* The trend slope (per-step loss change) over the trailing window,
 * linear least squares. Negative = improving. */
float wubu_plateau_slope(const float *losses, int n, int window);

#endif

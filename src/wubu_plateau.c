/* wubu_plateau.c -- the amoeba plateau trigger. */
#include "wubu_plateau.h"

float wubu_plateau_slope(const float *losses, int n, int window)
{
    if (!losses || n < 2 || window < 2) return 0;
    if (window > n) window = n;
    const float *w = losses + (n - window);
    /* linear least squares over the window: slope = cov(x,y)/var(x) */
    double mx = 0, my = 0;
    for (int i = 0; i < window; i++) { mx += i; my += w[i]; }
    mx /= window; my /= window;
    double num = 0, den = 0;
    for (int i = 0; i < window; i++) {
        double dx = i - mx, dy = w[i] - my;
        num += dx * dy;
        den += dx * dx;
    }
    return (float)(den > 0 ? num / den : 0);
}

int wubu_plateau_detect(const float *losses, int n, int window,
                        float min_improve)
{
    if (!losses || n < window || window < 2 || min_improve < 0) return 0;
    float s = wubu_plateau_slope(losses, n, window);
    /* a plateau = the improvement (the negative slope) fell below the
     * threshold -- including a RISING slope (the divergence!) */
    return s > -min_improve;
}

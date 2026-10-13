/* wubu_manifoldviz.c -- BRAIN #127 manifold visualization.
 * See wubu_manifoldviz.h. */
#include "wubu_manifoldviz.h"
#include <math.h>
#include <string.h>

int wubu_mv_init(wubu_mv_t *r, double focal) {
    if (!r || focal <= 0) return -1;
    r->focal = focal;
    return 0;
}

int wubu_mv_project(const wubu_mv_t *r, const double *point, wubu_mv_pixel_t *out) {
    if (!r || !point || !out) return -1;
    /* take the first 3 dims as the 3D ball position */
    double x = point[0], y = point[1], z = point[2];
    double norm3 = sqrt(x*x + y*y + z*z);
    /* clamp strictly inside the unit ball */
    const double maxn = 1.0 - 1e-6;
    if (norm3 >= maxn) {
        double s = maxn / (norm3 > 0 ? norm3 : 1.0);
        x *= s; y *= s; z *= s;
        norm3 = maxn;
    }
    /* perspective projection: screen = focal * (x,y) / (focal - z)
     * (z in [-1,1]; points closer to the viewer (+z) project larger) */
    double denom = r->focal - z;
    if (denom < 1e-6) denom = 1e-6;
    out->x = r->focal * x / denom;
    out->y = r->focal * y / denom;
    out->depth = norm3;   /* 0 = center, -> 1 = boundary */
    out->visible = 1;     /* all in-ball points are visible */
    return 0;
}

int wubu_mv_project_all(const wubu_mv_t *r, const double *points, int n,
                        wubu_mv_pixel_t *out) {
    if (!r || !points || !out || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        if (wubu_mv_project(r, points + i * WUBU_MV_DIM, &out[i]) != 0) return -1;
    }
    return n;
}

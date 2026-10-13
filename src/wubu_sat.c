/* wubu_sat.c -- WORLDSIM #268 narrowphase SAT collision.
 * See wubu_sat.h. Separating Axis Theorem for oriented boxes. */
#include "wubu_sat.h"
#include <math.h>
#include <stddef.h>

/* Project an OBB onto an axis (unit vector ax,ay); return [min,max] radius. */
static void project(const wubu_obb_t *o, double ax, double ay, double *min, double *max) {
    double c = cos(o->angle), s = sin(o->angle);
    /* box half-extents projected onto the axis */
    double ex = fabs(o->hw * c * ax + o->hw * (-s) * ay);
    double ey = fabs(o->hh * s * ax + o->hh * c * ay);
    double radius = ex + ey;
    double center = o->cx * ax + o->cy * ay;
    *min = center - radius;
    *max = center + radius;
}

int wubu_sat_collide(const wubu_obb_t *a, const wubu_obb_t *b,
                     double *nx, double *ny, double *depth) {
    if (!a || !b) return 0;
    /* 4 candidate axes: 2 edge normals of each box */
    double axes[4][2];
    double ca = cos(a->angle), sa = sin(a->angle);
    double cb = cos(b->angle), sb = sin(b->angle);
    axes[0][0] = ca;  axes[0][1] = sa;   /* a's x-edge normal */
    axes[1][0] = -sa; axes[1][1] = ca;   /* a's y-edge normal */
    axes[2][0] = cb;  axes[2][1] = sb;   /* b's x-edge normal */
    axes[3][0] = -sb; axes[3][1] = cb;   /* b's y-edge normal */

    double min_overlap = 1e18;
    double best_axis[2] = {0, 0};
    for (int i = 0; i < 4; i++) {
        double ax = axes[i][0], ay = axes[i][1];
        double amin, amax, bmin, bmax;
        project(a, ax, ay, &amin, &amax);
        project(b, ax, ay, &bmin, &bmax);
        /* overlap along this axis */
        double overlap = (amax < bmax ? amax : bmax) - (amin > bmin ? amin : bmin);
        if (overlap < 0) return 0;   /* separating axis found -> no collision */
        if (overlap < min_overlap) {
            min_overlap = overlap;
            best_axis[0] = ax; best_axis[1] = ay;
        }
    }
    /* all axes overlap -> collision; orient normal from a to b */
    if (nx) *nx = best_axis[0];
    if (ny) *ny = best_axis[1];
    if (depth) *depth = min_overlap;
    /* ensure normal points from a to b */
    if (nx && ny) {
        double dx = b->cx - a->cx, dy = b->cy - a->cy;
        if (dx * *nx + dy * *ny < 0) { *nx = -*nx; *ny = -*ny; }
    }
    return 1;
}

int wubu_sat_overlap(const wubu_obb_t *a, const wubu_obb_t *b) {
    return wubu_sat_collide(a, b, NULL, NULL, NULL);
}

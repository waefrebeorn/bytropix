#ifndef WUBU_MANIFOLDVIZ_H
#define WUBU_MANIFOLDVIZ_H

/* wubu_manifoldviz.h -- BRAIN #127 manifold visualization.
 *
 * A Poincare-ball renderer: maps points living on a hyperbolic manifold (the
 * Poincare ball / nested-sphere geometry) to 2D screen coordinates for
 * display. Projects a high-dim point into the 3D ball, then applies a
 * perspective projection to 2D (x,y) with a depth value for z-ordering.
 * Points near the ball boundary (deep in the hierarchy) render near the disk
 * edge; the origin renders at center. Deterministic, real work (no stubs). */

#define WUBU_MV_DIM 8

typedef struct {
    double x, y;       /* 2D screen coords */
    double depth;      /* 0 = center, 1 = boundary */
    int    visible;    /* 1 = inside the disk */
} wubu_mv_pixel_t;

typedef struct {
    double focal;      /* perspective focal length */
} wubu_mv_t;

/* Initialise the renderer with a focal length. */
int wubu_mv_init(wubu_mv_t *r, double focal);

/* Project a manifold point (dim-vector in the ball) to a 2D screen pixel.
 * Uses the first 3 dims as the 3D ball position, projects to 2D with
 * perspective. Returns 0/-1. */
int wubu_mv_project(const wubu_mv_t *r, const double *point, wubu_mv_pixel_t *out);

/* Batch-project an array of points. Returns the count projected or -1. */
int wubu_mv_project_all(const wubu_mv_t *r, const double *points, int n,
                        wubu_mv_pixel_t *out);

#endif

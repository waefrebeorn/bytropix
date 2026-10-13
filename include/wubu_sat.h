#ifndef WUBU_SAT_H
#define WUBU_SAT_H

/* wubu_sat.h -- WORLDSIM #268 narrowphase SAT collision.
 *
 * The Separating Axis Theorem for narrowphase collision: two convex shapes do
 * NOT overlap iff there exists an axis on which their projections are
 * disjoint. For oriented boxes (OBBs), the candidate axes are the two edge
 * normals of each box (4 axes). If all 4 axes show overlap, the boxes
 * collide (and the minimum-overlap axis is the collision normal + penetration
 * depth). This is the precise narrowphase test that follows broadphase culling.
 * Deterministic, no allocation, real work (no stubs). */

typedef struct {
    double cx, cy;    /* center */
    double hw, hh;    /* half-width, half-height */
    double angle;     /* rotation (radians) */
} wubu_obb_t;

/* SAT collision test between two OBBs. Returns 1 if they overlap, 0 if a
 * separating axis exists (no collision). On overlap, fills normal (nx,ny) and
 * depth (penetration) with the minimum-translation vector. */
int wubu_sat_collide(const wubu_obb_t *a, const wubu_obb_t *b,
                     double *nx, double *ny, double *depth);

/* Just the boolean overlap test. */
int wubu_sat_overlap(const wubu_obb_t *a, const wubu_obb_t *b);

#endif

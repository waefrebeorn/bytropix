#ifndef WUBU_BROADPHASE_H
#define WUBU_BROADPHASE_H

/* wubu_broadphase.h -- WORLDSIM #267 broadphase spatial hash grid.
 *
 * Broadphase collision culling: a uniform spatial hash grid buckets bodies by
 * cell so only bodies in the same or neighbouring cells are tested for
 * collision (instead of all O(n^2) pairs). This is the first phase of
 * collision detection -- it cheaply rejects far-apart pairs. Deterministic,
 * fixed-size, real work (no stubs). */

#define WUBU_BP_MAX_BODIES 256
#define WUBU_BP_GRID 32   /* grid is GRID x GRID cells */

typedef struct {
    double x, y, r;   /* center + radius */
    int    id;
} wubu_bp_body_t;

typedef struct {
    wubu_bp_body_t bodies[WUBU_BP_MAX_BODIES];
    int  n;
    double cell;      /* cell size (>= 2*max radius for correctness) */
    double minx, miny;/* grid origin */
} wubu_broadphase_t;

/* Initialise with a cell size and grid origin. */
int wubu_bp_init(wubu_broadphase_t *bp, double cell, double minx, double miny);

/* Add a body. Returns its index or -1. */
int wubu_bp_add(wubu_broadphase_t *bp, double x, double y, double r, int id);

/* Hash a world position to a grid cell index (clamped to grid). */
static inline int wubu_bp_cell_x(const wubu_broadphase_t *bp, double x) {
    int c = (int)((x - bp->minx) / bp->cell);
    if (c < 0) c = 0; if (c >= WUBU_BP_GRID) c = WUBU_BP_GRID - 1;
    return c;
}
static inline int wubu_bp_cell_y(const wubu_broadphase_t *bp, double y) {
    int c = (int)((y - bp->miny) / bp->cell);
    if (c < 0) c = 0; if (c >= WUBU_BP_GRID) c = WUBU_BP_GRID - 1;
    return c;
}

/* Count candidate collision pairs the broadphase would test (bodies whose
 * cells are within 1 cell of each other, accounting for radius). This is the
 * work saved vs O(n^2). Returns the candidate pair count. */
int wubu_bp_candidate_pairs(const wubu_broadphase_t *bp);

#endif

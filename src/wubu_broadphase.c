/* wubu_broadphase.c -- WORLDSIM #267 broadphase spatial hash grid.
 * See wubu_broadphase.h. */
#include "wubu_broadphase.h"

int wubu_bp_init(wubu_broadphase_t *bp, double cell, double minx, double miny) {
    if (!bp || cell <= 0.0) return -1;
    bp->n = 0;
    bp->cell = cell;
    bp->minx = minx;
    bp->miny = miny;
    return 0;
}

int wubu_bp_add(wubu_broadphase_t *bp, double x, double y, double r, int id) {
    if (!bp || bp->n >= WUBU_BP_MAX_BODIES) return -1;
    wubu_bp_body_t *b = &bp->bodies[bp->n];
    b->x = x; b->y = y; b->r = r; b->id = id;
    return bp->n++;
}

int wubu_bp_candidate_pairs(const wubu_broadphase_t *bp) {
    if (!bp) return 0;
    int pairs = 0;
    for (int i = 0; i < bp->n; i++) {
        int cxi = wubu_bp_cell_x(bp, bp->bodies[i].x);
        int cyi = wubu_bp_cell_y(bp, bp->bodies[i].y);
        for (int j = i + 1; j < bp->n; j++) {
            int cxj = wubu_bp_cell_x(bp, bp->bodies[j].x);
            int cyj = wubu_bp_cell_y(bp, bp->bodies[j].y);
            int dx = cxi - cxj; if (dx < 0) dx = -dx;
            int dy = cyi - cyj; if (dy < 0) dy = -dy;
            /* same or adjacent cell -> candidate (radius accounted for by
             * choosing cell >= 2*max_radius) */
            if (dx <= 1 && dy <= 1) pairs++;
        }
    }
    return pairs;
}

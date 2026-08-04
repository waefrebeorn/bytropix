/*
 * wubu_gravity.h -- THE GRAVITY FIELD (research/060, INDEX AN12).
 *
 * The user's directive (2026-08-04): "that's how gravity works on the
 * central mass system — to allow us to organize using the Poincaré
 * sphere polar system." WuBu's weights are a solar system: a dense
 * central mass (the Colonel boot core) surrounded by cells in orbit,
 * organized by gravity F = G·M·m/r² on the Poincaré ball.
 *
 *   radius r  = depth in the hierarchy (near = boot-critical,
 *               far = specialized outer knowledge)
 *   angle θ   = position on the sphere (specialization)
 *   the conformal factor λ = 2/(1-c‖x‖²) IS the field
 *
 * Everything stays in the ball (r < 1/√c). The core (innermost
 * sphere) NEVER grows — the body grows outward around it.
 *
 * C11, self-contained, no third-party.
 */
#ifndef WUBU_GRAVITY_H
#define WUBU_GRAVITY_H

#include <stddef.h>
#include <stdint.h>

/* The gravity cell: one body in orbit around the central mass. */
typedef struct wubu_gravity_cell {
    int    id;          /* cell index (expert/layer/block id) */
    double mass;        /* m > 0 (the cell's own mass) */
    double r;           /* polar radius in the ball, 0 <= r < 1/sqrt(c) */
    double theta;       /* polar angle (position on the sphere) */
    double vr;          /* radial velocity (positive = outward) */
    double vtheta;      /* tangential velocity (the orbit) */
    double acc;         /* accumulated routes (utilization proxy) */
    uint8_t core;       /* 1 = part of the Colonel boot core (protected) */
    uint8_t flags;
} wubu_gravity_cell_t;

typedef struct wubu_gravity wubu_gravity_t;

/* config: the field's physics */
typedef struct {
    double G;           /* gravitational constant (default 1.0) */
    double M;           /* central mass (the Colonel core's mass) */
    double c;           /* Poincaré curvature (r_max = 1/sqrt(c)) */
    size_t n_cells;     /* number of cells */
    double boot_r;      /* the boot-core radius: r < boot_r is protected */
} wubu_gravity_cfg_t;

/* G1: init the field with n cells at a seed radius (all core-protected
 * if within boot_r). Returns NULL on alloc failure. */
wubu_gravity_t *wubu_gravity_init(const wubu_gravity_cfg_t *cfg);

/* G2: free the field. */
void wubu_gravity_free(wubu_gravity_t *g);

/* G3: one orbit step — apply the central-mass force
 *   F = G·M·m/r²  (inward),  stable circular orbit: vθ² = G·M/r
 * Cells with vθ < v_orbit fall inward (spiral in); vθ > v_orbit
 * drift outward. Everything is clamped to the Poincaré ball.
 * Returns 0 on success. */
int wubu_gravity_step(wubu_gravity_t *g);

/* G4: route a token/query (given by a 2-D polar point or a hash) to
 * the cell whose orbit it intersects (nearest by polar distance).
 * Returns the cell id, or -1 on error. */
int wubu_gravity_route(const wubu_gravity_t *g, double r_in, double theta_in);

/* G5: grow — the overworked cell (acc above threshold) splits into
 * two daughters pushed OUTWARD (+dr in radius, ±dtheta in angle).
 * The core (r < boot_r) never splits; it is protected. Returns the
 * new cell id, or -1. */
int wubu_gravity_grow(wubu_gravity_t *g, int cell_id, double dr, double dtheta);

/* G6: shrink — the dead cell (acc below threshold) falls inward and
 * is absorbed into the core (its mass merges, the cell is removed).
 * The core itself is never removed. Returns 0 on success. */
int wubu_gravity_shrink(wubu_gravity_t *g, int cell_id);

/* G7: the cell accessor (read-only). NULL if out of range. */
const wubu_gravity_cell_t *wubu_gravity_cell(const wubu_gravity_t *g,
                                             int cell_id);

/* G8: the live count (cells currently in orbit). */
size_t wubu_gravity_count(const wubu_gravity_t *g);

/* G9: the Colonel-core invariant: the number of core-protected cells
 * (r < boot_r). The core never shrinks below this under grow/shrink. */
size_t wubu_gravity_core_count(const wubu_gravity_t *g);

#endif /* WUBU_GRAVITY_H */

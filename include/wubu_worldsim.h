#ifndef WUBU_WORLDSIM_H
#define WUBU_WORLDSIM_H
#include "wubu_physics.h"
#include "wubu_collision.h"
#include "wubu_fixstep.h"

/* wubu_worldsim.h -- WORLDSIM #257 worldsim is a solid WuBu math dynamic
 * system.
 *
 * The world simulation that COMPOSES the fixed-step clock, the force
 * integrator (gravity + drag), and the rigid-body collision resolver into a
 * single stepping dynamic system. Each world step: advance the fixed-step
 * clock, integrate every body under the forces, then resolve all pairwise
 * collisions with restitution. This is what makes it a "solid dynamic system"
 * -- deterministic (fixed-step), force-driven, and collision-aware. The
 * result is a reproducible physics world. Fail-closed, deterministic, real
 * work (no stubs). */

#define WUBU_WORLD_MAX_BODIES 64

typedef struct {
    wubu_body_t   bodies[WUBU_WORLD_MAX_BODIES];
    int           n_bodies;
    wubu_pforces_t forces;
    double        restitution;
    wubu_fixstep_t clock;
    uint64_t      steps;      /* total world steps taken */
} wubu_world_t;

/* Create a world with n_bodies (<=MAX), forces, restitution, fixed dt. */
int wubu_world_init(wubu_world_t *w, int n_bodies,
                    wubu_pforces_t forces, double restitution, double dt, int max_steps);

/* Add a body (position, velocity, mass, radius). Returns body index or -1. */
int wubu_world_add(wubu_world_t *w, double x, double y, double vx, double vy,
                   double mass, double radius);

/* Advance the world by `elapsed` wall time: runs the fixed number of physics
 * steps, each integrating forces then resolving all collisions. Returns the
 * number of fixed steps run. */
int wubu_world_step(wubu_world_t *w, double elapsed);

/* Total kinetic energy of the world (for conservation/dissipation checks). */
double wubu_world_ke(const wubu_world_t *w);

/* Access a body (read-only). */
const wubu_body_t *wubu_world_body(const wubu_world_t *w, int i);

#endif

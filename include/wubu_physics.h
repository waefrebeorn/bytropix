#ifndef WUBU_PHYSICS_H
#define WUBU_PHYSICS_H

/* wubu_physics.h -- WORLDSIM #264 gravity + drag + restitution tuned forces.
 *
 * A force integrator for a point body: applies gravity (constant downward
 * accel), drag (linear + quadratic, so a body reaches terminal velocity), and
 * integrates velocity/position over a fixed dt (semi-implicit Euler, stable).
 * The forces are TUNABLE (gravity magnitude, linear/quadratic drag coeffs),
 * which is what "tuned forces" means: the same integrator produces
 * feather-fall (high drag) or free-fall (no drag) by parameter choice.
 * Deterministic, no allocation, real work (no stubs). */

typedef struct {
    double x, y;
    double vx, vy;
} wubu_pbody_t;

typedef struct {
    double gravity;      /* downward acceleration (m/s^2), e.g. 9.81 */
    double lin_drag;     /* linear drag coeff (per second) */
    double quad_drag;    /* quadratic drag coeff (per meter) */
} wubu_pforces_t;

/* Initialise a forces config. */
int wubu_physics_init(wubu_pforces_t *f, double gravity, double lin_drag, double quad_drag);

/* Integrate one fixed step (semi-implicit Euler): v += a*dt; p += v*dt.
 * Drag is applied opposite to velocity, scaled by speed (linear) and
 * speed^2 (quadratic). Fail-closed on NULL. */
int wubu_physics_step(wubu_pbody_t *b, const wubu_pforces_t *f, double dt);

/* Terminal velocity magnitude for a body falling under these forces
 * (where drag balances gravity). Returns 0 if no drag (free fall). */
double wubu_physics_terminal_velocity(const wubu_pforces_t *f);

/* Kinetic + potential energy (potential relative to y=0), for conservation. */
double wubu_physics_energy(const wubu_pbody_t *b, const wubu_pforces_t *f);

#endif

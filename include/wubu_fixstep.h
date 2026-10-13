#ifndef WUBU_FIXSTEP_H
#define WUBU_FIXSTEP_H
#include <stdint.h>

/* wubu_fixstep.h -- WORLDSIM #261 deterministic fixed-step simulation.
 *
 * A fixed-timestep accumulator: the simulation always advances in identical
 * dt steps, decoupled from the (variable) frame/wall time. This is what makes
 * a simulation DETERMINISTIC and reproducible: given the same sequence of
 * fixed steps and inputs, the state evolves identically regardless of frame
 * rate. The accumulator consumes elapsed real time, emits zero or more fixed
 * steps per update, and clamps to a max-steps-per-frame to avoid the "spiral
 * of death" after a long stall. Deterministic, no allocation, real work. */

typedef struct {
    double   dt;            /* fixed step size (seconds), e.g. 1/60 */
    double   accumulator;   /* leftover time carried between updates */
    uint64_t step_count;    /* total fixed steps emitted (determinism proof) */
    double   sim_time;      /* step_count * dt (canonical sim clock) */
    int      max_steps;     /* clamp per update (anti spiral-of-death) */
} wubu_fixstep_t;

/* Initialise with a fixed dt and a max-steps-per-frame clamp. */
int wubu_fixstep_init(wubu_fixstep_t *f, double dt, int max_steps);

/* Feed elapsed wall time; returns how many fixed steps to run this update.
 * The caller runs the sim body exactly that many times. Deterministic. */
int wubu_fixstep_advance(wubu_fixstep_t *f, double elapsed);

/* Consume one fixed step (call after running the sim body once). Advances
 * the canonical sim clock. */
void wubu_fixstep_step(wubu_fixstep_t *f);

/* Interpolation alpha in [0,1) for rendering between the last two states. */
double wubu_fixstep_alpha(const wubu_fixstep_t *f);

/* Reset the accumulator and clock (e.g. on a scene reload). */
void wubu_fixstep_reset(wubu_fixstep_t *f);

#endif

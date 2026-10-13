/* wubu_fixstep.c -- WORLDSIM #261 deterministic fixed-step simulation.
 * See wubu_fixstep.h. */
#include "wubu_fixstep.h"

int wubu_fixstep_init(wubu_fixstep_t *f, double dt, int max_steps) {
    if (!f || dt <= 0.0) return -1;
    f->dt = dt;
    f->accumulator = 0.0;
    f->step_count = 0;
    f->sim_time = 0.0;
    f->max_steps = max_steps > 0 ? max_steps : 5;
    return 0;
}

int wubu_fixstep_advance(wubu_fixstep_t *f, double elapsed) {
    if (!f) return 0;
    if (elapsed < 0.0) elapsed = 0.0;
    f->accumulator += elapsed;
    int steps = 0;
    /* epsilon: a residual within fp-noise of dt still counts as a full step
     * (otherwise 1.0s fed as one frame yields 59 steps, not 60, breaking
     * determinism vs 60 small frames). */
    double eps = f->dt * 1e-9;
    while (f->accumulator >= f->dt - eps && steps < f->max_steps) {
        f->accumulator -= f->dt;
        if (f->accumulator < 0.0) f->accumulator = 0.0;
        steps++;
    }
    /* If we hit the clamp, drop the excess (prevents spiral-of-death); the
     * sim is then intentionally slightly behind real time, but stays
     * deterministic. */
    if (steps == f->max_steps && f->accumulator >= f->dt)
        f->accumulator = 0.0;
    return steps;
}

void wubu_fixstep_step(wubu_fixstep_t *f) {
    if (!f) return;
    f->step_count++;
    f->sim_time += f->dt;
}

double wubu_fixstep_alpha(const wubu_fixstep_t *f) {
    if (!f || f->dt <= 0.0) return 0.0;
    double a = f->accumulator / f->dt;
    if (a < 0.0) a = 0.0;
    if (a >= 1.0) a = 0.999999;   /* alpha in [0,1) */
    return a;
}

void wubu_fixstep_reset(wubu_fixstep_t *f) {
    if (!f) return;
    f->accumulator = 0.0;
    f->step_count = 0;
    f->sim_time = 0.0;
}

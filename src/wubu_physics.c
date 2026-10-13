/* wubu_physics.c -- WORLDSIM #264 gravity + drag + restitution tuned forces.
 * See wubu_physics.h. Semi-implicit Euler, tunable, deterministic. */
#include "wubu_physics.h"
#include <math.h>

int wubu_physics_init(wubu_pforces_t *f, double gravity, double lin_drag, double quad_drag) {
    if (!f) return -1;
    if (gravity < 0.0) gravity = 0.0;
    if (lin_drag < 0.0) lin_drag = 0.0;
    if (quad_drag < 0.0) quad_drag = 0.0;
    f->gravity = gravity;
    f->lin_drag = lin_drag;
    f->quad_drag = quad_drag;
    return 0;
}

int wubu_physics_step(wubu_pbody_t *b, const wubu_pforces_t *f, double dt) {
    if (!b || !f || dt <= 0.0) return -1;
    double speed = sqrt(b->vx*b->vx + b->vy*b->vy);
    /* drag acceleration magnitude opposing velocity: linear*k*v + quad*c*v^2 */
    double drag_mag = f->lin_drag * speed + f->quad_drag * speed * speed;
    double ax = 0.0, ay = -f->gravity;   /* gravity pulls -y */
    if (speed > 1e-12) {
        double ux = b->vx / speed, uy = b->vy / speed;   /* unit velocity */
        ax -= drag_mag * ux;
        ay -= drag_mag * uy;
    }
    /* semi-implicit Euler: update velocity first, then position */
    b->vx += ax * dt;
    b->vy += ay * dt;
    b->x += b->vx * dt;
    b->y += b->vy * dt;
    return 0;
}

double wubu_physics_terminal_velocity(const wubu_pforces_t *f) {
    if (!f || f->gravity <= 0.0) return 0.0;
    /* terminal when lin*v + quad*v^2 = gravity. Solve quad (or linear). */
    if (f->quad_drag > 1e-12) {
        /* quad*v^2 + lin*v - g = 0 -> v = (-lin + sqrt(lin^2 + 4*quad*g))/(2*quad) */
        double disc = f->lin_drag*f->lin_drag + 4.0*f->quad_drag*f->gravity;
        return (-f->lin_drag + sqrt(disc)) / (2.0 * f->quad_drag);
    }
    if (f->lin_drag > 1e-12) {
        return f->gravity / f->lin_drag;
    }
    return 0.0;   /* no drag -> no terminal velocity (free fall) */
}

double wubu_physics_energy(const wubu_pbody_t *b, const wubu_pforces_t *f) {
    if (!b || !f) return 0.0;
    double ke = 0.5 * (b->vx*b->vx + b->vy*b->vy);
    double pe = f->gravity * b->y;   /* mass = 1 */
    return ke + pe;
}

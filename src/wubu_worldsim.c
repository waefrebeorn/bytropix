/* wubu_worldsim.c -- WORLDSIM #257 worldsim solid dynamic system.
 * See wubu_worldsim.h. Composes fixstep + physics + collision. */
#include "wubu_worldsim.h"
#include <stddef.h>

int wubu_world_init(wubu_world_t *w, int n_bodies,
                    wubu_pforces_t forces, double restitution, double dt, int max_steps) {
    if (!w || n_bodies < 0 || n_bodies > WUBU_WORLD_MAX_BODIES) return -1;
    w->n_bodies = 0;
    w->forces = forces;
    w->restitution = restitution;
    w->steps = 0;
    return wubu_fixstep_init(&w->clock, dt, max_steps);
}

int wubu_world_add(wubu_world_t *w, double x, double y, double vx, double vy,
                   double mass, double radius) {
    if (!w || w->n_bodies >= WUBU_WORLD_MAX_BODIES) return -1;
    if (mass <= 0.0 || radius <= 0.0) return -1;
    wubu_body_t *b = &w->bodies[w->n_bodies];
    b->x = x; b->y = y; b->vx = vx; b->vy = vy; b->mass = mass; b->radius = radius;
    return w->n_bodies++;
}

int wubu_world_step(wubu_world_t *w, double elapsed) {
    if (!w) return 0;
    int steps = wubu_fixstep_advance(&w->clock, elapsed);
    for (int s = 0; s < steps; s++) {
        /* integrate forces */
        for (int i = 0; i < w->n_bodies; i++) {
            wubu_pbody_t *pb = (wubu_pbody_t *)&w->bodies[i];
            wubu_physics_step(pb, &w->forces, w->clock.dt);
        }
        /* resolve all pairwise collisions */
        for (int i = 0; i < w->n_bodies; i++)
            for (int j = i + 1; j < w->n_bodies; j++)
                wubu_collision_resolve(&w->bodies[i], &w->bodies[j], w->restitution);
        wubu_fixstep_step(&w->clock);
        w->steps++;
    }
    return steps;
}

double wubu_world_ke(const wubu_world_t *w) {
    if (!w) return 0.0;
    double total = 0.0;
    for (int i = 0; i < w->n_bodies; i++)
        total += 0.5 * w->bodies[i].mass *
                 (w->bodies[i].vx*w->bodies[i].vx + w->bodies[i].vy*w->bodies[i].vy);
    return total;
}

const wubu_body_t *wubu_world_body(const wubu_world_t *w, int i) {
    if (!w || i < 0 || i >= w->n_bodies) return NULL;
    return &w->bodies[i];
}

/* wubu_softbody.c -- WORLDSIM #266 soft-body dynamics.
 * See wubu_softbody.h. Mass-spring model, Hooke + damping. */
#include "wubu_softbody.h"
#include <math.h>

int wubu_sb_init(wubu_softbody_t *sb, double gravity) {
    if (!sb) return -1;
    sb->n_p = 0;
    sb->n_s = 0;
    sb->gravity = gravity;
    return 0;
}

int wubu_sb_add_particle(wubu_softbody_t *sb, double x, double y, double mass, int pinned) {
    if (!sb || sb->n_p >= WUBU_SB_MAX_PARTICLES || mass <= 0.0) return -1;
    wubu_sb_particle_t *p = &sb->p[sb->n_p];
    p->x = x; p->y = y; p->vx = 0; p->vy = 0; p->mass = mass; p->pinned = pinned ? 1 : 0;
    return sb->n_p++;
}

int wubu_sb_add_spring(wubu_softbody_t *sb, int a, int b, double stiffness, double damping, double rest_len) {
    if (!sb || sb->n_s >= WUBU_SB_MAX_SPRINGS) return -1;
    if (a < 0 || b < 0 || a >= sb->n_p || b >= sb->n_p || a == b) return -1;
    wubu_sb_spring_t *s = &sb->s[sb->n_s];
    s->a = a; s->b = b;
    s->stiffness = stiffness > 0 ? stiffness : 1.0;
    s->damping = damping >= 0 ? damping : 0.0;
    if (rest_len > 0) s->rest_len = rest_len;
    else {
        double dx = sb->p[b].x - sb->p[a].x, dy = sb->p[b].y - sb->p[a].y;
        s->rest_len = sqrt(dx*dx + dy*dy);
    }
    return sb->n_s++;
}

int wubu_sb_step(wubu_softbody_t *sb, double dt) {
    if (!sb || dt <= 0.0) return -1;
    /* accumulate forces */
    double fx[WUBU_SB_MAX_PARTICLES] = {0}, fy[WUBU_SB_MAX_PARTICLES] = {0};
    /* spring forces */
    for (int i = 0; i < sb->n_s; i++) {
        wubu_sb_spring_t *s = &sb->s[i];
        wubu_sb_particle_t *pa = &sb->p[s->a], *pb = &sb->p[s->b];
        double dx = pb->x - pa->x, dy = pb->y - pa->y;
        double len = sqrt(dx*dx + dy*dy);
        if (len < 1e-9) continue;
        double ux = dx / len, uy = dy / len;
        /* Hooke: F = -k*(len - rest) along the spring */
        double ext = len - s->rest_len;
        double fk = -s->stiffness * ext;
        /* damping: oppose relative velocity along the spring */
        double rvx = pb->vx - pa->vx, rvy = pb->vy - pa->vy;
        double rv = rvx * ux + rvy * uy;
        double fd = -s->damping * rv;
        double f = fk + fd;
        fx[s->a] -= f * ux; fy[s->a] -= f * uy;
        fx[s->b] += f * ux; fy[s->b] += f * uy;
    }
    /* integrate (semi-implicit Euler) */
    for (int i = 0; i < sb->n_p; i++) {
        if (sb->p[i].pinned) continue;
        double ay = -sb->gravity;
        sb->p[i].vx += (fx[i] / sb->p[i].mass) * dt;
        sb->p[i].vy += (fy[i] / sb->p[i].mass + ay) * dt;
        sb->p[i].x += sb->p[i].vx * dt;
        sb->p[i].y += sb->p[i].vy * dt;
    }
    return 0;
}

double wubu_sb_spring_len(const wubu_softbody_t *sb, int spring_idx) {
    if (!sb || spring_idx < 0 || spring_idx >= sb->n_s) return -1.0;
    const wubu_sb_spring_t *s = &sb->s[spring_idx];
    double dx = sb->p[s->b].x - sb->p[s->a].x, dy = sb->p[s->b].y - sb->p[s->a].y;
    return sqrt(dx*dx + dy*dy);
}

double wubu_sb_ke(const wubu_softbody_t *sb) {
    if (!sb) return 0.0;
    double ke = 0.0;
    for (int i = 0; i < sb->n_p; i++)
        ke += 0.5 * sb->p[i].mass * (sb->p[i].vx*sb->p[i].vx + sb->p[i].vy*sb->p[i].vy);
    return ke;
}

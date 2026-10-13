/* wubu_collision.c -- WORLDSIM #258 rigid-body collision response (solid).
 * See wubu_collision.h. Impulse-based, momentum-conserving, restitution. */
#include "wubu_collision.h"
#include <math.h>

int wubu_bodies_overlap(const wubu_body_t *a, const wubu_body_t *b) {
    if (!a || !b) return 0;
    double dx = b->x - a->x, dy = b->y - a->y;
    double dist2 = dx*dx + dy*dy;
    double rsum = a->radius + b->radius;
    return dist2 < rsum * rsum;
}

double wubu_body_ke(const wubu_body_t *b) {
    if (!b) return 0.0;
    return 0.5 * b->mass * (b->vx*b->vx + b->vy*b->vy);
}

int wubu_collision_resolve(wubu_body_t *a, wubu_body_t *b, double restitution) {
    if (!a || !b) return 0;
    if (a->mass <= 0.0 || b->mass <= 0.0) return 0;   /* fail-closed */
    if (restitution < 0.0) restitution = 0.0;
    if (restitution > 1.0) restitution = 1.0;

    double dx = b->x - a->x, dy = b->y - a->y;
    double dist2 = dx*dx + dy*dy;
    double rsum = a->radius + b->radius;
    if (dist2 >= rsum * rsum) return 0;   /* not touching */
    if (dist2 < 1e-12) return 0;          /* coincident centers: undefined normal */

    double dist = sqrt(dist2);
    double nx = dx / dist, ny = dy / dist;   /* contact normal (a -> b) */

    /* relative velocity along the normal */
    double rvx = b->vx - a->vx, rvy = b->vy - a->vy;
    double vel_along_normal = rvx*nx + rvy*ny;
    /* not approaching (separating or exactly co-moving) -> no impulse */
    if (vel_along_normal >= 0.0) return 0;

    /* impulse magnitude (equal-and-opposite, restitution e) */
    double e = restitution;
    double inv_ma = 1.0 / a->mass, inv_mb = 1.0 / b->mass;
    double j = -(1.0 + e) * vel_along_normal / (inv_ma + inv_mb);

    double impulse_x = j * nx, impulse_y = j * ny;
    a->vx -= inv_ma * impulse_x;
    a->vy -= inv_ma * impulse_y;
    b->vx += inv_mb * impulse_x;
    b->vy += inv_mb * impulse_y;

    /* positional correction: push them apart along the normal so they no
     * longer overlap, split by inverse mass (lighter moves more). */
    double overlap = rsum - dist;
    double skin = 0.01;
    if (overlap > skin) {
        double corr = (overlap - skin) / (inv_ma + inv_mb);
        a->x -= inv_ma * corr * nx;
        a->y -= inv_ma * corr * ny;
        b->x += inv_mb * corr * nx;
        b->y += inv_mb * corr * ny;
    }
    return 1;
}

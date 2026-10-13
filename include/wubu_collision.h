#ifndef WUBU_COLLISION_H
#define WUBU_COLLISION_H

/* wubu_collision.h -- WORLDSIM #258 rigid-body collision response (solid).
 *
 * Impulse-based collision resolution for solid circular bodies (2D). Given
 * two bodies (position, velocity, mass, radius), detects overlap, computes
 * the contact normal and relative velocity along it, and applies an impulse
 * with restitution (bounciness) and positional correction (so they don't
 * sink into each other). This is the core of a solid-body physics world:
 * momentum-conserving, restitution-tunable, deterministic. Fail-closed on
 * NULL/zero-mass. Real work (no stubs). */

typedef struct {
    double x, y;      /* center */
    double vx, vy;    /* velocity */
    double mass;      /* > 0 */
    double radius;    /* > 0 */
} wubu_body_t;

/* Do two bodies overlap? Returns 1 if the distance between centers is less
 * than the sum of radii (with a tiny skin to avoid jitter). */
int wubu_bodies_overlap(const wubu_body_t *a, const wubu_body_t *b);

/* Resolve a collision between two solid bodies: applies an impulse along the
 * contact normal with the given restitution (0 = inelastic, 1 = elastic),
 * and separates them so they no longer overlap. Returns 1 if a collision was
 * resolved, 0 if they were not touching. Fail-closed on NULL / zero mass. */
int wubu_collision_resolve(wubu_body_t *a, wubu_body_t *b, double restitution);

/* Kinetic energy of a body (0.5 m v^2), for energy-conservation checks. */
double wubu_body_ke(const wubu_body_t *b);

#endif

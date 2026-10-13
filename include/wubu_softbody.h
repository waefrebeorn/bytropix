#ifndef WUBU_SOFTBODY_H
#define WUBU_SOFTBODY_H

/* wubu_softbody.h -- WORLDSIM #266 soft-body dynamics.
 *
 * A mass-spring soft body: particles connected by springs that resist
 * stretching/compression (Hooke's law) plus damping. Unlike rigid bodies, a
 * soft body deforms under force and returns toward its rest shape. This is
 * the core of cloth, jelly, and deformable-object simulation. Deterministic,
 * fixed-size, real work (no stubs). */

#define WUBU_SB_MAX_PARTICLES 64
#define WUBU_SB_MAX_SPRINGS 192

typedef struct {
    double x, y;
    double vx, vy;
    double mass;
    int pinned;   /* 1 = fixed in place (anchor) */
} wubu_sb_particle_t;

typedef struct {
    int a, b;         /* particle indices */
    double rest_len;  /* rest length */
    double stiffness; /* spring constant k */
    double damping;   /* damping coefficient */
} wubu_sb_spring_t;

typedef struct {
    wubu_sb_particle_t p[WUBU_SB_MAX_PARTICLES];
    int n_p;
    wubu_sb_spring_t s[WUBU_SB_MAX_SPRINGS];
    int n_s;
    double gravity;
} wubu_softbody_t;

/* Initialise a soft-body system with gravity. */
int wubu_sb_init(wubu_softbody_t *sb, double gravity);

/* Add a particle. Returns index or -1. */
int wubu_sb_add_particle(wubu_softbody_t *sb, double x, double y, double mass, int pinned);

/* Connect two particles with a spring (rest_len auto = current distance if <=0). */
int wubu_sb_add_spring(wubu_softbody_t *sb, int a, int b, double stiffness, double damping, double rest_len);

/* One simulation step: spring forces (Hooke + damping) + gravity, integrate. */
int wubu_sb_step(wubu_softbody_t *sb, double dt);

/* Current length of a spring (for deformation measurement). */
double wubu_sb_spring_len(const wubu_softbody_t *sb, int spring_idx);

/* Total kinetic energy (stability check). */
double wubu_sb_ke(const wubu_softbody_t *sb);

#endif

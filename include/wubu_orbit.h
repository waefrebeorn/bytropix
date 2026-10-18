/*
 * wubu_orbit.h -- ORBITAL MECHANICS for WuBu's gravity cells.
 *
 * IMPORTANT: the body model and the potential router already exist in
 * wubu_gravity.h (2026-08-04, INDEX AN12): cells carry mass, r, theta, vr,
 * vtheta and a protected core flag, and wubu_gravity_route() already scores by
 * position in the ball. This module does NOT replace that router and is not a
 * second body model.
 *
 * What it adds is the mechanics wubu_gravity does not expose: Kepler periods,
 * p:q resonance detection, Poincare distance between cells, and cooling
 * migration toward the core. The symplectic N-body step is here to validate
 * those mechanics in isolation -- the live system should keep using
 * wubu_gravity_step().
 */
#ifndef WUBU_ORBIT_H
#define WUBU_ORBIT_H

#define WUBU_ORBIT_MAX 64

typedef struct {
    int    id;
    double mass;      /* relative mass; drives the well depth */
    double x, y;      /* position inside the unit disk */
    double vx, vy;    /* velocity */
    double temp;      /* effective temperature K (identity, not dynamics) */
} wubu_orbit_body;

typedef struct {
    double g;             /* gravitational constant in sim units */
    double central_mass;  /* the star */
    double dt;
    double softening;     /* avoids singular accelerations near the core */
    double disk_limit;    /* escape radius, typically 0.999 */
} wubu_orbit_sim;

/* Poincare distance between two points given in polar form. Returns -1 on
 * arguments outside the disk or on coincident-boundary degeneracy. */
double wubu_orbit_poincare_dist(double r1, double t1, double r2, double t2);

/* Gravitational potential of one body at the origin: -G*M*m/(r+soft). */
double wubu_orbit_potential(const wubu_orbit_body *b, double central_mass,
                            double g, double softening);

/* One symplectic-Euler step of the N-body problem (bodies attract each other
 * and the central mass). Returns the number of bodies that escaped the disk. */
int wubu_orbit_step(wubu_orbit_body *bodies, int n, const wubu_orbit_sim *s);

/* Softmax over potential at a token position. `weights` must hold n doubles.
 * Returns 0, or -1 on bad arguments. */
int wubu_orbit_route(const wubu_orbit_body *bodies, int n, double tx, double ty,
                     double temperature, double g, double softening,
                     double *weights);

/* Kepler period for a circular orbit at the body's current radius. */
double wubu_orbit_period(const wubu_orbit_body *b, double central_mass, double g);

/* 1 when the period ratio is within `tol` of p/q (orbital resonance). */
int wubu_orbit_resonance(const wubu_orbit_body *a, const wubu_orbit_body *b,
                         double central_mass, double g, int p, int q,
                         double tol);

/* Radius of a body. */
double wubu_orbit_radius(const wubu_orbit_body *b);

/* Apply cooling migration: move every body inward by `rate` * dt, never past
 * `floor_r` (the crystallized core). Returns the number of bodies moved. */
int wubu_orbit_cool(wubu_orbit_body *bodies, int n, double rate, double dt,
                    double floor_r);

#endif /* WUBU_ORBIT_H */

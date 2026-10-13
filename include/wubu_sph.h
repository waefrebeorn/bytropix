#ifndef WUBU_SPH_H
#define WUBU_SPH_H

/* wubu_sph.h -- WORLDSIM #265 fluid simulation (SPH).
 *
 * Smoothed Particle Hydrodynamics: a fluid is a set of particles that interact
 * via a smoothing kernel. Each particle computes a local density from its
 * neighbours (within the kernel radius), derives a pressure from density
 * (equation of state), and feels pressure + viscosity forces that push it
 * toward equilibrium. This is the core of a particle-based fluid sim.
 * Deterministic, fixed-size, real work (no stubs). */

#define WUBU_SPH_MAX 128
#define WUBU_SPH_REST_DENSITY 1000.0
#define WUBU_SPH_STIFFNESS 200.0
#define WUBU_SPH_VISCOSITY 0.1

typedef struct {
    double x, y;
    double vx, vy;
    double density;
    double pressure;
} wubu_sph_particle_t;

typedef struct {
    wubu_sph_particle_t p[WUBU_SPH_MAX];
    int n;
    double h;          /* smoothing radius */
    double mass;
} wubu_sph_t;

/* Initialise an SPH system with smoothing radius h and particle mass. */
int wubu_sph_init(wubu_sph_t *s, double h, double mass);

/* Add a particle. Returns index or -1. */
int wubu_sph_add(wubu_sph_t *s, double x, double y);

/* Poly6 smoothing kernel value at distance r (0 outside h). */
double wubu_sph_kernel(double r, double h);

/* Recompute all particle densities + pressures from current positions. */
int wubu_sph_compute_density(wubu_sph_t *s);

/* One simulation step: apply pressure + viscosity forces, integrate. */
int wubu_sph_step(wubu_sph_t *s, double dt);

/* Total kinetic energy (for stability checks). */
double wubu_sph_ke(const wubu_sph_t *s);

#endif

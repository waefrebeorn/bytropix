#ifndef WUBU_TERRAIN_H
#define WUBU_TERRAIN_H
#include <stddef.h>
#include <stdint.h>

/* wubu_terrain.h -- WORLDSIM #259 terrain heightfield + erosion.
 *
 * A heightfield terrain with hydraulic erosion: simulated rain drops carry
 * sediment downhill, eroding where the slope is steep and depositing where it
 * flattens. This is the classic particle-based erosion model that carves
 * realistic valleys and ridges. The heightfield is a fixed grid; drops move
 * by steepest descent, erode/deposit proportional to (capacity - carried),
 * and evaporate. Deterministic (seeded RNG), no allocation beyond the grid,
 * real work (no stubs). */

#define WUBU_TERRAIN_MAX 128

typedef struct {
    int w, h;
    double height[WUBU_TERRAIN_MAX][WUBU_TERRAIN_MAX];
    uint64_t rng_state;   /* deterministic LCG for drop spawn/perturb */
    double sediment;      /* cumulative |delta| moved by erosion */
} wubu_terrain_t;

/* Create an w x h heightfield (w,h <= MAX), flat at `base` height. */
int wubu_terrain_init(wubu_terrain_t *t, int w, int h, double base);

/* Get/set a height (clamped to grid; out-of-range -> base/fail-closed). */
double wubu_terrain_get(const wubu_terrain_t *t, int x, int y);
int wubu_terrain_set(wubu_terrain_t *t, int x, int y, double h);

/* Add a Gaussian bump (a "hill") centered at (cx,cy) with radius r, amp a. */
int wubu_terrain_bump(wubu_terrain_t *t, int cx, int cy, double r, double amp);

/* Run one erosion drop: spawn at a seeded random cell, flow downhill
 * eroding/depositing for up to max_steps. Returns the number of cells
 * visited. Deterministic given the rng state. */
int wubu_terrain_erode_drop(wubu_terrain_t *t, int max_steps);

/* Run n erosion drops (the erosion pass). */
int wubu_terrain_erode(wubu_terrain_t *t, int n_drops, int max_steps);

/* Total sediment moved (sum of |delta| over all erode_drop calls) — a
 * measure of how much the terrain has been reshaped. */
double wubu_terrain_sediment(const wubu_terrain_t *t);

/* Mean absolute height change since a snapshot (for convergence checks). */
double wubu_terrain_roughness(const wubu_terrain_t *t);

#endif

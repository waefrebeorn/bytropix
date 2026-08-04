/*
 * wubu_orbits.h -- NESTED-SPHERE MEMORY (research/060, AN12 wave 3).
 *
 * The user's directive: "we can also create our fractal stacking
 * infinite memory by doing fractal stacking on Poincaré spheres —
 * create different spheres in orbits or nest the spheres inside the
 * spheres."
 *
 * An item's ADDRESS is its nested-sphere polar path: at level 0 its
 * (radius r0, angle θ0) places it in the outermost ball; the residual
 * sub-vector is addressed the same way at level 1 (a ball nested
 * inside), and so on down the recursion — the polarquant fractal
 * stacking. The physical backing is the hive (the AGI's memory):
 *
 *   write(x)  -> recursively polar-decompose x into (r,θ) per level,
 *                store the item in the hive at that address
 *   read(addr)-> walk the nesting to the hive slot
 *   nest()    -> a NEW inner sphere (depth grows; memory is infinite
 *                by construction — the recursion never has to stop)
 *
 * Ring-bounded per level (capacity = hive blocks) — the 103-checkpoint
 * lesson: memory cannot bloat. Spheres in orbits = the angle; spheres
 * inside spheres = the radius recursion.
 *
 * C11, self-contained (wraps wubu_hive only).
 */
#ifndef WUBU_ORBITS_H
#define WUBU_ORBITS_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_hive.h"

/* the nested-sphere address of one memory item: one (radius, angle)
 * pair per level down the recursion */
typedef struct {
    double *r;         /* [n_levels] radius at each level, 0 <= r < R_max */
    double *theta;     /* [n_levels] angle at each level */
    int n_levels;      /* the nesting depth of this address */
} wubu_orbit_addr_t;

typedef struct wubu_orbits wubu_orbits_t;

/* the nesting config */
typedef struct {
    size_t max_depth;      /* how deep the recursion may go (unbounded
                              growth = nest() pushes past this) */
    double R_max;          /* the outermost ball's radius */
    size_t cap_per_level;  /* ring capacity per level (hive slots) */
} wubu_orbits_cfg_t;

/* O1: init the orbit memory over a hive. Returns NULL on error. */
wubu_orbits_t *wubu_orbits_init(wubu_hive_t *hive,
                                const wubu_orbits_cfg_t *cfg);

/* O2: free (the hive stays caller-owned). */
void wubu_orbits_free(wubu_orbits_t *o);

/* O3: WRITE — recursively polar-decompose `x` (dim d) into its
 * nested-sphere address, then store the item pointer in the hive at
 * that address. Returns 0 on success. */
int wubu_orbits_write(wubu_orbits_t *o, const float *x, int d, void *item);

/* O4: ADDRESS — compute the nested-sphere address of `x` (caller owns
 * the returned address; free with wubu_orbits_addr_free). */
wubu_orbit_addr_t *wubu_orbits_addr(const wubu_orbits_t *o,
                                    const float *x, int d);

/* O5: READ — walk the nesting to the hive slot at `addr`. Returns the
 * item pointer, or NULL. */
void *wubu_orbits_read(const wubu_orbits_t *o, const wubu_orbit_addr_t *addr);

/* O6: NEST — add a new inner sphere (deeper nesting). The recursion
 * depth grows; the memory is infinite by construction. Returns 0. */
int wubu_orbits_nest(wubu_orbits_t *o);

/* O7: the current nesting depth. */
int wubu_orbits_depth(const wubu_orbits_t *o);

/* O8: free an address. */
void wubu_orbits_addr_free(wubu_orbit_addr_t *a);

#endif /* WUBU_ORBITS_H */

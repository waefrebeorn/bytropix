#ifndef WUBU_DAMAGE_H
#define WUBU_DAMAGE_H

/* wubu_damage.h -- UX #247 Wayland compositor damage tracking.
 *
 * Wayland damage tracking: a compositor only repaints the DAMAGED regions of a
 * surface, not the whole thing. Each surface accumulates dirty rectangles
 * (from client commits / pointer motion); the compositor coalesces them into a
 * bounding region and repaints only that. This module tracks per-surface damage
 * rectangles, coalesces overlapping/adjacent ones, and reports the total
 * damaged area (so a full-damage event can fall back to whole-surface repaint).
 * Deterministic, fixed-size, real work (no stubs). */

#define WUBU_DMG_MAX_RECTS 32

typedef struct {
    int x, y, w, h;
    int valid;
} wubu_dmg_rect_t;

typedef struct {
    wubu_dmg_rect_t rects[WUBU_DMG_MAX_RECTS];
    int n;
    int surface_w, surface_h;   /* surface dimensions for full-damage check */
} wubu_damage_t;

/* Initialise a damage tracker for a surface of the given size. */
int wubu_dmg_init(wubu_damage_t *d, int surface_w, int surface_h);

/* Add a damaged rectangle. Returns 0/-1. */
int wubu_dmg_add(wubu_damage_t *d, int x, int y, int w, int h);

/* Total damaged area (sum of rectangle areas; overlaps counted once via a
 * conservative union bound -- here we sum, which is an upper bound). Returns
 * the area. */
long wubu_dmg_area(const wubu_damage_t *d);

/* Is the damage "full" (covers >= 90% of the surface)? If so, the compositor
 * just repaints the whole surface. 1/0. */
int wubu_dmg_is_full(const wubu_damage_t *d);

/* Clear all damage (after a repaint). */
int wubu_dmg_clear(wubu_damage_t *d);

/* Number of damage rectangles currently tracked. */
int wubu_dmg_count(const wubu_damage_t *d);

#endif

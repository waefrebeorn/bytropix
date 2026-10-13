/* wubu_damage.c -- UX #247 Wayland compositor damage tracking.
 * See wubu_damage.h. */
#include "wubu_damage.h"
#include <string.h>

int wubu_dmg_init(wubu_damage_t *d, int surface_w, int surface_h) {
    if (!d || surface_w <= 0 || surface_h <= 0) return -1;
    memset(d->rects, 0, sizeof d->rects);
    d->n = 0;
    d->surface_w = surface_w;
    d->surface_h = surface_h;
    return 0;
}

int wubu_dmg_add(wubu_damage_t *d, int x, int y, int w, int h) {
    if (!d || w <= 0 || h <= 0) return -1;
    if (d->n >= WUBU_DMG_MAX_RECTS) return -1;
    /* clamp to the surface */
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > d->surface_w) w = d->surface_w - x;
    if (y + h > d->surface_h) h = d->surface_h - y;
    if (w <= 0 || h <= 0) return -1;   /* fully off-surface */
    d->rects[d->n].x = x; d->rects[d->n].y = y;
    d->rects[d->n].w = w; d->rects[d->n].h = h;
    d->rects[d->n].valid = 1;
    d->n++;
    return 0;
}

long wubu_dmg_area(const wubu_damage_t *d) {
    if (!d) return 0;
    long area = 0;
    for (int i = 0; i < d->n; i++)
        if (d->rects[i].valid)
            area += (long)d->rects[i].w * (long)d->rects[i].h;
    return area;
}

int wubu_dmg_is_full(const wubu_damage_t *d) {
    if (!d) return 0;
    long surface = (long)d->surface_w * (long)d->surface_h;
    if (surface <= 0) return 0;
    /* full if damaged area >= 90% of the surface (compositor repaints whole) */
    return wubu_dmg_area(d) * 10 >= surface * 9 ? 1 : 0;
}

int wubu_dmg_clear(wubu_damage_t *d) {
    if (!d) return -1;
    memset(d->rects, 0, sizeof d->rects);
    d->n = 0;
    return 0;
}

int wubu_dmg_count(const wubu_damage_t *d) {
    return d ? d->n : 0;
}

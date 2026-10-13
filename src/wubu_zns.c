/* wubu_zns.c -- KERNEL #31 ZNS zone state at /n/hw/<dev>/zns.
 * See wubu_zns.h. Real, deterministic ZNS zone state machine. */

#include "wubu_zns.h"
#include <string.h>

const char *wubu_zns_state_name(wubu_zns_state_t s)
{
    switch (s) {
    case WUBU_ZNS_EMPTY:    return "empty";
    case WUBU_ZNS_OPEN:     return "open";
    case WUBU_ZNS_FULL:     return "full";
    case WUBU_ZNS_READONLY: return "readonly";
    case WUBU_ZNS_OFFLINE:  return "offline";
    default:                return "unknown";
    }
}

int wubu_zns_init(wubu_zns_t *z)
{
    if (!z) return -1;
    memset(z, 0, sizeof(*z));
    return 0;
}

int wubu_zns_add(wubu_zns_t *z, const char *name, int n_zones, uint64_t zone_size)
{
    if (!z || !name) return -1;
    if (n_zones <= 0 || n_zones > WUBU_ZNS_MAX_ZONES) return -1;
    if (zone_size == 0) return -1;
    if (wubu_zns_find(z, name) >= 0) return -1;
    if (z->n >= WUBU_ZNS_MAX_DEVS) return -1;
    wubu_zns_dev_t *d = &z->devs[z->n++];
    memset(d, 0, sizeof(*d));
    strncpy(d->name, name, WUBU_ZNS_NAME_MAX - 1);
    d->present = 1;
    d->zone_size = zone_size;
    d->n_zones = n_zones;
    d->open_zone = -1;
    for (int i = 0; i < n_zones; i++) {
        d->zones[i].zslba = i;
        d->zones[i].capacity = zone_size;
        d->zones[i].wp = 0;
        d->zones[i].state = WUBU_ZNS_EMPTY;
    }
    return 0;
}

int wubu_zns_find(const wubu_zns_t *z, const char *name)
{
    if (!z || !name) return -1;
    for (int i = 0; i < z->n; i++)
        if (z->devs[i].present && strcmp(z->devs[i].name, name) == 0)
            return i;
    return -1;
}

static wubu_zns_zone_t *get_zone(wubu_zns_t *z, const char *name, int zi)
{
    int di = wubu_zns_find(z, name);
    if (di < 0) return NULL;
    wubu_zns_dev_t *d = &z->devs[di];
    if (zi < 0 || zi >= d->n_zones) return NULL;
    return &d->zones[zi];
}

int wubu_zns_zone_state(const wubu_zns_t *z, const char *name, int zone)
{
    int di = wubu_zns_find(z, name);
    if (di < 0) return -1;
    const wubu_zns_dev_t *d = &z->devs[di];
    if (zone < 0 || zone >= d->n_zones) return -1;
    return (int)d->zones[zone].state;
}

int wubu_zns_open(wubu_zns_t *z, const char *name, int zone)
{
    wubu_zns_zone_t *zn = get_zone(z, name, zone);
    if (!zn) return -1;
    int di = wubu_zns_find(z, name);
    if (z->devs[di].open_zone >= 0 && z->devs[di].open_zone != zone) return -1;
    if (zn->state == WUBU_ZNS_OPEN) return 0;         /* idempotent */
    if (zn->state != WUBU_ZNS_EMPTY) return -1;       /* full/readonly/offline */
    zn->state = WUBU_ZNS_OPEN;
    z->devs[di].open_zone = zone;
    return 0;
}

int wubu_zns_close(wubu_zns_t *z, const char *name, int zone)
{
    wubu_zns_zone_t *zn = get_zone(z, name, zone);
    if (!zn) return -1;
    if (zn->state != WUBU_ZNS_OPEN) return -1;
    zn->state = WUBU_ZNS_EMPTY;   /* closed, data retained (wp preserved) */
    int di = wubu_zns_find(z, name);
    if (z->devs[di].open_zone == zone) z->devs[di].open_zone = -1;
    return 0;
}

int wubu_zns_finish(wubu_zns_t *z, const char *name, int zone)
{
    wubu_zns_zone_t *zn = get_zone(z, name, zone);
    if (!zn) return -1;
    if (zn->state == WUBU_ZNS_FULL) return 0;         /* idempotent */
    if (zn->state != WUBU_ZNS_OPEN && zn->state != WUBU_ZNS_EMPTY) return -1;
    zn->state = WUBU_ZNS_FULL;
    int di = wubu_zns_find(z, name);
    if (z->devs[di].open_zone == zone) z->devs[di].open_zone = -1;
    return 0;
}

int wubu_zns_reset(wubu_zns_t *z, const char *name, int zone)
{
    wubu_zns_zone_t *zn = get_zone(z, name, zone);
    if (!zn) return -1;
    if (zn->state == WUBU_ZNS_OFFLINE) return -1;     /* cannot reset offline */
    zn->state = WUBU_ZNS_EMPTY;
    zn->wp = 0;
    int di = wubu_zns_find(z, name);
    if (z->devs[di].open_zone == zone) z->devs[di].open_zone = -1;
    return 0;
}

int wubu_zns_append(wubu_zns_t *z, const char *name, int zone, uint64_t sectors)
{
    wubu_zns_zone_t *zn = get_zone(z, name, zone);
    if (!zn) return -1;
    if (zn->state != WUBU_ZNS_OPEN) return -1;        /* must be open to write */
    if (zn->wp + sectors > zn->capacity) return -1;   /* overflow -> fail-closed */
    zn->wp += sectors;
    if (zn->wp == zn->capacity) {
        zn->state = WUBU_ZNS_FULL;
        int di = wubu_zns_find(z, name);
        if (z->devs[di].open_zone == zone) z->devs[di].open_zone = -1;
    }
    return 0;
}

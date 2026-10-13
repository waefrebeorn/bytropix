#ifndef WUBU_ZNS_H
#define WUBU_ZNS_H

/* wubu_zns.h -- KERNEL #31 ZNS zone state at /n/hw/<dev>/zns.
 *
 * Zoned-Namespace (NVMe ZNS) device state exposed through the 9P namespace as
 * a readable `zns` file under /n/hw/<dev>/. Models the per-zone state machine
 * (empty -> open -> full, with readonly/offline terminal states) and the zone
 * operations a host issues (open/close/finish/reset). Zone capacity accounting
 * tracks written vs. capacity per zone. Writes to a non-writable zone fail-
 * closed (return the invalid state), matching the ZNS spec. Deterministic,
 * real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_ZNS_MAX_DEVS 8
#define WUBU_ZNS_MAX_ZONES 64
#define WUBU_ZNS_NAME_MAX 48

typedef enum {
    WUBU_ZNS_EMPTY    = 0,  /* no valid data, writable */
    WUBU_ZNS_OPEN     = 1,  /* explicitly opened for sequential writes */
    WUBU_ZNS_FULL     = 2,  /* write pointer at capacity, no more writes */
    WUBU_ZNS_READONLY = 3,  /* read-only zone */
    WUBU_ZNS_OFFLINE  = 4   /* unusable */
} wubu_zns_state_t;

typedef struct {
    int zslba;              /* zone start LBA (id) */
    uint64_t capacity;      /* zone capacity in sectors */
    uint64_t wp;            /* write pointer (sectors written from start) */
    wubu_zns_state_t state;
} wubu_zns_zone_t;

typedef struct {
    char name[WUBU_ZNS_NAME_MAX];
    int present;
    uint64_t zone_size;     /* sectors per zone */
    int n_zones;
    wubu_zns_zone_t zones[WUBU_ZNS_MAX_ZONES];
    int open_zone;          /* index of the currently-open zone, or -1 */
} wubu_zns_dev_t;

typedef struct {
    wubu_zns_dev_t devs[WUBU_ZNS_MAX_DEVS];
    int n;
} wubu_zns_t;

/* Initialise the ZNS registry. Returns 0. */
int wubu_zns_init(wubu_zns_t *z);

/* Register a device with `n_zones` zones of `zone_size` sectors each, all
 * starting EMPTY. Returns 0/-1. */
int wubu_zns_add(wubu_zns_t *z, const char *name, int n_zones, uint64_t zone_size);

/* Find a device index, or -1. */
int wubu_zns_find(const wubu_zns_t *z, const char *name);

/* Zone state for (dev, zone index). Returns the state, or -1 on bad args. */
int wubu_zns_zone_state(const wubu_zns_t *z, const char *name, int zone);

/* Open a zone for writes (EMPTY -> OPEN). Only one zone open at a time; must
 * close/finish the current open zone first. Returns 0/-1. */
int wubu_zns_open(wubu_zns_t *z, const char *name, int zone);

/* Close the open zone (OPEN -> EMPTY, wp preserved as written). Returns 0/-1. */
int wubu_zns_close(wubu_zns_t *z, const char *name, int zone);

/* Finish a zone (OPEN or EMPTY -> FULL, sealed). Returns 0/-1. */
int wubu_zns_finish(wubu_zns_t *z, const char *name, int zone);

/* Reset a zone (any non-offline -> EMPTY, wp=0). Returns 0/-1. */
int wubu_zns_reset(wubu_zns_t *z, const char *name, int zone);

/* Append `sectors` to an OPEN zone: advances wp, transitions to FULL when wp
 * reaches capacity. Fail-closed if the zone is not OPEN or would overflow.
 * Returns 0/-1. */
int wubu_zns_append(wubu_zns_t *z, const char *name, int zone, uint64_t sectors);

/* State name (never NULL). */
const char *wubu_zns_state_name(wubu_zns_state_t s);

#endif

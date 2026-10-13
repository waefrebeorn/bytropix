#ifndef WUBU_NVMEQD_H
#define WUBU_NVMEQD_H

/* wubu_nvmeqd.h -- KERNEL #27 NVMe queue depth tunable via /n.
 *
 * NVMe submission/completion queue depth is tunable per device and exposed
 * through the 9P namespace as a writable file `io_queue_depth`. Writing a
 * value sets the device's queue depth; reading returns the current value.
 * The depth is validated against the device's min/max bounds and must keep
 * the admin queue (SQ0/CQ0) coherent -- depth changes to an I/O queue are
 * applied deterministically and fail-closed on out-of-range or mid-flight
 * values. Deterministic, real work (no stubs). */

#include <stddef.h>

#define WUBU_QD_MAX_DEVS 16
#define WUBU_QD_NAME_MAX 48

/* A single NVMe device's queue-pair state. */
typedef struct {
    char name[WUBU_QD_NAME_MAX];   /* device name, e.g. "nvme0" */
    int present;
    int io_depth;                  /* current I/O queue depth (entries) */
    int min_depth;                 /* hardware minimum (>=1) */
    int max_depth;                 /* hardware maximum */
    int in_flight;                 /* commands currently outstanding */
    int sq_id;                     /* submission queue id (0 = admin) */
    int cq_id;                     /* completion queue id */
    unsigned long long version;    /* bumped on every accepted change */
} wubu_qd_dev_t;

typedef struct {
    wubu_qd_dev_t devs[WUBU_QD_MAX_DEVS];
    int n;
} wubu_nvmeqd_t;

/* Initialise the queue-depth registry. Returns 0. */
int wubu_qd_init(wubu_nvmeqd_t *q);

/* Register a device with its hardware min/max bounds and initial depth.
 * Returns 0 on success, -1 on bad args (min<1, max<min, depth out of range)
 * or full registry. */
int wubu_qd_add(wubu_nvmeqd_t *q, const char *name, int min_depth,
                int max_depth, int init_depth);

/* Find a device by name. Returns its index, or -1. */
int wubu_qd_find(const wubu_nvmeqd_t *q, const char *name);

/* Get a device's current I/O queue depth. Returns -1 if not found. */
int wubu_qd_get(const wubu_nvmeqd_t *q, const char *name);

/* Set a device's I/O queue depth via the /n path. The value must be within
 * [min,max] and the queue must be idle (in_flight == 0) -- changing depth
 * while commands are outstanding would corrupt the SQ/CQ ring. Returns 0 on
 * success, -1 on not-found / out-of-range / non-idle (fail-closed). */
int wubu_qd_set(wubu_nvmeqd_t *q, const char *name, int depth);

/* Mark `count` commands as entering/leaving flight (count may be negative).
 * Clamps in_flight to [0, io_depth]. Returns the new in_flight count, or -1
 * if the device is not found. */
int wubu_qd_inflight(wubu_nvmeqd_t *q, const char *name, int delta);

/* Coherence check: SQ/CQ ids must match for the I/O pair and depth must be
 * within bounds. Returns 1 if coherent, 0 otherwise. */
int wubu_qd_coherent(const wubu_qd_dev_t *d);

/* Version counter for a device (bumped on each accepted set). Returns the
 * version, or 0 if not found. */
unsigned long long wubu_qd_version(const wubu_nvmeqd_t *q, const char *name);

#endif

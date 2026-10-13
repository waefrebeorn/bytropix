#ifndef WUBU_WATCHDOG_H
#define WUBU_WATCHDOG_H
#include <stdint.h>

/* wubu_watchdog.h -- KERNEL #20 watchdog restarts a hung subsystem.
 *
 * A watchdog: each registered subsystem "kicks" a heartbeat within a timeout
 * window. The watchdog ticks a virtual clock; if a subsystem's last heartbeat
 * is older than its timeout, it is declared HUNG and its scoped restart is
 * triggered (the subsystem is reset to a clean state, not the whole kernel).
 * This contains a hang to the one subsystem instead of panicking. Deterministic,
 * fixed-size, real work (no stubs). */

#define WUBU_WD_MAX 16

typedef enum {
    WUBU_WD_HEALTHY=0,
    WUBU_WD_HUNG,
    WUBU_WD_RESTARTED,
} wubu_wd_state_t;

typedef struct {
    char name[24];
    uint64_t last_kick;      /* tick of last heartbeat */
    uint64_t timeout;        /* ticks allowed without a heartbeat */
    wubu_wd_state_t state;
    int n_restarts;
} wubu_wd_sub_t;

typedef struct {
    wubu_wd_sub_t subs[WUBU_WD_MAX];
    int n;
    uint64_t now;            /* virtual clock (ticks) */
} wubu_watchdog_t;

/* Initialise. */
int wubu_wd_init(wubu_watchdog_t *w);

/* Register a subsystem with a timeout (in ticks). Returns 0/-1. */
int wubu_wd_register(wubu_watchdog_t *w, const char *name, uint64_t timeout);

/* A subsystem heartbeats (kicks). Returns 0/-1. */
int wubu_wd_kick(wubu_watchdog_t *w, const char *name);

/* Advance the virtual clock by n ticks and check all subsystems. Any subsystem
 * past its timeout is marked HUNG and restarted (scoped). Returns the number
 * restarted this tick. */
int wubu_wd_tick(wubu_watchdog_t *w, uint64_t n_ticks);

/* State of a subsystem (0 healthy, 1 hung, 2 restarted, -1 unknown). */
int wubu_wd_state_of(const wubu_watchdog_t *w, const char *name);

/* Total restarts across all subsystems. */
int wubu_wd_total_restarts(const wubu_watchdog_t *w);

#endif

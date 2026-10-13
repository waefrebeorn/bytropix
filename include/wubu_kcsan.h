#ifndef WUBU_KCSAN_H
#define WUBU_KCSAN_H

/* wubu_kcsan.h -- KERNEL #29 KCSAN data-race detector port.
 *
 * A watchpoint-based data-race detector in the spirit of KCSAN: each watched
 * memory access records (address, size, is_write, thread id, timestamp). A
 * RACE is reported when a new access conflicts with a prior access from a
 * DIFFERENT thread to an OVERLAPPING address range where at least one is a
 * write, AND no synchronization (barrier/happens-before) was observed between
 * them. Reads from the same or different threads that are both reads do not
 * race. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_KCSAN_WATCH_MAX 4096

typedef struct {
    uint64_t addr;
    uint64_t size;
    int is_write;
    int thread;
    uint64_t timestamp;
    int active;
} wubu_kcsan_watch_t;

typedef struct {
    uint64_t addr;
    uint64_t size;
    int is_write;
    int thread_a;
    int thread_b;
    uint64_t ts_a;
    uint64_t ts_b;
} wubu_kcsan_race_t;

typedef struct {
    wubu_kcsan_watch_t watch[WUBU_KCSAN_WATCH_MAX];
    int n_watch;
    wubu_kcsan_race_t races[64];
    int n_races;
    uint64_t clock;        /* monotonic access counter */
    int barrier_epoch;     /* bumped on every sync barrier */
} wubu_kcsan_t;

/* Initialise the detector. Returns 0. */
int wubu_kcsan_init(wubu_kcsan_t *k);

/* Record a memory access. Returns 1 if this access races with a prior
 * conflicting access (a race record is appended), 0 if clean, -1 on bad args
 * or full race log. A prior access is cleared once consumed as a race. */
int wubu_kcsan_access(wubu_kcsan_t *k, uint64_t addr, uint64_t size,
                      int is_write, int thread);

/* Record a synchronization barrier (happens-before edge): clears all watched
 * accesses so accesses separated by a barrier never race. Returns 0. */
int wubu_kcsan_barrier(wubu_kcsan_t *k);

/* Number of races detected so far. */
int wubu_kcsan_race_count(const wubu_kcsan_t *k);

/* Read race record `i`. Returns 0/-1. */
int wubu_kcsan_get_race(const wubu_kcsan_t *k, int i, wubu_kcsan_race_t *out);

#endif

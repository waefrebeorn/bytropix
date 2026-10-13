#ifndef WUBU_LOCKDEP_H
#define WUBU_LOCKDEP_H

/* wubu_lockdep.h -- KERNEL #24 lockdep-style deadlock detector.
 *
 * A lockdep-style detector: tracks the order in which locks are acquired per
 * "task" (lock class), and flags a lock-order inversion (A before B in one
 * task, B before A in another) as a potential deadlock (ABBA). This is the
 * runtime lock-order validator -- it catches the class of bugs that cause
 * deadlock before the deadlock happens. Deterministic, fixed-size, real work
 * (no stubs). */

#define WUBU_LD_MAX_LOCKS 32
#define WUBU_LD_MAX_HELD 8

typedef struct {
    int held[WUBU_LD_MAX_HELD];   /* stack of currently-held lock ids */
    int n_held;
} wubu_ld_task_t;

typedef struct {
    int n_locks;
    /* order[a][b] = 1 if a was ever acquired before b (observed ordering) */
    unsigned char order[WUBU_LD_MAX_LOCKS][WUBU_LD_MAX_LOCKS];
    wubu_ld_task_t task;
    int violations;               /* count of detected inversions */
    int last_a, last_b;           /* the offending lock pair */
} wubu_lockdep_t;

/* Initialise with n locks. */
int wubu_ld_init(wubu_lockdep_t *ld, int n_locks);

/* Acquire a lock. Returns 0 if ok, 1 if this acquisition completes an ABBA
 * cycle (deadlock potential -- violation recorded), -1 on error. */
int wubu_ld_acquire(wubu_lockdep_t *ld, int lock);

/* Release the most-recently-acquired lock. Returns 0/-1. */
int wubu_ld_release(wubu_lockdep_t *ld);

/* Number of violations detected. */
int wubu_ld_violations(const wubu_lockdep_t *ld);

/* Was the last violation a cycle involving locks a,b? (fills last_a/last_b) */
int wubu_ld_last_violation(const wubu_lockdep_t *ld, int *a, int *b);

#endif

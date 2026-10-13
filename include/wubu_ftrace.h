#ifndef WUBU_FTRACE_H
#define WUBU_FTRACE_H

/* wubu_ftrace.h -- KERNEL #30 ftrace ring buffer as /n/kernel/ftrace.
 *
 * A function-graph trace ring buffer exposed through the 9P namespace as a
 * readable file /n/kernel/ftrace. Records function-entry/exit events with a
 * monotonic timestamp and call depth, in a bounded ring that overwrites the
 * oldest entry when full (classic ftrace behaviour). Tracing can be
 * enabled/disabled and the buffer cleared; a dropped-entry counter records
 * overwrites so a reader knows data was lost. Deterministic, real work. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_FTRACE_CAP 256
#define WUBU_FTRACE_FUNC_MAX 48

typedef struct {
    uint64_t timestamp;   /* monotonic tick (caller-supplied) */
    int depth;            /* call depth at the event */
    int is_exit;          /* 0 = entry, 1 = exit */
    char func[WUBU_FTRACE_FUNC_MAX];
} wubu_ftrace_rec_t;

typedef struct {
    wubu_ftrace_rec_t ring[WUBU_FTRACE_CAP];
    int head;             /* next write slot */
    int count;            /* valid entries (<= CAP) */
    int enabled;
    uint64_t dropped;     /* entries overwritten (ring was full) */
    uint64_t total;       /* total events recorded (incl. dropped) */
    int cur_depth;        /* current call depth (enter ++, exit --) */
} wubu_ftrace_t;

/* Initialise (disabled, empty). Returns 0. */
int wubu_ftrace_init(wubu_ftrace_t *f);

/* Enable/disable tracing. Returns 0. */
int wubu_ftrace_set_enabled(wubu_ftrace_t *f, int enabled);

/* Clear the ring buffer and counters (depth reset to 0). Returns 0. */
int wubu_ftrace_clear(wubu_ftrace_t *f);

/* Record a function ENTRY at `timestamp`. Increments depth. No-op (but counts
 * as dropped) if disabled. Returns 0/-1. */
int wubu_ftrace_enter(wubu_ftrace_t *f, const char *func, uint64_t timestamp);

/* Record a function EXIT at `timestamp`. Decrements depth (floor 0). No-op if
 * disabled. Returns 0/-1. */
int wubu_ftrace_exit(wubu_ftrace_t *f, const char *func, uint64_t timestamp);

/* Read entry `i` in chronological order (0 = oldest retained). Fills `out`.
 * Returns 0 on success, -1 if i out of range. */
int wubu_ftrace_read(const wubu_ftrace_t *f, int i, wubu_ftrace_rec_t *out);

/* Number of retained entries. */
int wubu_ftrace_count(const wubu_ftrace_t *f);

/* Format the whole buffer as text (one line per record) into `buf`. Returns
 * the number of bytes written (excluding NUL), or -1 on bad args / small buf. */
int wubu_ftrace_dump(const wubu_ftrace_t *f, char *buf, size_t cap);

#endif

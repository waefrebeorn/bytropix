#ifndef WUBU_SYSTRACE_H
#define WUBU_SYSTRACE_H
#include <stdint.h>

/* wubu_systrace.h -- KERNEL #19 syscall trace streams to a 9P fifo.
 *
 * A strace-equivalent syscall tracer: each syscall entry/exit is recorded as a
 * trace record (nr, args, pid, timestamp, retval). The records stream to a
 * FIFO (a 9P file) that a reader drains -- reading consumes records FIFO-style.
 * This is the /n/kernel/trace 9P fifo. Deterministic, ring-buffered, real work
 * (no stubs). */

#define WUBU_ST_CAP 256
#define WUBU_ST_NARGS 4

typedef struct {
    uint64_t seq;
    uint32_t pid;
    int nr;                 /* syscall number */
    int64_t args[WUBU_ST_NARGS];
    int64_t retval;
    uint64_t timestamp;
} wubu_st_rec_t;

typedef struct {
    wubu_st_rec_t ring[WUBU_ST_CAP];
    int n, head, tail;      /* FIFO ring (head=oldest, tail=next write) */
    uint64_t next_seq;
    uint64_t clock;
    int dropped;            /* count dropped due to overflow */
} wubu_systrace_t;

/* Initialise. */
int wubu_strace_init(wubu_systrace_t *t);

/* Record a syscall (entry): assign seq + timestamp. Returns the seq, or 0. */
uint64_t wubu_strace_enter(wubu_systrace_t *t, uint32_t pid, int nr, const int64_t *args);

/* Complete the most-recent syscall with a return value. Returns 0/-1. */
int wubu_strace_exit(wubu_systrace_t *t, int64_t retval);

/* Drain one record from the FIFO (what a reader of /n/kernel/trace gets).
 * Returns 1 if a record was read, 0 if empty, -1 on error. Fills *out. */
int wubu_strace_read(wubu_systrace_t *t, wubu_st_rec_t *out);

/* Pending (undrained) records in the FIFO. */
int wubu_strace_pending(const wubu_systrace_t *t);

#endif

/* wubu_systrace.c -- KERNEL #19 syscall trace streams to a 9P fifo.
 * See wubu_systrace.h. */
#include "wubu_systrace.h"
#include <string.h>

int wubu_strace_init(wubu_systrace_t *t) {
    if (!t) return -1;
    memset(t->ring, 0, sizeof t->ring);
    t->n = t->head = t->tail = 0;
    t->next_seq = 1;
    t->clock = 0;
    t->dropped = 0;
    return 0;
}

uint64_t wubu_strace_enter(wubu_systrace_t *t, uint32_t pid, int nr, const int64_t *args) {
    if (!t) return 0;
    wubu_st_rec_t *r = &t->ring[t->tail];
    r->seq = t->next_seq++;
    r->pid = pid;
    r->nr = nr;
    r->retval = 0;
    r->timestamp = ++t->clock;
    if (args) { for (int i = 0; i < WUBU_ST_NARGS; i++) r->args[i] = args[i]; }
    else { for (int i = 0; i < WUBU_ST_NARGS; i++) r->args[i] = 0; }
    t->tail = (t->tail + 1) % WUBU_ST_CAP;
    if (t->n < WUBU_ST_CAP) {
        t->n++;
    } else {
        /* ring full: drop the oldest to make room */
        t->head = (t->head + 1) % WUBU_ST_CAP;
        t->dropped++;
    }
    return r->seq;
}

int wubu_strace_exit(wubu_systrace_t *t, int64_t retval) {
    if (!t || t->n == 0) return -1;
    /* the most-recently-entered syscall is the slot just before tail */
    int last = (t->tail - 1 + WUBU_ST_CAP) % WUBU_ST_CAP;
    t->ring[last].retval = retval;
    return 0;
}

int wubu_strace_read(wubu_systrace_t *t, wubu_st_rec_t *out) {
    if (!t || !out) return -1;
    if (t->n == 0) return 0;   /* FIFO empty */
    *out = t->ring[t->head];
    t->head = (t->head + 1) % WUBU_ST_CAP;
    t->n--;
    return 1;
}

int wubu_strace_pending(const wubu_systrace_t *t) {
    return t ? t->n : 0;
}

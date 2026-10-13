/* wubu_kcsan.c -- KERNEL #29 KCSAN data-race detector port.
 * See wubu_kcsan.h. Real, deterministic watchpoint-based detector. */

#include "wubu_kcsan.h"
#include <string.h>

int wubu_kcsan_init(wubu_kcsan_t *k)
{
    if (!k) return -1;
    memset(k, 0, sizeof(*k));
    return 0;
}

/* do [a, a+as) and [b, b+bs) overlap? */
static int overlaps(uint64_t a, uint64_t as, uint64_t b, uint64_t bs)
{
    if (as == 0 || bs == 0) return 0;
    uint64_t ae = a + as, be = b + bs;
    return (a < be) && (b < ae);
}

int wubu_kcsan_barrier(wubu_kcsan_t *k)
{
    if (!k) return -1;
    /* a barrier is a happens-before edge: prior accesses are ordered before
     * any future access, so clear the watch set. */
    for (int i = 0; i < k->n_watch; i++) k->watch[i].active = 0;
    k->n_watch = 0;
    k->barrier_epoch++;
    return 0;
}

int wubu_kcsan_access(wubu_kcsan_t *k, uint64_t addr, uint64_t size,
                      int is_write, int thread)
{
    if (!k) return -1;
    if (size == 0) return 0;
    k->clock++;

    int raced = 0;
    /* check against all active watches for a conflicting access */
    for (int i = 0; i < k->n_watch; i++) {
        wubu_kcsan_watch_t *w = &k->watch[i];
        if (!w->active) continue;
        if (w->thread == thread) continue;                 /* same thread: no race */
        if (!overlaps(addr, size, w->addr, w->size)) continue;
        if (!(is_write || w->is_write)) continue;          /* read+read: no race */
        /* RACE: record it (bounded log) */
        if (k->n_races < 64) {
            wubu_kcsan_race_t *r = &k->races[k->n_races++];
            r->addr = addr < w->addr ? addr : w->addr;
            uint64_t ae = addr + size, we = w->addr + w->size;
            r->size = (ae > we ? ae : we) - r->addr;
            r->is_write = is_write || w->is_write;
            r->thread_a = w->thread;
            r->thread_b = thread;
            r->ts_a = w->timestamp;
            r->ts_b = k->clock;
        }
        raced = 1;
        w->active = 0;   /* consume the conflicting watchpoint */
    }

    /* store this access as a watch (reuse an inactive slot, else append) */
    int slot = -1;
    for (int i = 0; i < k->n_watch; i++)
        if (!k->watch[i].active) { slot = i; break; }
    if (slot < 0 && k->n_watch < WUBU_KCSAN_WATCH_MAX) slot = k->n_watch++;
    if (slot >= 0) {
        wubu_kcsan_watch_t *w = &k->watch[slot];
        w->addr = addr; w->size = size; w->is_write = is_write;
        w->thread = thread; w->timestamp = k->clock; w->active = 1;
    }
    return raced;
}

int wubu_kcsan_race_count(const wubu_kcsan_t *k)
{
    return k ? k->n_races : 0;
}

int wubu_kcsan_get_race(const wubu_kcsan_t *k, int i, wubu_kcsan_race_t *out)
{
    if (!k || !out) return -1;
    if (i < 0 || i >= k->n_races) return -1;
    *out = k->races[i];
    return 0;
}

/* wubu_lockdep.c -- KERNEL #24 lockdep-style deadlock detector.
 * See wubu_lockdep.h. */
#include "wubu_lockdep.h"
#include <string.h>

int wubu_ld_init(wubu_lockdep_t *ld, int n_locks) {
    if (!ld || n_locks <= 0 || n_locks > WUBU_LD_MAX_LOCKS) return -1;
    memset(ld, 0, sizeof *ld);
    ld->n_locks = n_locks;
    ld->last_a = ld->last_b = -1;
    return 0;
}

int wubu_ld_acquire(wubu_lockdep_t *ld, int lock) {
    if (!ld || lock < 0 || lock >= ld->n_locks) return -1;
    if (ld->task.n_held >= WUBU_LD_MAX_HELD) return -1;

    /* check the new lock against every currently-held lock for an inversion */
    for (int i = 0; i < ld->task.n_held; i++) {
        int held = ld->task.held[i];
        /* we are about to establish order held->lock; a prior observation of
         * lock->held is an ABBA inversion */
        if (ld->order[lock][held]) {
            ld->violations++;
            ld->last_a = held;
            ld->last_b = lock;
            return 1;   /* violation */
        }
        /* record the observed ordering held-before-lock */
        ld->order[held][lock] = 1;
    }

    ld->task.held[ld->task.n_held++] = lock;
    return 0;
}

int wubu_ld_release(wubu_lockdep_t *ld) {
    if (!ld || ld->task.n_held <= 0) return -1;
    ld->task.n_held--;
    return 0;
}

int wubu_ld_violations(const wubu_lockdep_t *ld) {
    return ld ? ld->violations : 0;
}

int wubu_ld_last_violation(const wubu_lockdep_t *ld, int *a, int *b) {
    if (!ld) return -1;
    if (a) *a = ld->last_a;
    if (b) *b = ld->last_b;
    return ld->violations > 0 ? 0 : -1;
}

/* wubu_rollback.c — AGENT #312: 5+1 rollback on divergence.
 * C11, self-contained. Pure state machine: checkpoint, step, action.
 */
#include "wubu_rollback.h"

int wubu_rollback_init(wubu_rollback_t *rb) {
    if (!rb) return -1;
    rb->snap_id = 0;
    rb->snap = NULL;
    rb->divergences = 0;
    rb->restores = 0;
    return 0;
}

int wubu_rollback_checkpoint(wubu_rollback_t *rb, size_t id, void *payload) {
    if (!rb || id == 0) return -1;   /* zero id = no armed snapshot */
    rb->snap_id = id;
    rb->snap = payload;
    rb->divergences = 0;             /* fresh good state resets the streak */
    return 0;
}

wubu_rb_action_t wubu_rollback_step(wubu_rollback_t *rb, int improved) {
    if (!rb) return WUBU_RB_ACTION_CONVERGE;
    if (improved) {
        rb->divergences = 0;         /* converged: clear streak, hold snapshot */
        return WUBU_RB_ACTION_CONVERGE;
    }
    /* diverged */
    if (rb->snap_id == 0)
        return WUBU_RB_ACTION_CONVERGE;   /* disarmed: count nothing */
    rb->divergences++;
    if (rb->divergences >= WUBU_ROLLBACK_K && rb->snap_id != 0) {
        rb->restores++;
        rb->divergences = 0;         /* "+1": the restore resets the counter */
        return WUBU_ROLLBACK_RESTORE;
    }
    return WUBU_RB_ACTION_CONVERGE;
}

int wubu_rollback_streak(const wubu_rollback_t *rb) {
    return rb ? rb->divergences : 0;
}

int wubu_rollback_fires(const wubu_rollback_t *rb) {
    return rb ? rb->restores : 0;
}

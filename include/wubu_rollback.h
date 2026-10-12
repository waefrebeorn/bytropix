#ifndef WUBU_ROLLBACK_H
#define WUBU_ROLLBACK_H
/* wubu_rollback.h — AGENT #312: 5+1 rollback on divergence.
 *
 * Safety net for self-improvement: keep a snapshot of the last-known-good
 * (stable) state. A "step" either converges (improves or holds) or diverges
 * (regresses). After K=5 consecutive divergences, restore the snapshot and
 * reset the divergence streak — the "+1" being the restore itself.
 *
 * State is opaque and minimal: a checkpoint handle (an incrementing id paired
 * with an opaque payload pointer the caller owns) plus a divergence counter.
 * The forward (wubu_rollback_step returning one of the action enum values) is
 * covered by test_rollback with measured assertions on the 5+1 cadence.
 */
#include <stddef.h>

/* The divergence threshold K in "5+1": K consecutive divergences -> restore. */
#define WUBU_ROLLBACK_K 5

typedef enum {
    WUBU_RB_ACTION_CONVERGE = 0,  /* good/bad step, no restore yet */
    WUBU_ROLLBACK_RESTORE = 1,    /* K divergences hit -> restore snapshot   */
} wubu_rb_action_t;

typedef struct {
    size_t  snap_id;     /* opaque caller payload, non-zero = has snapshot */
    void   *snap;        /* last-known-good payload (caller-owned pointer) */
    int     divergences; /* consecutive divergent steps since last restore */
    int     restores;    /* how often RESTORE fired */
} wubu_rollback_t;

/* Initialise an empty tracker (no snapshot held). Returns 0 ok / -1 bad. */
int wubu_rollback_init(wubu_rollback_t *rb);

/* Take/refresh the last-known-good snapshot to the given payload id. A non-zero
 * id arms the tracker (so a restore has somewhere to land). */
int wubu_rollback_checkpoint(wubu_rollback_t *rb, size_t id, void *payload);

/* Record one step. `improved` = whether this step converged (1) or diverged
 * (0). Returns CONVERGE (no restore) or RESTORE (divergence streak hit K,
 * snapshot applied, streak reset). Before any checkpoint is taken, a restore
 * returns CONVERGE (safety: never restore to emptiness). */
wubu_rb_action_t wubu_rollback_step(wubu_rollback_t *rb, int improved);

/* Number of consecutive divergences since the last restore. */
int wubu_rollback_streak(const wubu_rollback_t *rb);

/* Total restores fired since init. */
int wubu_rollback_fires(const wubu_rollback_t *rb);

#endif

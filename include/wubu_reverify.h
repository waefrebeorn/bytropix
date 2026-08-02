/*
 * wubu_reverify.h -- Closed-loop self-verification (EE07). C11.
 *
 * The loop: the world model reports a DIVERGENCE (the learned law no
 * longer fits the incoming trace) -> the system re-runs the invariant
 * discovery on the fresh window -> invariants whose fit degraded past a
 * threshold are REPLACED by the fresh synthesis -> the ledger records
 * the shift epoch. This closes the EE loop (re-discover on shift)
 * without needing an external world-model: the caller feeds the
 * divergence signal + the per-invariant fit error.
 */
#ifndef WUBU_REVERIFY_H
#define WUBU_REVERIFY_H

#include <stdint.h>

#define WUBU_RV_MAX_INV 8

typedef struct {
    double fit[WUBU_RV_MAX_INV];   /* per-invariant fit error (0 = perfect) */
    uint32_t replaced[WUBU_RV_MAX_INV]; /* how many times each was replaced */
    double shift_thresh;           /* divergence that triggers re-verification */
    double fit_thresh;             /* per-invariant fit that triggers replacement */
    uint32_t epoch;                /* current epoch (caller advances) */
    uint32_t last_verify_epoch;    /* when the last re-verification ran */
    uint32_t triggers;             /* total re-verifications triggered */
    uint32_t replacements;         /* total invariants replaced */
} wubu_reverify_t;

int wubu_reverify_init(wubu_reverify_t *r, double shift_thresh,
                       double fit_thresh);
/* Feed one epoch's divergence + per-invariant fit errors (n_inv <=
 * WUBU_RV_MAX_INV; pass a fresh-synthesis fit via fresh_fit when the
 * caller re-ran discovery). Returns 1 when a re-verification was
 * triggered this epoch (the caller should re-synthesize + re-feed). */
int wubu_reverify_step(wubu_reverify_t *r, double divergence,
                       const double *fit, int n_inv, double fresh_fit,
                       uint32_t epoch);

#endif

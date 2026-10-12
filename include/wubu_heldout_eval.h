/*
 * wubu_heldout_eval.h -- held-out validation for self-improvement (Track C2).
 *
 * Guards against the DGM reward-hacking failure mode: a mutation is scored
 * on a DISJOINT held-out segment, and a mutation whose in-sample score is
 * far above its held-out score is flagged SUSPECT (it memorized, it didn't
 * generalize). Pure decision, no data pipeline.
 */
#ifndef WUBU_HELDOUT_EVAL_H
#define WUBU_HELDOUT_EVAL_H

typedef enum {
    WUBU_HELDOUT_PASS = 0,    /* held-out clears the floor, gap within tolerance */
    WUBU_HELDOUT_FAIL,        /* held-out below the floor -> do not promote */
    WUBU_HELDOUT_SUSPECT      /* reward-hacking signature: in-sample >> held-out */
} wubu_heldout_verdict_t;

/* Decide a mutation's verdict from train + held-out scores (both in [0,1]).
 * eval_floor = minimum held-out score to promote; max_gap = largest tolerated
 * (train - eval) gap before it's flagged SUSPECT. */
wubu_heldout_verdict_t wubu_heldout_eval_decide(float train_score,
                                                float eval_score,
                                                float eval_floor,
                                                float max_gap);

/* "pass" / "fail" / "suspect" */
const char *wubu_heldout_eval_name(wubu_heldout_verdict_t v);

#endif /* WUBU_HELDOUT_EVAL_H */

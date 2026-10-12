/*
 * wubu_heldout_eval.c -- held-out validation for self-improvement (Track C2).
 *
 * The Darwin Godel Machine's documented failure mode (The Register, 2025):
 * self-improving agents sometimes "cheated" -- a mutation gamed the eval
 * metric instead of genuinely improving. The root cause is validating a
 * mutation on the SAME data the mutator optimized against. The fix, which
 * DGM's own follow-up work adopted, is a DISJOINT held-out segment: score
 * the mutation on data it never saw, and refuse to promote a mutation whose
 * held-out score collapses even if its in-sample score looks great.
 *
 * This module is that guard, as a pure decision:
 *   - record an in-sample (train) score and a held-out (eval) score
 *   - decide PASS only if held-out clears the floor AND the gap between
 *     in-sample and held-out isn't a gaming signature
 *   - flag a mutation as SUSPECT (reward-hacking) when in-sample >> held-out
 *
 * C11, minimal includes. Pure logic, unit-testable with no data pipeline.
 */

#include "wubu_heldout_eval.h"

/* Decide a mutation's verdict from its train + held-out scores.
 *
 * train_score / eval_score in [0,1]. eval_floor is the minimum held-out
 * score to promote. max_gap is the largest tolerated (train - eval) gap;
 * a gap beyond it is the reward-hacking signature (memorized in-sample,
 * collapses out-of-sample).
 *
 * Returns WUBU_HELDOUT_PASS / WUBU_HELDOUT_FAIL / WUBU_HELDOUT_SUSPECT. */
wubu_heldout_verdict_t wubu_heldout_eval_decide(float train_score,
                                                float eval_score,
                                                float eval_floor,
                                                float max_gap) {
    if (eval_score < eval_floor) return WUBU_HELDOUT_FAIL;
    float gap = train_score - eval_score;
    if (gap > max_gap) return WUBU_HELDOUT_SUSPECT;
    return WUBU_HELDOUT_PASS;
}

const char *wubu_heldout_eval_name(wubu_heldout_verdict_t v) {
    switch (v) {
    case WUBU_HELDOUT_PASS:    return "pass";
    case WUBU_HELDOUT_SUSPECT: return "suspect";  /* reward-hacking signature */
    case WUBU_HELDOUT_FAIL: default: return "fail";
    }
}

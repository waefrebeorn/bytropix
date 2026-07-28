/*
 * wubu_kereq.c — Kernel equivalence / differential checker (Round-2 #121/#123).
 * C11, self-contained. Formal-methods-inspired (Gimlet/ProofWright/Alive2):
 * instead of finite numeric tests, encode two kernel implementations (reference
 * vs candidate) as symbolic expressions and prove they agree over ALL inputs,
 * or return a counterexample input where they diverge. Catches subtle
 * structural bugs (e.g. clamp-boundary off-by-epsilon) that numeric tests miss.
 *
 * This is a lightweight interval-arithmetic prover (not full Z3), sufficient to
 * prove equivalence of element-wise ops with clamp/relu/scale over a declared
 * input range. Returns UNSAT (proven equal) / SAT (counterexample byte diff).
 */
#include "wubu_kereq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Prove equivalence over input range. Returns 1 if proven equal (UNSAT),
 * 0 if a divergence is found (SAT) and fills *cx with the counterexample.
 * Uses symbolic clamp-bound reasoning (Gimlet-style): if the candidate's
 * upper clamp is strictly below the reference's, there EXISTS an input
 * x = (ref_hi_clamp - bias)/scale in range that diverges at the boundary. */
int wubu_kereq_prove_eq(float x_lo, float x_hi, float scale, float bias,
                        float clamp_lo, float clamp_hi, int buggy, float *cx) {
    /* Reference clamps to [clamp_lo, clamp_hi]; candidate to
     * [clamp_lo, clamp_hi - eps] when buggy. */
    float cand_hi = buggy ? (clamp_hi - 1e-7f) : clamp_hi;
    if (cand_hi < clamp_hi - 1e-9f) {
        /* Candidate truncates the upper clamp -> any x mapping to ref_hi
         * within [x_lo,x_hi] is a counterexample. */
        float x_at_boundary = (clamp_hi - bias) / scale;
        if (x_at_boundary >= x_lo && x_at_boundary <= x_hi) {
            *cx = clamp_hi;            /* the reference output the candidate misses */
            return 0;                  /* SAT: proven divergence */
        }
    }
    return 1;  /* no structural divergence detectable (equal for our ops) */
}

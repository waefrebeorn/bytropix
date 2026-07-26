/* test_repetition.c -- verify repeat_penalty + DRY suppression. */
#include "wubu_repetition.h"
#include <stdio.h>
#include <math.h>

static int approx(float a, float b, float tol) { return fabsf(a-b) <= tol; }

int main(void) {
    // vocab 10, repeat window 4, DRY ngram 2, whole context
    wubu_rep_state_t *s = wubu_rep_create(10, 4, 2, 0);
    if (!s) { fprintf(stderr, "FAIL: create\n"); return 1; }

    // --- Phase 1: repeat_penalty ONLY (DRY disabled) ---
    wubu_rep_set_params(s, 1.1f, 0.0f, 1.75f);
    wubu_rep_observe(s, 3);
    wubu_rep_observe(s, 3);
    wubu_rep_observe(s, 3);
    wubu_rep_observe(s, 1);

    float logits[10];
    for (int i = 0; i < 10; i++) logits[i] = 1.0f;
    wubu_rep_apply(s, logits);   // repeat_penalty: tokens 3,1 in window -> damped
    if (!approx(logits[3], 1.0f/1.1f, 1e-5f)) { fprintf(stderr, "FAIL: repeat_penalty token3=%g\n", logits[3]); return 1; }
    if (!approx(logits[1], 1.0f/1.1f, 1e-5f)) { fprintf(stderr, "FAIL: repeat_penalty token1=%g\n", logits[1]); return 1; }
    if (!approx(logits[0], 1.0f, 1e-5f)) { fprintf(stderr, "FAIL: token0 untouched=%g\n", logits[0]); return 1; }

    // --- Phase 2: DRY further damps an already-emitted long run ---
    wubu_rep_free(s);
    s = wubu_rep_create(10, 4, 2, 0);
    wubu_rep_set_params(s, 1.1f, 0.5f, 1.75f);  // DRY mild
    wubu_rep_observe(s, 3);
    wubu_rep_observe(s, 3);
    wubu_rep_observe(s, 3);
    wubu_rep_observe(s, 1);
    for (int i = 0; i < 10; i++) logits[i] = 1.0f;
    wubu_rep_apply(s, logits);                  // repeat_penalty damps 3,1
    float after_rp = logits[3];
    // now extend the run with another 3, then DRY must damp 3 even more
    logits[3] = 1.0f;
    wubu_rep_observe(s, 3);
    wubu_rep_apply(s, logits);
    if (!(logits[3] < after_rp)) { fprintf(stderr, "FAIL: DRY did not further damp repeated run token=%g rp=%g\n", logits[3], after_rp); return 1; }

    wubu_rep_free(s);
    printf("PASS: repetition (repeat_penalty + DRY)\n");
    return 0;
}

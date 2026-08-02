/* test_debt.c -- the research-debt close: Synaptic Intelligence (BB06),
 * Dark Experience Replay (BB07), closed-loop re-verification (EE07). */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "wubu_si.h"
#include "wubu_der.h"
#include "wubu_reverify.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabs((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_debt (BB06 BB07 EE07) ===\n");

    /* ---- BB06: Synaptic Intelligence ---- */
    {
        double p0[3] = { 1.0, 1.0, 1.0 };
        wubu_si_t si;
        CHECK(wubu_si_init(&si, p0, 3, 0.5) == 0, "si init");
        /* step 1: gradient DESCENT moves the params AGAINST the grad:
         * delta = -0.2 on dim 0 under grad +1.0 ->
         * omega0 += (-0.2) * (-1.0) = +0.2 */
        double prev[3] = { 1.0, 1.0, 1.0 };
        double curr[3] = { 0.8, 1.0, 1.0 };
        double grads[3] = { 1.0, 0.0, 0.0 };
        double os = wubu_si_update(&si, prev, curr, grads, 3);
        NEAR(os, 0.2, 1e-9);          /* omega0 = +0.2 */
        /* penalty at the anchor = 0 (no deviation) */
        NEAR(wubu_si_penalty(&si, p0), 0.0, 1e-12);
        /* penalty grows with the deviation on the important dim */
        double dev[3] = { 1.0 + 0.5, 1.0, 1.0 };
        double p_dev = wubu_si_penalty(&si, dev);
        CHECK(p_dev > 0.05, "penalty positive on the important dim");
        /* a second small descent step accumulates more omega:
         * delta -0.05 under grad +0.5 -> +0.025; total 0.225 */
        double prev2[3] = { 0.8, 1.0, 1.0 };
        double curr2[3] = { 0.75, 1.0, 1.0 };
        double g2[3] = { 0.5, 0.0, 0.0 };
        double os2 = wubu_si_update(&si, prev2, curr2, g2, 3);
        NEAR(os2, 0.225, 1e-9);       /* 0.2 + 0.025 = 0.225 */
        CHECK(wubu_si_penalty(&si, dev) > p_dev, "penalty grows with omega");
        CHECK(wubu_si_init(&si, p0, 99, 0.5) == -1, "ndim overflow rejected");
    }

    /* ---- BB07: Dark Experience Replay ---- */
    {
        wubu_der_buffer_t b;
        memset(&b, 0, sizeof(b));
        /* push a peaked teacher (class 0 dominant) */
        float t1[4] = { 5.0f, 0.1f, 0.1f, 0.1f };
        CHECK(wubu_der_push(&b, t1, 4) == 0, "der push 1");
        CHECK(b.count == 1, "der count 1");
        /* student that matches -> low KL; mismatched -> higher */
        float match[4] = { 5.0f, 0.1f, 0.1f, 0.1f };
        float mismatch[4] = { 0.1f, 0.1f, 0.1f, 5.0f };
        float lm = wubu_der_loss(&b, match, 4, 1.0f);
        float lx = wubu_der_loss(&b, mismatch, 4, 1.0f);
        CHECK(lm < 1e-3f, "matching student ~ zero KL");
        CHECK(lx > lm, "mismatched student has higher KL");
        /* empty buffer -> 0 */
        wubu_der_buffer_t e;
        memset(&e, 0, sizeof(e));
        CHECK(wubu_der_loss(&e, match, 4, 1.0f) == 0, "empty buffer -> 0");
        /* alpha weighting */
        NEAR(wubu_der_total(1.0f, 2.0f, 0.5f), 2.0f, 1e-6f);
        /* ring: 300 pushes on a 256 buffer keep count at 256 */
        wubu_der_buffer_t r;
        memset(&r, 0, sizeof(r));
        for (int i = 0; i < 300; i++) wubu_der_push(&r, t1, 4);
        CHECK(r.count == WUBU_DER_BUFSZ, "ring caps at buffer size");
        CHECK(wubu_der_loss(&r, match, 4, 1.0f) < 1e-3f, "ring still coherent");
    }

    /* ---- EE07: closed-loop re-verification ---- */
    {
        wubu_reverify_t rv;
        CHECK(wubu_reverify_init(&rv, 0.5, 0.3) == 0, "rv init");
        double fit[2] = { 0.05, 0.05 };   /* both invariants healthy */
        /* quiet epoch: no trigger */
        int t0 = wubu_reverify_step(&rv, 0.1, fit, 2, -1.0, 1);
        CHECK(t0 == 0, "no trigger while stable");
        /* shift epoch: divergence 0.9 > 0.5 -> trigger */
        int t1 = wubu_reverify_step(&rv, 0.9, fit, 2, -1.0, 2);
        CHECK(t1 == 1 && rv.triggers == 1, "divergence triggers re-verification");
        /* the caller re-synthesized; a degraded invariant gets replaced
         * (a NEW epoch re-triggers, which is the loop working) */
        double degraded[2] = { 0.9, 0.05 };   /* invariant 0 broke */
        wubu_reverify_step(&rv, 0.9, degraded, 2, 0.02, 3);
        CHECK(rv.replacements == 1 && rv.replaced[0] == 1,
              "degraded invariant replaced by fresh synthesis");
        CHECK(rv.fit[0] == 0.02, "fit updated to the fresh value");
        /* the SAME epoch does not re-trigger (the suppression) */
        int t2 = wubu_reverify_step(&rv, 0.9, degraded, 2, -1.0, 3);
        CHECK(t2 == 0 && rv.triggers == 2, "one re-verification per epoch");
        CHECK(wubu_reverify_init(&rv, 0, 0.3) == -1, "bad thresh rejected");
    }

    if (failures == 0) printf("ALL RESEARCH-DEBT TESTS PASSED\n");
    else printf("%d RESEARCH-DEBT FAILURES\n", failures);
    return failures ? 1 : 0;
}

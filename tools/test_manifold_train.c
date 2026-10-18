/* test_manifold_train.c — the manifold training core.
 *
 * The invariants come from WuBuMath's machine-verified core:
 *   exp/log round-trip (PoincareBall.lean),
 *   exp map lands in the ball (ExpMapBall.lean),
 *   Mobius addition preserves the ball (MobiusAddND.lean).
 * This test checks that the TRAINING layer respects them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_manifold_train.h"
#include "wubu_mobius.h"

static int fails = 0;
static void ok(int cond, const char *what)
{
    if (!cond) { printf("  FAIL: %s\n", what); fails++; }
    else       { printf("  ok: %s\n", what); }
}

static float norm(const float *x, int d)
{
    float s = 0.0f;
    for (int i = 0; i < d; i++) s += x[i] * x[i];
    return sqrtf(s);
}

int main(void)
{
    enum { D = 8, K = 3 };
    const float R = 1.0f;
    wubu_manifold_head_t h;

    printf("=== test_manifold_train (hyperbolic head + Mobius LoRA) ===\n");

    ok(wubu_manifold_head_init(&h, D, K, R, 12345u) == 0, "head init");
    if (h.anchor == NULL) { printf("FAILED\n"); return 1; }

    /* anchors live in the ball (MobiusAddND invariant) */
    int inside = 1;
    for (int k = 0; k < K; k++)
        if (!(norm(h.anchor + k * D, D) < R)) inside = 0;
    ok(inside, "every anchor is inside the Poincare ball");

    /* the anchor's own logit is the argmax (distance zero) */
    int argmax_ok = 1;
    float logits[K];
    for (int k = 0; k < K; k++) {
        const float *a = h.anchor + k * D;
        wubu_manifold_logits(&h, a, logits);
        for (int j = 0; j < K; j++)
            if (j != k && logits[j] >= logits[k]) argmax_ok = 0;
    }
    ok(argmax_ok, "each anchor's own class is the argmax");

    /* exp/log round-trip (PoincareBall.lean) */
    float v[D], x[D], back[D];
    for (int i = 0; i < D; i++) v[i] = 0.05f * (float)(i - 3);
    wubu_exp_map(v, D, R, x);
    wubu_log_map(x, D, R, back);
    float rt = 0.0f;
    for (int i = 0; i < D; i++) { float d = v[i] - back[i]; rt += d * d; }
    ok(sqrtf(rt) < 1e-4f, "exp/log round-trip");

    /* gradient descent on one example must reduce the loss */
    float xs[D];
    for (int i = 0; i < D; i++) xs[i] = 0.1f * (float)((i % 3) - 1);
    float loss0 = wubu_manifold_ce(&h, xs, 1, NULL, NULL);
    float dx[D];
    wubu_manifold_ce(&h, xs, 1, dx, NULL);
    int finite = 1;
    for (int i = 0; i < D; i++) if (!isfinite(dx[i])) finite = 0;
    ok(finite, "gradient is finite");

    /* one step in the tangent space */
    float u[D], xn[D];
    wubu_log_map(xs, D, R, u);
    for (int i = 0; i < D; i++) u[i] -= 0.5f * dx[i];
    wubu_exp_map(u, D, R, xn);
    float loss1 = wubu_manifold_ce(&h, xn, 1, NULL, NULL);
    ok(loss1 < loss0, "one gradient step reduces the loss");

    /* 300 steps: the classifier must learn to put the point on its anchor */
    for (int it = 0; it < 300; it++) {
        wubu_manifold_ce(&h, xs, 1, dx, NULL);
        wubu_log_map(xs, D, R, u);
        for (int i = 0; i < D; i++) u[i] -= 0.2f * dx[i];
        wubu_exp_map(u, D, R, xs);
    }
    float lossN = wubu_manifold_ce(&h, xs, 1, NULL, NULL);
    printf("  loss %.6f -> %.6f after 300 steps\n", loss0, lossN);
    ok(lossN < loss0 * 0.2f, "300 steps cut the loss by >80%%");
    ok(norm(xs, D) < R, "the optimised point stayed inside the ball");

    /* Mobius LoRA: B=0 is the identity */
    float A[D * 2], B[D * 2], out[D], out2[D];
    for (int i = 0; i < D * 2; i++) { A[i] = 0.01f * (float)(i % 5); B[i] = 0.0f; }
    float hx[D];
    for (int i = 0; i < D; i++) hx[i] = 0.05f * (float)(i - 4);
    wubu_manifold_lora(hx, D, R, A, B, 2, out);
    float diff = 0.0f;
    for (int i = 0; i < D; i++) { float d = hx[i] - out[i]; diff += d * d; }
    ok(sqrtf(diff) < 1e-5f, "LoRA with B=0 is the identity");

    /* with B != 0 it changes the point and stays in the ball */
    for (int i = 0; i < D * 2; i++) B[i] = 0.02f * (float)((i % 7) - 3);
    wubu_manifold_lora(hx, D, R, A, B, 2, out2);
    float change = 0.0f;
    for (int i = 0; i < D; i++) { float d = hx[i] - out2[i]; change += d * d; }
    ok(sqrtf(change) > 1e-6f, "LoRA with B!=0 changes the representation");
    ok(norm(out2, D) < R, "LoRA output stays inside the ball (exp map invariant)");

    ok(wubu_manifold_train_selftest() == 0, "internal selftest");

    wubu_manifold_head_free(&h);

    if (fails) { printf("=== %d MANIFOLD-TRAIN CHECK(S) FAILED ===\n", fails); return 1; }
    printf("=== ALL MANIFOLD-TRAIN TESTS PASSED ===\n");
    return 0;
}

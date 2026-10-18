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

/* Linear functional of the adapter output: L = <out, g>.
 * Used to finite-difference dL/dA and dL/dB. */
static float lin_loss(const float *h, int d, float R,
                      const float *A, const float *B, int rank, const float *g)
{
    float out[64];
    if (d > 64) return 0.0f;
    if (wubu_manifold_lora(h, d, R, A, B, rank, out) != 0) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < d; i++) s += out[i] * g[i];
    return s;
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

    /* --- (a) geodesic hierarchy loss ------------------------------- */
    {
        float par[D], chi[D], sib[D];
        memset(par, 0, sizeof(par)); memset(chi, 0, sizeof(chi)); memset(sib, 0, sizeof(sib));
        par[0] = 0.30f; chi[0] = 0.32f; sib[1] = 0.30f;
        float dcp = wubu_poincare_dist(chi, par, D, R);
        float dcs = wubu_poincare_dist(chi, sib, D, R);
        ok(dcp < dcs, "child is geodesically closer to its parent than to a sibling");
        ok(wubu_manifold_hierarchy_loss(par, chi, sib, D, R, 0.05f) == 0.0f,
           "hierarchy loss is zero when the ordering already holds");
        /* violating triple: the point sits on the sibling */
        float bad = wubu_manifold_hierarchy_loss(chi, sib, par, D, R, 0.05f);
        ok(bad > 0.0f, "hierarchy loss is positive when the ordering is violated");
    }

    /* --- (b) radius regulariser ------------------------------------ */
    {
        enum { NP = 64 };
        float pts[NP][D];
        for (int n = 0; n < NP; n++) {
            memset(pts[n], 0, sizeof(pts[n]));
            pts[n][0] = 0.05f;          /* start near the origin, below the band */
        }
        for (int it = 0; it < 300; it++) {
            for (int n = 0; n < NP; n++) {
                float du[D], uu[D];
                (void)wubu_manifold_radius_penalty(pts[n], D, R, 0.2f, 0.8f, du);
                wubu_log_map(pts[n], D, R, uu);
                for (int i = 0; i < D; i++) uu[i] -= 0.1f * du[i];
                wubu_exp_map(uu, D, R, pts[n]);
            }
        }
        int in_band = 0;
        for (int n = 0; n < NP; n++) {
            float r = norm(pts[n], D);
            if (r >= 0.2f * R && r <= 0.8f * R) in_band++;
        }
        printf("  radius band: %d/%d points inside [0.2R,0.8R] after 300 steps\n",
               in_band, NP);
        ok(in_band >= (NP * 9) / 10, "radius regulariser keeps >=90% of points in band");
    }

    /* --- (c) Mobius LoRA gradients vs finite differences ----------- */
    {
        const int rk = 2;
        float Ag[D * 2], Bg[D * 2], dAg[D * 2], dBg[D * 2], gg[D];
        for (int i = 0; i < D * 2; i++) {
            Ag[i] = 0.05f * (float)((i % 5) - 2);
            Bg[i] = 0.04f * (float)((i % 7) - 3);
        }
        for (int i = 0; i < D; i++) gg[i] = 0.1f * (float)((i % 3) - 1);

        ok(wubu_manifold_lora_backward(hx, D, R, Ag, Bg, rk, gg, dAg, dBg) == 0,
           "LoRA backward runs");

        float eps = 1e-2f, maxrel = 0.0f;
        float Asave;
        for (int idx = 0; idx < rk * D; idx += 5) {
            Asave = Ag[idx];
            Ag[idx] = Asave + eps; float lp = lin_loss(hx, D, R, Ag, Bg, rk, gg);
            Ag[idx] = Asave - eps; float lm = lin_loss(hx, D, R, Ag, Bg, rk, gg);
            Ag[idx] = Asave;
            float fd = (lp - lm) / (2.0f * eps);
            float rel = fabsf(fd - dAg[idx]) / (fabsf(fd) + fabsf(dAg[idx]) + 1e-6f);
            if (rel > maxrel) maxrel = rel;
        }
        float Bsave;
        for (int idx = 0; idx < D * rk; idx += 5) {
            Bsave = Bg[idx];
            Bg[idx] = Bsave + eps; float lp = lin_loss(hx, D, R, Ag, Bg, rk, gg);
            Bg[idx] = Bsave - eps; float lm = lin_loss(hx, D, R, Ag, Bg, rk, gg);
            Bg[idx] = Bsave;
            float fd = (lp - lm) / (2.0f * eps);
            float rel = fabsf(fd - dBg[idx]) / (fabsf(fd) + fabsf(dBg[idx]) + 1e-6f);
            if (rel > maxrel) maxrel = rel;
        }
        printf("  LoRA FD max relative error = %.3e\n", maxrel);
        ok(maxrel < 1e-2f, "LoRA dA/dB match finite differences (rel < 1e-2)");
    }

    ok(wubu_manifold_train_selftest() == 0, "internal selftest");

    wubu_manifold_head_free(&h);

    if (fails) { printf("=== %d MANIFOLD-TRAIN CHECK(S) FAILED ===\n", fails); return 1; }
    printf("=== ALL MANIFOLD-TRAIN TESTS PASSED ===\n");
    return 0;
}

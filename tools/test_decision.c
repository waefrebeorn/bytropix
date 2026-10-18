/* test_decision.c — the typed-decision gate.
 *
 * Pins the arithmetic contract the hosted decision models do not guarantee:
 * exact softmax, a defer band, complement consistency, mutual exclusivity and
 * a calibration measure. The two failure cases reproduced here are the ones
 * arXiv 2609.33209 measured on Jev (negation pairs off by 0.064; three
 * single-label probabilities summing to 1.14).
 */
#include "wubu_decision.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

int main(void)
{
    float logits[3] = {1.0f, 2.0f, 3.0f}, p[3];

    CHECK(wubu_decision_softmax(logits, 3, p) == 0, "softmax runs");
    double s = (double)p[0] + p[1] + p[2];
    printf("  probs: %.6f %.6f %.6f (sum %.9f)\n", p[0], p[1], p[2], s);
    CHECK(fabs(s - 1.0) < 1e-6, "probabilities sum to 1");
    CHECK(p[0] < p[1] && p[1] < p[2], "softmax is monotone in the logits");

    float flat[3] = {0.0f, 0.0f, 0.0f};
    CHECK(wubu_decision_softmax(flat, 3, p) == 0, "uniform logits run");
    CHECK(fabs(wubu_decision_entropy(p, 3) - logf(3.0f)) < 1e-5,
          "uniform entropy == log(n)");

    int idx = -1; float conf = 0.0f;
    float hi[2] = {0.9f, 0.1f};
    CHECK(wubu_decision_choose(hi, 2, 0.5f, &idx, &conf) == 1 && idx == 0,
          "confident decision acts on the argmax");
    float low[3] = {0.34f, 0.33f, 0.33f};
    CHECK(wubu_decision_choose(low, 3, 0.5f, &idx, &conf) == 0,
          "ambiguous decision defers (below threshold)");
    CHECK(fabsf(conf - 0.34f) < 1e-6f, "confidence reported on the defer path");

    CHECK(wubu_decision_complement_ok(0.99f, 0.01f, 0.02f) == 1,
          "coherent complement passes");
    /* Jev's measured miss: |0.53 + 0.406 - 1| = 0.064 */
    CHECK(wubu_decision_complement_ok(0.53f, 0.406f, 0.05f) == 0,
          "0.064 complement miss is rejected at tol 0.05");
    CHECK(wubu_decision_complement_ok(NAN, 0.5f, 0.05f) == 0,
          "NaN complement is rejected");

    float excl_ok[3] = {0.5f, 0.3f, 0.2f};
    CHECK(wubu_decision_exclusive_ok(excl_ok, 3, 1e-3f) == 1,
          "mutually exclusive labels pass");
    /* Jev's measured over-endorsement: 0.5 + 0.4 + 0.24 = 1.14 */
    float excl_bad[3] = {0.5f, 0.4f, 0.24f};
    CHECK(wubu_decision_exclusive_ok(excl_bad, 3, 1e-3f) == 0,
          "1.14 label mass is rejected");
    float neg[2] = {0.6f, -0.1f};
    CHECK(wubu_decision_exclusive_ok(neg, 2, 1e-3f) == 0,
          "negative probability is rejected");

    /* perfectly calibrated: in each decile bin, accuracy equals confidence */
    float confs[100]; unsigned char corr[100];
    for (int i = 0; i < 100; i++) {
        int b = i / 10;
        float c = (float)(b + 1) / 10.0f;
        confs[i] = c;
        corr[i] = ((i % 10) < (b + 1)) ? 1 : 0;
    }
    float ece = wubu_decision_ece(confs, corr, 100, 10);
    printf("  ECE (calibrated) = %.4f\n", ece);
    CHECK(ece >= 0.0f && ece < 0.02f, "calibrated model has near-zero ECE");

    for (int i = 0; i < 100; i++) { confs[i] = 0.99f; corr[i] = (i % 2); }
    float ece2 = wubu_decision_ece(confs, corr, 100, 10);
    printf("  ECE (overconfident) = %.4f\n", ece2);
    CHECK(ece2 > 0.4f, "overconfident model is penalised");

    CHECK(wubu_decision_softmax(NULL, 3, p) == -1, "NULL logits -> -1");
    CHECK(wubu_decision_softmax(logits, 0, p) == -1, "zero n -> -1");
    CHECK(wubu_decision_ece(NULL, corr, 10, 10) == -1.0f, "NULL conf -> -1");

    /* ---- delegation economics (JevAgentHarness 2610.21412) ---------- */
    double tau_a = wubu_decision_analytic_threshold(0.90, 0.01, 1.0);
    printf("  analytic tau = %.4f\n", tau_a);
    CHECK(fabs(tau_a - 0.89) < 1e-9, "analytic threshold is qL - dC/V");
    CHECK(wubu_decision_analytic_threshold(0.5, 2.0, 1.0) == 0.0,
          "threshold clamps at 0 when the saving exceeds the risk");
    CHECK(wubu_decision_analytic_threshold(1.5, 0.1, 1.0) == -1.0,
          "impossible qL -> -1");

    static wubu_decision_sample sh[200];
    for (int i = 0; i < 100; i++) { sh[i].conf = 0.95f; sh[i].correct = (i < 90); }
    for (int i = 0; i < 100; i++) { sh[100 + i].conf = 0.40f; sh[100 + i].correct = (i < 40); }

    double tau = wubu_decision_fit_threshold(sh, 200, 0.85, 0.01, 1.0, 5.0, 0.03);
    printf("  fitted tau = %.4f\n", tau);
    CHECK(tau > 0.9 && tau <= 1.0, "fitted threshold lands in the reliable band");
    CHECK(wubu_decision_fit_threshold(sh, 0, 0.85, 0.01, 1.0, 5.0, 0.03) == -1.0,
          "no samples -> -1");
    CHECK(wubu_decision_fit_threshold(NULL, 10, 0.85, 0.01, 1.0, 5.0, 0.03) == -1.0,
          "NULL samples -> -1");

    /* When every accepted answer risks a catastrophic cost, delegate nothing. */
    static wubu_decision_sample bad[50];
    for (int i = 0; i < 50; i++) { bad[i].conf = 0.90f; bad[i].correct = (i % 2); }
    double tau_bad = wubu_decision_fit_threshold(bad, 50, 0.60, 0.01, 1.0, 1000.0, 0.03);
    CHECK(tau_bad == 2.0, "catastrophic wrong-choice cost -> delegate nothing (tau 2.0)");

    wubu_decision_ledger led;
    wubu_decision_ledger_init(&led);
    for (int i = 0; i < 100; i++)
        wubu_decision_ledger_record(&led, 1, (i < 90), 0.01, 0.031, 1.0, 0.809, 5.0);
    for (int i = 0; i < 100; i++)
        wubu_decision_ledger_record(&led, 0, 1, 0.0, 0.0, 1.0, 0.809, 5.0);

    double cost = wubu_decision_ledger_cost(&led);
    printf("  ledger: delegated=%lu deferred=%lu wrong=%lu cost=%.3f\n",
           led.delegated, led.deferred, led.wrong_delegated, cost);
    CHECK(led.delegated == 100 && led.deferred == 100, "ledger counts both paths");
    CHECK(led.wrong_delegated == 10, "ledger counts wrong delegations");
    CHECK(fabs(cost - 151.0) < 1e-6, "cost = dm 1.0 + llm 100.0 + penalty 50.0");
    CHECK(fabs(wubu_decision_ledger_saved_pct(&led, 200.0) - 24.5) < 1e-6,
          "saved 24.5% against an LLM-always baseline");
    CHECK(wubu_decision_ledger_cost(NULL) == -1.0, "NULL ledger -> -1");

    printf("\n=== test_decision: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
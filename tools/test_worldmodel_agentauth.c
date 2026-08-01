/*
 * test_worldmodel_agentauth.c -- AG04 (world-model closed-loop) + AG07 (inter-agent auth).
 */
#include "wubu_worldmodel.h"
#include "wubu_agentauth.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_worldmodel_agentauth (AG04/AG07) ===\n");

    /* AG04: world-model closed loop */
    wubu_wm_t m; m.n = 2;
    /* identity-ish transition: s' = s + 1 (A=I, b=1) */
    for (int i = 0; i < 4; i++) m.A[i] = 0;
    m.A[0] = 1; m.A[3] = 1;   /* identity */
    m.b[0] = 1; m.b[1] = 1;
    double cur[2] = { 3, 5 };
    double pred[2];
    wubu_wm_predict(&m, cur, pred);
    CHECK(pred[0] == 4 && pred[1] == 6, "predict s' = s+1");

    /* closed step: observed matches prediction -> no replan */
    double obs_ok[2] = { 4, 6 };
    CHECK(wubu_wm_closed_step(&m, cur, obs_ok, 0.5, NULL) == 0, "observed matches -> no replan");

    /* observed diverges -> REPLAN signaled (open-loop detected) */
    double obs_bad[2] = { 9, 2 };
    CHECK(wubu_wm_closed_step(&m, cur, obs_bad, 0.5, NULL) == 1, "divergence -> REPLAN");
    CHECK(wubu_wm_divergence(pred, obs_bad, 2) > 0.5, "divergence magnitude > thr");

    /* AG07: inter-agent authentication */
    unsigned long long mac = wubu_agent_mac("cog", "sibling", "step=5;lock=x", "shared-secret");
    CHECK(mac != 0, "MAC produced");
    CHECK(wubu_agent_verify("cog", "sibling", "step=5;lock=x", "shared-secret", mac) == 1,
          "valid MAC -> authenticated");
    /* tamper: payload changed -> MAC mismatch -> reject (default-deny) */
    CHECK(wubu_agent_verify("cog", "sibling", "step=6;lock=x", "shared-secret", mac) == 0,
          "tampered payload -> REJECTED");
    /* spoof: wrong sender -> reject */
    CHECK(wubu_agent_verify("rogue", "sibling", "step=5;lock=x", "shared-secret", mac) == 0,
          "spoofed sender -> REJECTED");
    /* wrong secret -> reject (origin auth) */
    CHECK(wubu_agent_verify("cog", "sibling", "step=5;lock=x", "wrong-secret", mac) == 0,
          "wrong secret -> REJECTED");

    if (failures == 0) { printf("ALL WORLDMODEL-AGENTAUTH TESTS PASSED\n"); return 0; }
    printf("%d WORLDMODEL-AGENTAUTH TEST(S) FAILED\n", failures);
    return 1;
}

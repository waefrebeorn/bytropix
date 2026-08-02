/* test_serve.c -- Theme IR batch 1: the multi-tenant serving scheduler. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_serve.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_serve (IR batch 1) ===\n");

    /* IR01: admission */
    CHECK(wubu_serve_admit(80, 100, 19) == 1, "within budget");
    CHECK(wubu_serve_admit(80, 100, 21) == 0, "over budget rejected");
    CHECK(wubu_serve_admit(0, 0, 1) == 0, "zero budget rejects");

    /* IR02: fair share */
    NEAR(wubu_serve_fair_share(50, 100, 1.0f), 0.5f, 1e-6f);

    /* IR03: preempt vs wait */
    CHECK(wubu_serve_preempt(0.3f, 1.0f) == 1, "cheap rebuild -> preempt");
    CHECK(wubu_serve_preempt(2.0f, 1.0f) == 0, "expensive rebuild -> wait");

    /* IR04: activation guard */
    CHECK(wubu_serve_guard(90, 100, 5) == 1, "under threshold");
    CHECK(wubu_serve_guard(90, 100, 15) == 0, "over threshold denied");

    /* IR06: LCP */
    {
        uint32_t a[5] = { 1, 2, 3, 4, 5 }, b[5] = { 1, 2, 9, 4, 5 };
        CHECK(wubu_serve_lcp(a, b, 5) == 2, "shared prefix of 2");
    }

    /* IR07: decoupled scheduling */
    {
        wubu_serve_dec_t d;
        CHECK(wubu_serve_decouple(1, 100, 50, &d) == 0 && d.decision == 1 &&
              d.reserved == 50, "schedule + reserve");
        wubu_serve_decouple(0, 100, 50, &d);
        CHECK(d.decision == 0 && d.reserved == 0, "no schedule -> no reserve");
    }

    /* IR09: burst headroom */
    CHECK(wubu_serve_burst_headroom(100, 1.5f, 200) == 50, "50 headroom");
    CHECK(wubu_serve_burst_headroom(100, 1.5f, 120) == 20, "budget-capped");

    /* IR10: tier starvation bounds */
    {
        long starve = 0;
        CHECK(wubu_serve_tier_admit(0, &starve, 3) == 1, "top tier always");
        CHECK(wubu_serve_tier_admit(1, &starve, 3) == 0, "low tier waits");
        wubu_serve_tier_admit(1, &starve, 3);
        wubu_serve_tier_admit(1, &starve, 3);
        CHECK(wubu_serve_tier_admit(1, &starve, 3) == 1, "bound breaks through");
    }

    /* IR11: tenant share */
    CHECK(wubu_serve_tenant_share(100, 4) == 25, "equal split");

    /* IR13: victim selection */
    {
        float cost[4] = { 2.0f, 0.5f, 3.0f, 1.0f };
        CHECK(wubu_serve_victim(cost, 4) == 1, "cheapest victim");
    }

    /* IR14: checkpointed preemption */
    CHECK(wubu_serve_checkpoint(0.4f, 1.0f) == 1, "snapshot cheaper");
    CHECK(wubu_serve_checkpoint(1.0f, 0.4f) == 0, "restart cheaper");

    /* IR15: SLO slack */
    NEAR(wubu_serve_slo_slack(10.0f, 5.0f, 2.0f), 3.0f, 1e-5f);
    CHECK(wubu_serve_slo_slack(10.0f, 9.0f, 5.0f) < 0, "missed SLO negative");

    /* IR16: batch compaction */
    {
        long fill = 0;
        CHECK(wubu_serve_compact(3, 5, &fill) == 0 && fill == 3,
              "gaps filled to decode slots");
    }

    /* IR19: priority inheritance */
    {
        int inh = 0;
        CHECK(wubu_serve_pi(1, 3, &inh) == 0 && inh == 1, "inherits min prio");
    }

    /* IR20: hysteresis */
    {
        int state = 0;
        wubu_serve_hysteresis(95, 100, 50, &state);
        CHECK(state == 0, "below hi stays accepting");
        wubu_serve_hysteresis(110, 100, 50, &state);
        CHECK(state == 1, "above hi preempts");
        wubu_serve_hysteresis(70, 100, 50, &state);
        CHECK(state == 1, "stays preempting until the lo bound");
        wubu_serve_hysteresis(40, 100, 50, &state);
        CHECK(state == 0, "below lo resumes accepting");
    }

    if (failures == 0) printf("ALL SERVE TESTS PASSED\n");
    else printf("%d SERVE FAILURES\n", failures);
    return failures ? 1 : 0;
}

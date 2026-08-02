/* test_serve2.c -- Theme IR complete: the serving scheduler frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_serve2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_serve2 (IR complete) ===\n");
    NEAR(wubu_serve2_fairness(50, 100), 0.5f, 1e-6f);
    NEAR(wubu_serve2_ptel(2, 10, 0.5f), 0.1f, 1e-6f);
    {
        long fill = 0;
        CHECK(wubu_serve2_cosched(10, 20, 2.0f, &fill) == 0 && fill == 10,
              "co-schedule fills");
    }
    {
        int pl[3] = { 2, 5, 3 };
        CHECK(wubu_serve2_route(pl, 3) == 1, "routes to the prefix node");
    }
    CHECK(wubu_serve2_prof(100, 0.5f) == 150, "demand profile");
    CHECK(wubu_serve2_admit_pred(80, 100, 15) == 1, "predicted within");
    CHECK(wubu_serve2_admit_pred(80, 100, 25) == 0, "predicted over");
    CHECK(wubu_serve2_work_conserving(5, 2, 4) == 1, "work-conserving");
    CHECK(wubu_serve2_pbudget(90, 100) == 1, "preempt budget ok");
    CHECK(wubu_serve2_decode_prio(1, 10) == 1, "decode prioritized");
    {
        long q[3] = { 10, 3, 7 };
        int c = -1;
        CHECK(wubu_serve2_queue(q, 3, &c) == 0 && c == 1, "least-loaded queue");
    }
    CHECK(wubu_serve2_backfill(3, 9) == 3, "backfill bounded");
    CHECK(wubu_serve2_keepalive(0.9f, 0.5f) == 1, "hot kept");
    CHECK(wubu_serve2_arbitrate(0.3f, 0.9f) == 1, "evict cold");
    NEAR(wubu_serve2_cost(100, 0.5f), 50.0f, 1e-4f);
    NEAR(wubu_serve2_burst_weight(2.0f, 1.0f), 2.0f, 1e-5f);
    {
        uint32_t a[4] = { 1, 2, 3, 4 }, b[4] = { 1, 2, 9, 9 };
        CHECK(wubu_serve2_group(a, b, 4, 2) == 1, "similar grouped");
        CHECK(wubu_serve2_coalesce(a, b, 4, 0.9f) == 0, "not coalescible");
    }
    NEAR(wubu_serve2_restore(0.4f, 1.0f), 0.4f, 1e-5f);
    CHECK(wubu_serve2_resilient(5, 5) == 1, "no loss on restart");
    CHECK(wubu_serve2_isolate(90, 100) == 1, "tenant within cap");
    {
        long debt = 0;
        CHECK(wubu_serve2_debt(120, &debt, 100) == 1 && debt == 20, "debt tracked");
    }
    CHECK(wubu_serve2_slo_violation(10, 12, 1) == 1, "SLO breached");
    CHECK(wubu_serve2_concurrency(100, 200, 8) == 4, "pressure halves");
    CHECK(wubu_serve2_policy_select(0.9f, 0.1f) == 1, "throughput policy");
    CHECK(wubu_serve2_policy_select(0.5f, 0.9f) == 2, "burst policy");
    CHECK(wubu_serve2_scavenge(0.8f, 0.5f) == 1, "idle scavenged");
    CHECK(wubu_serve2_cost_benefit(2.0f, 0.5f, 1.0f) == 1, "preempt pays");
    {
        long conc = 8;
        CHECK(wubu_serve2_feedback(0.9f, 0.7f, &conc) == 1 && conc == 4,
              "pressure feedback");
    }
    CHECK(wubu_serve2_deadline(10, 5, 2) == 1, "meets deadline");
    {
        float crit[3] = { 0.9f, 0.2f, 0.5f };
        CHECK(wubu_serve2_fair_preempt(crit, 3) == 1, "least-critical first");
    }
    CHECK(wubu_serve2_shared_save((int[]){ 2, 8, 4 }, 3, 100) == 800, "shared save");
    {
        long ml[2] = { 40, 10 };
        int c = -1;
        CHECK(wubu_serve2_multi_model(ml, 2, &c) == 0 && c == 1, "idle model");
    }
    {
        int st = 0;
        wubu_serve2_sched_hysteresis(0.8f, 0.7f, 0.3f, &st);
        CHECK(st == 1, "high load engages");
        wubu_serve2_sched_hysteresis(0.5f, 0.7f, 0.3f, &st);
        CHECK(st == 1, "hysteresis holds");
    }
    NEAR(wubu_serve2_qdepth((long[]){ 10, 20 }, 2), 15.0f, 1e-5f);
    {
        float prio = 0;
        CHECK(wubu_serve2_aging(100, 100, &prio) == 1, "starved force-admits");
        NEAR(prio, 1.0f, 1e-5f);
    }
    {
        float costs[4] = { 1, 3, 2, 1 };
        int victims[4];
        CHECK(wubu_serve2_simulate(costs, 4, 4.0f, victims, 4) == 2,
              "budget fits {1,3} then exhausts");
    }
    CHECK(wubu_serve2_negotiate(150, 100, 40) == 1, "burst negotiation");
    CHECK(wubu_serve2_reclaim(50, 20) == 30, "reclaim rate");
    {
        long nc = 0;
        CHECK(wubu_serve2_prefill_plan(250, 100, &nc) == 0 && nc == 3, "chunks");
    }
    {
        uint32_t log[3] = { 0, 0, 0 };
        wubu_serve2_log(log, 3, 7);
        wubu_serve2_log(log, 3, 9);
        CHECK(log[0] == 9 && log[1] == 7, "event log newest-first");
    }
    CHECK(wubu_serve2_powercap(100, 0.5f, 60.0f) == 1, "within envelope");
    CHECK(wubu_serve2_powercap(100, 0.5f, 40.0f) == 0, "over envelope");

    if (failures == 0) printf("ALL SERVE2 TESTS PASSED\n");
    else printf("%d SERVE2 FAILURES\n", failures);
    return failures ? 1 : 0;
}

/* test_pim2.c -- Theme IS complete: the PIM/hardware frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_pim2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_pim2 (IS complete) ===\n");
    CHECK(wubu_pim2_bits(0.9f, 0.3f, 0.7f) == 8, "sensitive -> 8-bit");
    CHECK(wubu_pim2_bits(0.5f, 0.3f, 0.7f) == 4, "mid -> 4-bit");
    CHECK(wubu_pim2_bits(0.1f, 0.3f, 0.7f) == 2, "robust -> 2-bit");
    CHECK(wubu_pim2_endurance(90, 100) == 1, "within endurance");
    CHECK(wubu_pim2_endurance(120, 100) == 0, "wear exceeded");
    CHECK(wubu_pim2_frontier(100, 1.0f) > 1.0f, "capacity raises latency");
    {
        float eff[3] = { 0.5f, 0.9f, 0.7f };
        int c = -1;
        CHECK(wubu_pim2_hetero(eff, 3, &c) == 0 && c == 1, "best device");
    }
    CHECK(wubu_pim2_wall(50, 100) == 1, "within the wall");
    CHECK(wubu_pim2_wall(150, 100) == 0, "wall breached");
    {
        int32_t out[3];
        float kv[3] = { 1, -1, 0.5f };
        wubu_pim2_int_kv(kv, 3, 4, out);
        CHECK(out[0] == 7 && out[1] == -7, "4-bit integer KV");
    }
    CHECK(wubu_pim2_ns_rag(0.3f, 1.0f) == 1, "retrieve at the SSD");
    {
        wubu_pim2_dispatch_t table[2] = { { 0, 1, 5.0f }, { 0, 2, 2.0f } };
        int dev = -1;
        CHECK(wubu_pim2_dispatch(table, 2, 0, &dev) == 0 && dev == 2,
              "cheapest device");
    }
    {
        int na = 0;
        CHECK(wubu_pim2_place(10, 4, 4, &na) == 0 && na == 12, "arrays placed");
    }
    NEAR(wubu_pim2_energy(100, 0.5f), 50.0f, 1e-4f);
    CHECK(wubu_pim2_audit(8, 0.001f) > 0.001f, "audit bound positive");
    CHECK(wubu_pim2_mem_centric(0.9f, 0.5f) == 1, "memory-centric decode");
    CHECK(wubu_pim2_benefit(0.8f, 2.0f, 0.2f) == 1, "PIM wins");
    CHECK(wubu_pim2_benefit(1.9f, 2.0f, 0.2f) == 0, "margin not met");
    CHECK(wubu_pim2_counters(10, 20, 1) == 1030, "counter composite");
    CHECK(wubu_pim2_page_move(2.0f, 0.5f) == 1, "move to the hotter tier");
    CHECK(wubu_pim2_latency(4, 10) > wubu_pim2_latency(0, 10), "PCM slower");
    {
        long b = 0;
        CHECK(wubu_pim2_batch_shape(10, 4, &b) == 0 && b == 3, "array batches");
    }
    NEAR(wubu_pim2_acc_guard(5.0f, 3.0f), 3.0f, 1e-6f);
    CHECK(wubu_pim2_plan(900, 100, 1000) == 1, "fits");
    CHECK(wubu_pim2_plan(900, 200, 1000) == 0, "doesn't fit");
    NEAR(wubu_pim2_dataflow(0.8f, 0.2f), 0.6f, 1e-5f);
    NEAR(wubu_pim2_roofline(100, 2.0f), 200.0f, 1e-4f);
    CHECK(wubu_pim2_refresh(0.05f, 0.03f) == 1, "drift triggers refresh");
    CHECK(wubu_pim2_device_bits(0.5f, 0.1f) == 4, "low-precision device");
    {
        float h[3] = { 1, 2, 3 }, p[3] = { 1.1f, 2.1f, 3.1f };
        NEAR(wubu_pim2_parity(h, p, 3), 0.1f, 1e-5f);
    }
    {
        float v[4] = { 1, 4, 2, 3 };
        int idx[4];
        CHECK(wubu_pim2_topk(v, 4, 2, idx) == 2, "top-2");
        CHECK(idx[0] == 1, "argmax first");
    }
    {
        int devs[3] = { 0, 2, 5 };
        uint32_t m = wubu_pim2_matrix(devs, 3);
        CHECK((m & 0x25) == 0x25, "device bits set");
    }
    CHECK(wubu_pim2_powercap(50.0f, 60.0f) == 1, "within the cap");
    CHECK(wubu_pim2_powercap(70.0f, 60.0f) == 0, "over the cap");
    NEAR(wubu_pim2_ledger(100, 0.5f), 50.0f, 1e-4f);
    CHECK(wubu_pim2_plan_centric(0.8f, 0.5f) == 1, "memory-centric plan");

    if (failures == 0) printf("ALL PIM2 TESTS PASSED\n");
    else printf("%d PIM2 FAILURES\n", failures);
    return failures ? 1 : 0;
}

/* test_watchdog.c -- KERNEL #20 watchdog restarts a hung subsystem.
 * Proves: heartbeat keeps healthy, timeout -> hung -> scoped restart, restart
 * count, fail-closed on NULL. */
#include "wubu_watchdog.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_watchdog_t w;
    CHECK(wubu_wd_init(&w) == 0, "init ok");
    CHECK(wubu_wd_register(&w, "net", 100) == 0, "register net (timeout 100)");
    CHECK(wubu_wd_register(&w, "gpu", 50) == 0, "register gpu (timeout 50)");

    /* healthy: kick within timeout */
    wubu_wd_tick(&w, 40);
    wubu_wd_kick(&w, "net"); wubu_wd_kick(&w, "gpu");
    wubu_wd_tick(&w, 40);   /* now=80, both within timeout */
    CHECK(wubu_wd_state_of(&w, "net") == WUBU_WD_HEALTHY, "net healthy");
    CHECK(wubu_wd_state_of(&w, "gpu") == WUBU_WD_HEALTHY, "gpu healthy");
    CHECK(wubu_wd_tick(&w, 0) == 0, "no restart at now=80");

    /* let net go past its timeout (100). At now=80, tick 30 -> now=110, net last
     * kick was 40, age=70 < 100 ok; gpu last kick 40, age=70 > 50 -> gpu HUNG */
    int r = wubu_wd_tick(&w, 30);   /* now=110 */
    CHECK(r == 1, "1 restart (gpu exceeded timeout 50)");
    CHECK(wubu_wd_state_of(&w, "gpu") == WUBU_WD_RESTARTED, "gpu restarted");
    CHECK(wubu_wd_state_of(&w, "net") == WUBU_WD_HEALTHY, "net still healthy (scoped)");
    CHECK(wubu_wd_total_restarts(&w) == 1, "1 total restart");

    /* net now goes past 100 without a kick: at now=110, net last kick=40, age=70.
     * tick 40 -> now=150, net age=110 > 100 -> net restarts */
    r = wubu_wd_tick(&w, 40);   /* now=150 */
    CHECK(r == 1, "1 restart (net exceeded timeout)");
    CHECK(wubu_wd_state_of(&w, "net") == WUBU_WD_RESTARTED, "net restarted");
    CHECK(wubu_wd_total_restarts(&w) == 2, "2 total restarts");

    /* after restart, subsystem is fresh; a new kick + tick keeps it healthy */
    wubu_wd_kick(&w, "net");
    wubu_wd_tick(&w, 20);
    CHECK(wubu_wd_state_of(&w, "net") == WUBU_WD_RESTARTED, "net stays restarted (not re-hung yet)");

    /* fail-closed */
    CHECK(wubu_wd_register(&w, "x", 0) == -1, "zero timeout -> -1");
    CHECK(wubu_wd_kick(&w, "nope") == -1, "kick unknown -> -1");
    CHECK(wubu_wd_state_of(&w, "nope") == -1, "unknown state -> -1");
    CHECK(wubu_wd_tick(NULL, 1) == -1, "NULL tick -> -1");
    CHECK(wubu_wd_kick(NULL, "net") == -1, "NULL kick -> -1");
    CHECK(wubu_wd_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_watchdog: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}

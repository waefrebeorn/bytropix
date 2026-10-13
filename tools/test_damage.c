/* test_damage.c -- UX #247 Wayland compositor damage tracking.
 * Proves: add/clamp rects, area, full-damage threshold, clear, count,
 * fail-closed on NULL/bad args. */
#include "wubu_damage.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_damage_t d;
    CHECK(wubu_dmg_init(&d, 100, 100) == 0, "init ok (100x100 surface)");
    CHECK(wubu_dmg_count(&d) == 0, "0 rects initially");
    CHECK(wubu_dmg_area(&d) == 0, "0 area initially");

    /* add two small damage rects */
    CHECK(wubu_dmg_add(&d, 10, 10, 20, 20) == 0, "add rect (10,10,20,20)");
    CHECK(wubu_dmg_add(&d, 50, 50, 10, 10) == 0, "add rect (50,50,10,10)");
    CHECK(wubu_dmg_count(&d) == 2, "2 rects");
    /* area = 20*20 + 10*10 = 500 */
    CHECK(wubu_dmg_area(&d) == 500, "area = 500");
    /* 500/10000 = 5% < 90% -> not full */
    CHECK(wubu_dmg_is_full(&d) == 0, "5% damage -> not full");

    /* clamp: a rect extending past the surface is clipped */
    CHECK(wubu_dmg_add(&d, 90, 90, 50, 50) == 0, "add overflowing rect (clamped)");
    /* third rect clamped to (90,90,10,10) = 100 area; total 600 */
    CHECK(wubu_dmg_area(&d) == 600, "overflowing rect clamped (total 600)");

    /* a fully off-surface rect is rejected */
    CHECK(wubu_dmg_add(&d, 200, 200, 10, 10) == -1, "off-surface rect -> -1");
    CHECK(wubu_dmg_add(&d, -50, -50, 10, 10) == -1, "fully-negative rect -> -1");

    /* clear */
    CHECK(wubu_dmg_clear(&d) == 0, "clear ok");
    CHECK(wubu_dmg_count(&d) == 0, "0 rects after clear");
    CHECK(wubu_dmg_area(&d) == 0, "0 area after clear");

    /* full damage: cover the whole surface */
    CHECK(wubu_dmg_add(&d, 0, 0, 100, 100) == 0, "add full-surface rect");
    CHECK(wubu_dmg_area(&d) == 10000, "area = 10000 (full)");
    CHECK(wubu_dmg_is_full(&d) == 1, "100% damage -> full");

    /* near-full: 95% -> full (>= 90%) */
    wubu_dmg_clear(&d);
    wubu_dmg_add(&d, 0, 0, 100, 95);   /* 9500 = 95% */
    CHECK(wubu_dmg_is_full(&d) == 1, "95% -> full (>= 90% threshold)");

    /* 80% -> not full */
    wubu_dmg_clear(&d);
    wubu_dmg_add(&d, 0, 0, 100, 80);   /* 8000 = 80% */
    CHECK(wubu_dmg_is_full(&d) == 0, "80% -> not full");

    /* fail-closed */
    CHECK(wubu_dmg_init(NULL, 10, 10) == -1, "NULL init -> -1");
    CHECK(wubu_dmg_init(&d, 0, 10) == -1, "zero width -> -1");
    CHECK(wubu_dmg_add(&d, 0, 0, 0, 10) == -1, "zero-size rect -> -1");
    CHECK(wubu_dmg_add(NULL, 0, 0, 10, 10) == -1, "NULL add -> -1");
    CHECK(wubu_dmg_area(NULL) == 0, "NULL area -> 0");
    CHECK(wubu_dmg_clear(NULL) == -1, "NULL clear -> -1");
    CHECK(wubu_dmg_count(NULL) == 0, "NULL count -> 0");

    printf("\n=== test_damage: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}

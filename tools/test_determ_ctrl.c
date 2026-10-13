/* test_determ_ctrl.c -- AGIEDR #305 deterministic controls outside model plane.
 * Proves: all-controls-pass allows, first-failing-control denies (fail-closed),
 * order matters, telemetry, last-denied recorded, fail-closed on NULL. */
#include "wubu_determ_ctrl.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

/* a simple action: an int value the controls inspect */
static int ctrl_positive(const void *a, void *ctx) { (void)ctx; return *(const int*)a > 0; }
static int ctrl_under_100(const void *a, void *ctx) { (void)ctx; return *(const int*)a < 100; }
static int ctrl_even(const void *a, void *ctx) { (void)ctx; return (*(const int*)a % 2) == 0; }

int main(void) {
    wubu_determ_ctrl_t g;
    CHECK(wubu_dc_init(&g) == 0, "init ok");

    /* register controls in order: positive, under_100, even */
    CHECK(wubu_dc_add(&g, "positive", ctrl_positive) == 0, "add positive");
    CHECK(wubu_dc_add(&g, "under_100", ctrl_under_100) == 0, "add under_100");
    CHECK(wubu_dc_add(&g, "even", ctrl_even) == 0, "add even");

    /* action 4: passes all -> allowed */
    int a4 = 4;
    CHECK(wubu_dc_check(&g, &a4, NULL) == 1, "4 passes all controls");

    /* action -1: fails positive (first) -> denied */
    int an = -1;
    CHECK(wubu_dc_check(&g, &an, NULL) == 0, "-1 denied (fails positive)");
    CHECK(strcmp(wubu_dc_last_denied(&g), "positive") == 0, "last denied = positive");

    /* action 200: passes positive, fails under_100 -> denied */
    int a200 = 200;
    CHECK(wubu_dc_check(&g, &a200, NULL) == 0, "200 denied (fails under_100)");
    CHECK(strcmp(wubu_dc_last_denied(&g), "under_100") == 0, "last denied = under_100");

    /* action 3: passes positive+under_100, fails even -> denied */
    int a3 = 3;
    CHECK(wubu_dc_check(&g, &a3, NULL) == 0, "3 denied (fails even)");
    CHECK(strcmp(wubu_dc_last_denied(&g), "even") == 0, "last denied = even");

    /* fail-closed: no controls -> everything allowed (vacuous), but NULL gate denies */
    CHECK(wubu_dc_check(NULL, &a4, NULL) == 0, "NULL gate -> deny (fail-closed)");

    /* telemetry */
    CHECK(wubu_dc_allowed(&g) == 1, "1 allowed (action 4)");
    CHECK(wubu_dc_denied(&g) == 3, "3 denied (-1, 200, 3)");

    /* fail-closed on bad registration */
    CHECK(wubu_dc_add(&g, "bad", NULL) == -1, "NULL fn -> -1");
    CHECK(wubu_dc_add(NULL, "x", ctrl_positive) == -1, "NULL gate add -> -1");
    CHECK(wubu_dc_last_denied(NULL) != NULL, "NULL last_denied returns empty string");

    printf("\n=== test_determ_ctrl: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}

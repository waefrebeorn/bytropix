/* test_thermal.c -- KERNEL #32 tests: thermal governor via /n/hw/<dev>/thermal. */
#include "wubu_thermal.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_thermal_t t;

    /* init + add */
    CHECK(wubu_thermal_init(&t) == 0);
    CHECK(wubu_thermal_add(&t, "gpu0", WUBU_THERM_BALANCED) == 0);
    CHECK(wubu_thermal_find(&t, "gpu0") == 0);
    CHECK(wubu_thermal_add(&t, "gpu0", WUBU_THERM_BALANCED) == -1);  /* dup */
    CHECK(wubu_thermal_find(&t, "nope") == -1);

    /* governor name round-trip */
    CHECK(wubu_thermal_gov_from_name("performance") == WUBU_THERM_PERFORMANCE);
    CHECK(wubu_thermal_gov_from_name("balanced") == WUBU_THERM_BALANCED);
    CHECK(wubu_thermal_gov_from_name("powersave") == WUBU_THERM_POWERSAVE);
    CHECK(wubu_thermal_gov_from_name("turbo") == -1);
    CHECK(wubu_thermal_gov_from_name(NULL) == -1);
    CHECK(strcmp(wubu_thermal_gov_name(WUBU_THERM_PERFORMANCE), "performance") == 0);
    CHECK(strcmp(wubu_thermal_gov_name(WUBU_THERM_BALANCED), "balanced") == 0);
    CHECK(strcmp(wubu_thermal_gov_name(WUBU_THERM_POWERSAVE), "powersave") == 0);

    /* balanced thresholds: light 75k, heavy 90k, crit 100k */
    CHECK(wubu_thermal_update(&t, "gpu0", 50000) == WUBU_THERM_NONE);
    CHECK(wubu_thermal_update(&t, "gpu0", 74000) == WUBU_THERM_NONE);   /* just under light */
    CHECK(wubu_thermal_update(&t, "gpu0", 75000) == WUBU_THERM_LIGHT);  /* at light */
    CHECK(wubu_thermal_update(&t, "gpu0", 89000) == WUBU_THERM_LIGHT);
    CHECK(wubu_thermal_update(&t, "gpu0", 90000) == WUBU_THERM_HEAVY);  /* at heavy */
    CHECK(wubu_thermal_update(&t, "gpu0", 100000) == WUBU_THERM_CRITICAL); /* at crit */
    CHECK(wubu_thermal_throttle(&t, "gpu0") == WUBU_THERM_CRITICAL);

    /* fail-closed on implausible temp: keeps prior state */
    CHECK(wubu_thermal_update(&t, "gpu0", 999999) == WUBU_THERM_CRITICAL); /* rejected, prior kept */
    CHECK(wubu_thermal_update(&t, "gpu0", -999999) == WUBU_THERM_CRITICAL);

    /* switch to powersave: thresholds drop (light 60k, heavy 75k, crit 90k) */
    CHECK(wubu_thermal_set_gov(&t, "gpu0", WUBU_THERM_POWERSAVE) == 0);
    /* current temp is 100000 (last accepted) -> now above powersave crit 90k */
    CHECK(wubu_thermal_throttle(&t, "gpu0") == WUBU_THERM_CRITICAL);
    CHECK(wubu_thermal_update(&t, "gpu0", 65000) == WUBU_THERM_LIGHT);   /* powersave light 60k */
    CHECK(wubu_thermal_update(&t, "gpu0", 55000) == WUBU_THERM_NONE);

    /* switch to performance: thresholds rise (light 90k, heavy 100k, crit 110k) */
    CHECK(wubu_thermal_set_gov(&t, "gpu0", WUBU_THERM_PERFORMANCE) == 0);
    CHECK(wubu_thermal_update(&t, "gpu0", 85000) == WUBU_THERM_NONE);    /* under perf light 90k */
    CHECK(wubu_thermal_update(&t, "gpu0", 95000) == WUBU_THERM_LIGHT);
    CHECK(wubu_thermal_update(&t, "gpu0", 105000) == WUBU_THERM_HEAVY);
    CHECK(wubu_thermal_update(&t, "gpu0", 115000) == WUBU_THERM_CRITICAL);

    /* set_gov on unknown device / bad args */
    CHECK(wubu_thermal_set_gov(&t, "nope", WUBU_THERM_BALANCED) == -1);
    CHECK(wubu_thermal_throttle(&t, "nope") == -1);

    /* second device independent */
    CHECK(wubu_thermal_add(&t, "cpu0", WUBU_THERM_PERFORMANCE) == 0);
    CHECK(wubu_thermal_update(&t, "cpu0", 95000) == WUBU_THERM_LIGHT);
    CHECK(wubu_thermal_throttle(&t, "gpu0") == WUBU_THERM_CRITICAL);  /* gpu0 unchanged */

    if (fails == 0) printf("ALL THERMAL TESTS PASSED\n");
    else printf("%d THERMAL TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}

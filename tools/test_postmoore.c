/* test_postmoore.c -- PATHS #328 post-Moore algorithmic-scaling research.
 * Proves: combined speedup is the product, compute-saved fraction, effective
 * capability, algorithmic-years, neutral stack -> 1x, fail-closed on NULL/bad. */
#include "wubu_postmoore.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    /* realistic multipliers from the implemented techniques */
    wubu_postmoore_t p;
    CHECK(wubu_pm_init(&p, 1.4, 2.0, 3.0, 1.5) == 0, "init ok (isa 1.4, spec 2.0, moe 3.0, mem 1.5)");

    /* total speedup = product = 1.4*2.0*3.0*1.5 = 12.6 */
    double total = wubu_pm_total_speedup(&p);
    CHECK(NEAR(total, 12.6, 1e-9), "total speedup = 12.6x (product)");

    /* compute saved = 1 - 1/12.6 = 0.9206... */
    double saved = wubu_pm_compute_saved(&p);
    CHECK(NEAR(saved, 1.0 - 1.0/12.6, 1e-9), "compute saved = 1 - 1/speedup");
    CHECK(saved > 0.9, "~92% compute eliminated");

    /* effective capability = speedup */
    CHECK(NEAR(wubu_pm_effective_capability(&p), 12.6, 1e-9), "effective capability = 12.6x");

    /* algorithmic years = log2(12.6)*2 = ~7.32 years of Moore scaling */
    double years = wubu_pm_algorithmic_years(&p);
    CHECK(NEAR(years, log2(12.6)*2.0, 1e-9), "algorithmic years = log2(speedup)*2");
    CHECK(years > 7.0, "~7 years of Moore scaling gained algorithmically");

    /* neutral stack (all 1.0) -> 1x, 0% saved, 0 years */
    wubu_postmoore_t neutral;
    wubu_pm_init(&neutral, 1.0, 1.0, 1.0, 1.0);
    CHECK(NEAR(wubu_pm_total_speedup(&neutral), 1.0, 1e-9), "neutral stack -> 1x");
    CHECK(wubu_pm_compute_saved(&neutral) == 0.0, "neutral -> 0% saved");
    CHECK(wubu_pm_algorithmic_years(&neutral) == 0.0, "neutral -> 0 years");

    /* a single dominant technique: spec_decode 4x alone */
    wubu_postmoore_t single;
    wubu_pm_init(&single, 1.0, 4.0, 1.0, 1.0);
    CHECK(NEAR(wubu_pm_total_speedup(&single), 4.0, 1e-9), "single 4x technique -> 4x total");

    /* fail-closed: multipliers must be >= 1 */
    CHECK(wubu_pm_init(&p, 0.5, 1.0, 1.0, 1.0) == -1, "multiplier < 1 -> -1");
    CHECK(wubu_pm_init(NULL, 1, 1, 1, 1) == -1, "NULL init -> -1");
    CHECK(wubu_pm_total_speedup(NULL) == 0.0, "NULL total -> 0");
    CHECK(wubu_pm_compute_saved(NULL) == 0.0, "NULL saved -> 0");
    CHECK(wubu_pm_algorithmic_years(NULL) == 0.0, "NULL years -> 0");

    printf("\n=== test_postmoore: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}

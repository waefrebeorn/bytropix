/* test_behavior.c -- AGI-EDR runtime behavior classifier (AGIEDR #295).
 * Proves: steady stream -> NONE; a spike -> escalates; sustained anomaly ->
 * fail-closed halt (SUSPEND/STOP); NaN/garbage -> STOP (fail-closed); NULL ->
 * STOP; reset clears. Every assertion is measured, not assumed. */
#include "wubu_behavior.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_behavior_t b;

    /* init */
    CHECK(wubu_behavior_init(&b, 3.0) == 0, "init ok");
    CHECK(wubu_behavior_level(&b) == WUBU_CONT_NONE, "fresh level NONE");
    CHECK(wubu_behavior_halt(&b) == 0, "fresh no halt");

    /* fail-closed on NULL */
    CHECK(wubu_behavior_halt(NULL) == 1, "NULL classifier halts (fail-closed)");
    CHECK(wubu_behavior_level(NULL) == WUBU_CONT_STOP, "NULL level STOP");
    CHECK(wubu_behavior_observe(NULL, WUBU_MEAS_LOSS, 1.0) == WUBU_CONT_STOP, "NULL observe STOP");

    /* steady stream around 1.0 -> stays NONE, no halt */
    int escalated = 0;
    for (int i = 0; i < 40; i++) {
        double v = 1.0 + 0.01 * ((i % 5) - 2);   /* tiny jitter */
        int lvl = wubu_behavior_observe(&b, WUBU_MEAS_LOSS, v);
        if (lvl > WUBU_CONT_WARN) escalated = 1;
    }
    CHECK(escalated == 0, "steady jitter stays <= WARN");
    CHECK(wubu_behavior_halt(&b) == 0, "steady stream no halt");

    /* a big spike -> WARN or higher (z fires) */
    int spike_lvl = wubu_behavior_observe(&b, WUBU_MEAS_LOSS, 50.0);
    CHECK(spike_lvl >= WUBU_CONT_WARN, "spike escalates to >= WARN");

    /* reset clears the streak/severity */
    wubu_behavior_reset(&b);
    CHECK(wubu_behavior_level(&b) == WUBU_CONT_NONE, "reset -> NONE");
    CHECK(wubu_behavior_halt(&b) == 0, "reset clears halt");

    /* sustained anomaly -> fail-closed halt. Feed a rising trend so z stays
     * high across many samples and the streak escalates the ladder. */
    wubu_behavior_t c;
    wubu_behavior_init(&c, 3.0);
    int halted = 0;
    double v = 1.0;
    for (int i = 0; i < 60 && !halted; i++) {
        v *= 1.5;                 /* exponential blowup: every sample anomalous */
        int lvl = wubu_behavior_observe(&c, WUBU_MEAS_GRAD_NORM, v);
        if (wubu_behavior_halt(&c)) halted = 1;
        (void)lvl;
    }
    CHECK(halted == 1, "sustained blowup -> fail-closed halt");
    CHECK(wubu_behavior_level(&c) >= WUBU_CONT_SUSPEND, "halt level >= SUSPEND");

    /* NaN -> immediate STOP (fail-closed) */
    wubu_behavior_t d;
    wubu_behavior_init(&d, 3.0);
    CHECK(wubu_behavior_observe(&d, WUBU_MEAS_LOSS, NAN) == WUBU_CONT_STOP, "NaN -> STOP");
    CHECK(wubu_behavior_halt(&d) == 1, "NaN halts");
    CHECK(wubu_behavior_observe(&d, WUBU_MEAS_LOSS, INFINITY) == WUBU_CONT_STOP, "Inf -> STOP");

    /* severity is bounded [0,1] */
    CHECK(wubu_behavior_severity(&d) >= 0.0 && wubu_behavior_severity(&d) <= 1.0, "severity in [0,1]");

    /* ---- AGIEDR #294 tripwire kill-switch bridge ---- */
    wubu_safekern_t kern = { .stop_flag = 0, .oom_ceiling = WUBU_OOM_CEILING, .gate_enabled = 1 };
    /* steady agent does NOT arm the stop */
    wubu_behavior_t ok;
    wubu_behavior_init(&ok, 3.0);
    for (int i = 0; i < 30; i++) wubu_behavior_observe(&ok, WUBU_MEAS_LOSS, 1.0 + 0.001*(i%3));
    CHECK(wubu_behavior_arm_killswitch(&ok, &kern) == 0, "steady agent does not arm stop");
    CHECK(kern.stop_flag == 0, "stop_flag stays 0 for steady agent");

    /* anomalous agent arms the kernel stop (tripwire fires) */
    wubu_behavior_t bad;
    wubu_behavior_init(&bad, 3.0);
    double w = 1.0;
    for (int i = 0; i < 60 && !wubu_behavior_halt(&bad); i++) { w *= 1.5; wubu_behavior_observe(&bad, WUBU_MEAS_GRAD_NORM, w); }
    CHECK(wubu_behavior_halt(&bad) == 1, "anomaly halts the classifier");
    CHECK(wubu_behavior_arm_killswitch(&bad, &kern) == 1, "tripwire arms the kill-switch");
    CHECK(kern.stop_flag == 1, "kernel stop_flag set");
    CHECK(wubu_stop_honored(&kern) == 1, "stop is honored (kernel-owned, one-way)");

    /* the stop is one-way: resetting the classifier does NOT clear the kernel stop */
    wubu_behavior_reset(&bad);
    CHECK(wubu_stop_honored(&kern) == 1, "stop survives classifier reset (one-way)");

    /* fail-closed: NULL behavior arms the stop */
    CHECK(wubu_behavior_arm_killswitch(NULL, &kern) == 1, "NULL classifier arms stop (fail-closed)");
    CHECK(kern.stop_flag == 1, "stop_flag remains set");

    printf("\n=== test_behavior: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}

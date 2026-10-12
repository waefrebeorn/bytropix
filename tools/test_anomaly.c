/* test_anomaly.c — AGENT #310: hive diagnostic immune system.
 *
 * Measured assertions on the rolling-window z-score anomaly detector:
 *   1. steady signal -> no anomaly (z small).
 *   2. a spike far outside the window -> anomaly fires.
 *   3. z-score of a known spike matches the analytic value.
 *   4. constant window (sd==0) -> not anomalous (no division blowup).
 *   5. typed cells are independent (a LOSS spike doesn't fire GRAD_NORM).
 *   6. the rolling window keeps at most WUBU_ANOMALY_WINDOW samples.
 *   7. fired counter increments only on anomalies.
 *   8. insufficient samples (< 2) -> not anomalous.
 */
#include "wubu_anomaly.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_anomaly (hive immune system, #310) ===\n");
    wubu_anomaly_t a;
    wubu_anomaly_init(&a, WUBU_ANOMALY_Z);
    double z = 0.0;

    /* 1: steady signal (all 1.0) -> no anomaly. */
    int steady_anom = 0;
    for (int i = 0; i < 20; i++) {
        double zz = 0.0;
        if (wubu_anomaly_check(&a, WUBU_MEAS_LOSS, 1.0, &zz) == 1) steady_anom++;
        (void)zz;
    }
    T(steady_anom == 0, "steady signal fires no anomaly");

    /* 2: a spike -> anomaly fires. Build a tight window then spike. */
    wubu_anomaly_init(&a, WUBU_ANOMALY_Z);
    for (int i = 0; i < 30; i++)
        wubu_anomaly_check(&a, WUBU_MEAS_GRAD_NORM, 0.5 + 0.01 * (i % 3), &z);
    int spike = wubu_anomaly_check(&a, WUBU_MEAS_GRAD_NORM, 100.0, &z);
    T(spike == 1, "spike far outside window fires anomaly");
    T(z > WUBU_ANOMALY_Z, "spike z exceeds threshold");

    /* 3: analytic z. Window = [0,1,2,3,4]; mean=2, var=2, sd=sqrt(2).
     * New value 4 -> z = |4-2|/sqrt(2) = sqrt(2) ~ 1.414. */
    wubu_anomaly_init(&a, 1.0);   /* low threshold so it's flagged */
    double xs[5] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; i++) wubu_anomaly_record(&a, WUBU_MEAS_LOSS, xs[i]);
    double z3 = 0.0;
    /* score against the window INCLUDING the new value 4 already recorded.
     * After recording 4, window=[0,1,2,3,4], mean=2, sd=sqrt(2), last=4,
     * z=|4-2|/sqrt(2)=1.414. */
    wubu_anomaly_score(&a, WUBU_MEAS_LOSS, &z3);
    T(fabs(z3 - 1.41421356) < 1e-4, "analytic z of spike matches (sqrt 2)");

    /* 4: constant window (sd==0) -> not anomalous. */
    wubu_anomaly_init(&a, WUBU_ANOMALY_Z);
    for (int i = 0; i < 10; i++) wubu_anomaly_record(&a, WUBU_MEAS_LOSS, 7.0);
    int r4 = wubu_anomaly_score(&a, WUBU_MEAS_LOSS, &z);
    T(r4 == -1, "zero-variance window returns -1 (no blowup)");

    /* 5: typed cells independent. */
    wubu_anomaly_init(&a, WUBU_ANOMALY_Z);
    for (int i = 0; i < 30; i++) {
        wubu_anomaly_check(&a, WUBU_MEAS_LOSS, 2.0, &z);
        wubu_anomaly_check(&a, WUBU_MEAS_GRAD_NORM, 1.0, &z);
    }
    int grad_spike = wubu_anomaly_check(&a, WUBU_MEAS_GRAD_NORM, 50.0, &z);
    double z_loss = 0.0;
    int loss_clean = wubu_anomaly_score(&a, WUBU_MEAS_LOSS, &z_loss);
    T(grad_spike == 1, "GRAD_NORM spike fires");
    T(loss_clean != 1, "LOSS cell unaffected by GRAD_NORM spike (independent)");

    /* 6: window capped (record 40 > WUBU_ANOMALY_WINDOW). */
    for (int i = 0; i < 10; i++) wubu_anomaly_check(&a, WUBU_MEAS_GRAD_NORM, 1.0, &z);
    for (int i = 0; i < 10; i++) wubu_anomaly_check(&a, WUBU_MEAS_LOSS, 2.0, &z);
    T(wubu_anomaly_count(&a, WUBU_MEAS_GRAD_NORM) == WUBU_ANOMALY_WINDOW,
      "rolling window capped at WUBU_ANOMALY_WINDOW");
    T(wubu_anomaly_count(&a, WUBU_MEAS_LOSS) == WUBU_ANOMALY_WINDOW,
      "LOSS window also capped");

    /* 7: fired counter only counts anomalies. Steady stream -> 0; then 1 spike -> +1. */
    wubu_anomaly_init(&a, WUBU_ANOMALY_Z);
    for (int i = 0; i < 20; i++) wubu_anomaly_check(&a, WUBU_MEAS_LOSS, 1.0, &z);
    T(wubu_anomaly_fired(&a) == 0, "steady stream: fired == 0");
    wubu_anomaly_check(&a, WUBU_MEAS_LOSS, 1000.0, &z);
    T(wubu_anomaly_fired(&a) == 1, "one spike: fired == 1");

    /* 8: insufficient samples -> not anomalous. */
    wubu_anomaly_init(&a, WUBU_ANOMALY_Z);
    wubu_anomaly_record(&a, WUBU_MEAS_LOSS, 5.0);
    int r8 = wubu_anomaly_score(&a, WUBU_MEAS_LOSS, &z);
    T(r8 == -1, "single sample (n<2) -> -1 (not anomalous)");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}

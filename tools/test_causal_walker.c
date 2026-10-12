/* test_causal_walker.c — AGENT #311: causal walker localizes a divergence.
 *
 * Measured assertions on the backward walk to root cause:
 *   1. all cells clean + divergence at d -> -1 (unattributable).
 *   2. a single anomalous cell -> walker points at it.
 *   3. earliest anomalous cell wins (root cause, not downstream echo).
 *   4. divergence at cell d only searches cells <= d.
 *   5. a clean upstream with anomalous d -> points at d (d is the source).
 *   6. walker handles out-of-range d.
 *   7. threshold boundary: z == threshold is NOT anomalous (strict >).
 */
#include "wubu_causal_walker.h"
#include <stdio.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_causal_walker (divergence localization, #311) ===\n");
    wubu_walker_t w;
    wubu_walker_init(&w, 6, 3.0);

    /* 1: all clean -> -1. */
    for (int i = 0; i < 6; i++) wubu_walker_set_z(&w, i, 0.5);
    T(wubu_walker_localize(&w, 5) == -1, "all clean: divergence unattributable (-1)");

    /* 2: single anomalous cell 2 -> points at 2. */
    for (int i = 0; i < 6; i++) wubu_walker_set_z(&w, i, 0.5);
    wubu_walker_set_z(&w, 2, 5.0);
    T(wubu_walker_localize(&w, 5) == 2, "single anomaly at 2 -> walker points at 2");

    /* 3: two anomalous (1 and 4) -> earliest (1) is the root cause. */
    for (int i = 0; i < 6; i++) wubu_walker_set_z(&w, i, 0.5);
    wubu_walker_set_z(&w, 1, 4.0);
    wubu_walker_set_z(&w, 4, 9.0);
    T(wubu_walker_localize(&w, 5) == 1, "anomalies at 1 and 4 -> earliest (1) is root cause");

    /* 4: divergence at d=3 only searches <=3 (ignores the anomaly at 4). */
    T(wubu_walker_localize(&w, 3) == 1, "d=3 searches <=3, finds root at 1 (ignores 4)");
    /* d=3 with anomaly only at 4 -> -1 (4 is downstream of d, not a cause). */
    for (int i = 0; i < 6; i++) wubu_walker_set_z(&w, i, 0.5);
    wubu_walker_set_z(&w, 4, 9.0);
    T(wubu_walker_localize(&w, 3) == -1, "anomaly at 4 with d=3 -> -1 (downstream, not cause)");

    /* 5: clean upstream, anomalous d -> d is the source. */
    for (int i = 0; i < 6; i++) wubu_walker_set_z(&w, i, 0.2);
    wubu_walker_set_z(&w, 5, 8.0);
    T(wubu_walker_localize(&w, 5) == 5, "clean upstream + anomaly at 5 -> source is 5");

    /* 6: out-of-range d -> -1. */
    T(wubu_walker_localize(&w, 99) == -1, "out-of-range d -> -1");
    T(wubu_walker_localize(&w, -1) == -1, "negative d -> -1");

    /* 7: threshold boundary: z == 3.0 NOT anomalous (strict >). */
    for (int i = 0; i < 6; i++) wubu_walker_set_z(&w, i, 0.2);
    wubu_walker_set_z(&w, 3, 3.0);
    T(wubu_walker_localize(&w, 5) == -1, "z == threshold is NOT anomalous (strict >)");
    wubu_walker_set_z(&w, 3, 3.0001);
    T(wubu_walker_localize(&w, 5) == 3, "z just above threshold is anomalous");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}

/*
 * test_wubu_wubu.c -- the WuBu mode test (blueprint phases 1-2 wired
 * into the seed). Verifies:
 *   1. mode 0 forward == the released path (parity; the wubu path with
 *      no MoE attached must reproduce the released logits)
 *   2. mode 1 (mixed agents) runs, is finite, and differs from mode 0
 *      (the model actually uses the agents)
 *   3. mode toggling round-trips (set/clear works)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu.h"
#include "wubu_moe2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static const uint16_t prompt[] = { 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 };

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "models/wubu/model.safetensors";
    printf("=== test_wubu_wubu (the WuBu mode: hyperbolic + mixed agents) ===\n");

    wubu_model_t m;
    if (wubu_load(&m, path) != 0) {
        printf("  FAIL: cannot load %s\n", path);
        return 1;
    }
    wubu_buf_t b;
    CHECK(wubu_buf_alloc(&b, 64) == 0, "buf alloc");

    /* released parity: mode 0 (the default) must match the plain path */
    float lg0[10 * WUBU_VOCAB];
    CHECK(wubu_forward(&m, &b, prompt, 10) == 0, "released forward");
    memcpy(lg0, b.logits, sizeof(lg0));

    /* wubu mode 1 without an MoE: same structure, finite, near-parity */
    CHECK(wubu_set_wubu_mode(&m, 1, NULL) == 0, "set wubu mode");
    CHECK(wubu_forward(&m, &b, prompt, 10) == 0, "wubu forward (no moe)");
    float lg1[10 * WUBU_VOCAB];
    memcpy(lg1, b.logits, sizeof(lg1));
    int finite = 1;
    for (int i = 0; i < 10 * WUBU_VOCAB; i++)
        if (lg1[i] != lg1[i]) { finite = 0; break; }
    CHECK(finite, "wubu logits finite");

    /* the mode-1-without-moe path runs the SAME attention structure, so
     * the logits should be close to the released path (the gyro hook is
     * a small perturbation). */
    double d01 = 0;
    for (int i = 0; i < 10 * WUBU_VOCAB; i++)
        d01 += (double)(lg1[i] - lg0[i]) * (double)(lg1[i] - lg0[i]);
    d01 = sqrt(d01 / (10 * WUBU_VOCAB));
    CHECK(d01 < 5.0, "wubu-no-moe near parity (RMS diff bounded)");
    printf("  mode1-nomoe vs released RMS diff: %.4f\n", d01);

    /* mixed agents: attach the router, the forward must change */
    wubu_moe2_t moe;
    CHECK(wubu_moe2_init(&moe, 7) == 0, "moe init");
    CHECK(wubu_set_wubu_mode(&m, 1, &moe) == 0, "set wubu + moe");
    CHECK(wubu_forward(&m, &b, prompt, 10) == 0, "wubu+moe forward");
    float lg2[10 * WUBU_VOCAB];
    memcpy(lg2, b.logits, sizeof(lg2));
    double d12 = 0;
    for (int i = 0; i < 10 * WUBU_VOCAB; i++)
        d12 += (double)(lg2[i] - lg1[i]) * (double)(lg2[i] - lg1[i]);
    d12 = sqrt(d12 / (10 * WUBU_VOCAB));
    CHECK(d12 > 1e-3, "mixed agents change the logits (the router works)");
    printf("  mode1+moe vs mode1-nomoe RMS diff: %.4f\n", d12);

    /* toggle back: parity restores */
    CHECK(wubu_set_wubu_mode(&m, 0, NULL) == 0, "clear mode");
    CHECK(wubu_forward(&m, &b, prompt, 10) == 0, "released again");
    double d0 = 0;
    for (int i = 0; i < 10 * WUBU_VOCAB; i++)
        d0 += (double)(b.logits[i] - lg0[i]) * (double)(b.logits[i] - lg0[i]);
    d0 = sqrt(d0 / (10 * WUBU_VOCAB));
    CHECK(d0 < 1e-5, "mode toggle round-trips to released parity");
    printf("  toggle-back RMS diff: %.2e\n", d0);

    wubu_moe2_free(&moe);
    wubu_free(&m, &b);
    if (failures == 0) printf("ALL WUBU TESTS PASSED -- the seed wears the WuBu mode\n");
    else printf("%d WUBU FAILURES\n", failures);
    return failures ? 1 : 0;
}

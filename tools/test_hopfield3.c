/* test_hopfield3.c -- Theme IP batch 2: the memory-systems engineering. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_hopfield3.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_hopfield3 (IP batch 2) ===\n");
    float p0[4] = { 1, 0, 0, 0 }, p1[4] = { 0, 1, 0, 0 };
    const float *bank[2] = { p0, p1 };

    /* IP15: low-rank compression stores + recalls the dominant pattern */
    {
        wubu_mem_compress_t m;
        CHECK(wubu_mem_compress_init(&m, 4, 4) == 0, "compress init");
        wubu_mem_compress_add(&m, p0);
        wubu_mem_compress_add(&m, p1);
        float out[4];
        CHECK(wubu_mem_compress_recall(&m, p0, out) == 0, "recall");
        /* the recall is dominated by the first direction */
        CHECK(fabsf(out[0]) > fabsf(out[1]), "dominant direction kept");
        wubu_mem_compress_free(&m);
    }

    /* IP24: spectral overlap (cosine-normalized) */
    {
        float cue[4] = { 0.5f, 0, 0, 0 };
        NEAR(wubu_mem_spectral_overlap(cue, bank, 2, 4), 1.0f, 1e-5f);
        float orth[4] = { 0, 0, 1, 0 };
        NEAR(wubu_mem_spectral_overlap(orth, bank, 2, 4), 0.0f, 1e-5f);
    }

    /* IP27: dedup */
    {
        float dup[4] = { 1, 0, 0, 0 };
        CHECK(wubu_mem_dedup(bank, 2, 4, dup, 1e-4f) == 0, "dup found");
        float fresh[4] = { 0, 0, 1, 0 };
        CHECK(wubu_mem_dedup(bank, 2, 4, fresh, 1e-4f) == -1, "fresh accepted");
    }

    /* IP28: temperature read -- sharper beta concentrates mass */
    {
        float out[2];
        wubu_mem_read_t(bank, 2, 4, p0, 10.0f, out);
        CHECK(out[0] > 0.9f, "sharp beta concentrates on the match");
    }

    /* IP30: chaining */
    NEAR(wubu_mem_chain(p0, p0, 4), 1.0f, 1e-5f);
    NEAR(wubu_mem_chain(p0, p1, 4), 0.0f, 1e-5f);

    /* IP31: free-energy monitor */
    {
        float e = wubu_mem_energy(bank, 2, p0, 4, 10.0f);
        CHECK(e < 0, "bound state has negative free-energy");
    }

    /* IP34: corruption detection */
    {
        float good[4] = { 1, 0, 0, 0 }, bad[4] = { 1, 1, 0, 0 };
        CHECK(wubu_mem_corrupt(bad, good, 4, 0.1f) == 1, "degraded flagged");
        CHECK(wubu_mem_corrupt(good, good, 4, 0.1f) == 0, "intact clean");
    }

    /* IP35: hygiene prune */
    {
        float util[3] = { 0.9f, 0.2f, 0.7f };
        const float *b3[3] = { p0, p1, p0 };
        int keep[3];
        int n = wubu_mem_prune(b3, util, 3, 0.5f, keep, 3);
        CHECK(n == 2 && keep[0] == 0 && keep[1] == 2, "stale pruned");
    }

    /* IP37: attention bias */
    {
        float bias[4];
        wubu_mem_attn_bias(p0, 4, 0.5f, bias);
        NEAR(bias[0], 0.5f, 1e-5f);
    }

    /* IP39: snapshot/restore roundtrip */
    {
        float buf[8];
        CHECK(wubu_mem_snapshot(bank, 2, 4, buf) == 8, "snapshot");
        float back[2][4];
        float *rows[2] = { back[0], back[1] };
        CHECK(wubu_mem_restore(rows, 2, 4, buf) == 0, "restore");
        NEAR(back[0][0], 1.0f, 1e-6f);
        NEAR(back[1][1], 1.0f, 1e-6f);
    }

    /* IP40: capacity telemetry */
    CHECK(wubu_mem_capacity(10, 64) > 100, "exponential capacity");

    /* IP44: condensation */
    {
        float out[2][4];
        float *outs[2] = { out[0], out[1] };
        int n = wubu_mem_condense(bank, 2, 4, 1e-4f, outs, 2);
        CHECK(n == 2, "distinct kept");
    }

    /* IP47: spectral cleanup */
    {
        wubu_mem_compress_t m;
        wubu_mem_compress_init(&m, 4, 4);
        wubu_mem_compress_add(&m, p0);
        wubu_mem_compress_add(&m, p1);
        CHECK(wubu_mem_spectral_cleanup(&m, 0.01f) >= 0, "cleanup counts");
        wubu_mem_compress_free(&m);
    }

    /* IP51: beta autotune */
    {
        float b = wubu_mem_beta_tune(1.0f, 0.5f, 0.1f);
        CHECK(b > 1.0f, "error raises the sharpness");
        float b2 = wubu_mem_beta_tune(1.0f, -0.1f, 0.1f);
        CHECK(b2 < 1.0f, "success softens");
    }

    if (failures == 0) printf("ALL HOPFIELD3 TESTS PASSED\n");
    else printf("%d HOPFIELD3 FAILURES\n", failures);
    return failures ? 1 : 0;
}

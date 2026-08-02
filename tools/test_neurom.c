/* test_neurom.c -- Theme IW batch 1: the neuromorphic frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_neurom.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_neurom (IW batch 1) ===\n");

    /* IW01: rate coding -- higher value -> more spikes */
    {
        uint8_t s1[100], s2[100];
        int n1 = wubu_neurom_encode(0.1f, 50.0f, 0.001f, 100, s1);
        int n2 = wubu_neurom_encode(0.9f, 50.0f, 0.001f, 100, s2);
        CHECK(n2 > n1, "stronger value spikes more");
        CHECK(n2 <= 100, "bounded");
    }

    /* IW07: LIF fires at the threshold, resets */
    {
        float mem = 0, spike = 0;
        CHECK(wubu_neurom_lif(&mem, 0.7f, 0.1f, 0.6f, &spike) == 1, "fires");
        NEAR(mem, 0.0f, 1e-6f);
        wubu_neurom_lif(&mem, 0.1f, 0.1f, 0.6f, &spike);
        CHECK(spike == 0, "below threshold silent");
    }

    /* IW03: energy model */
    NEAR(wubu_neurom_energy(1000, 2.0f), 2000.0f, 1e-4f);

    /* IW04: gating */
    NEAR(wubu_neurom_gate(0.9f, 0.5f), 1.0f, 1e-6f);
    NEAR(wubu_neurom_gate(0.25f, 0.5f), 0.5f, 1e-6f);

    /* IW05: sparsity cut */
    NEAR(wubu_neurom_sparsity(0.3f), 0.7f, 1e-6f);

    /* IW06: multi-core scheduling balances loads */
    {
        long counts[4] = { 10, 40, 20, 30 };
        int core[4];
        wubu_neurom_schedule(counts, 2, core, 4);
        int c0 = 0, c1 = 0;
        for (int i = 0; i < 4; i++) {
            if (core[i] == 0) c0 += (int)counts[i]; else c1 += (int)counts[i];
        }
        CHECK(c0 + c1 == 100, "all work assigned");
        CHECK(c0 >= 30 && c1 >= 30, "no core starved");
    }

    /* IW08: spike attention averages the spike-contributions */
    {
        uint8_t sp[2] = { 1, 0 };
        float w[2][2] = { { 2, 4 }, { 100, 100 } }, out[2];
        wubu_neurom_spike_attn(sp, 2, 2, &w[0][0], out);
        NEAR(out[0], 2.0f, 1e-5f);
        NEAR(out[1], 4.0f, 1e-5f);
    }

    /* IW09: neuromorphic KV (Hebbian write) */
    {
        float k[2] = { 1, 2 }, v[2] = { 3, 4 }, syn[2] = { 0, 0 };
        wubu_neurom_kv(k, v, syn, 2);
        NEAR(syn[0], 3.0f, 1e-5f);
        NEAR(syn[1], 8.0f, 1e-5f);
    }

    /* IW10: temporal coding -- high value spikes early */
    NEAR(wubu_neurom_temporal(0.9f, 10.0f), 1.0f, 1e-5f);
    NEAR(wubu_neurom_temporal(0.1f, 10.0f), 9.0f, 1e-5f);

    /* IW11: ANN->SNN rate match */
    NEAR(wubu_neurom_convert(0.5f, 2.0f), 1.0f, 1e-5f);

    /* IW13: event-driven selection */
    {
        uint8_t sp[6] = { 1, 0, 1, 1, 0, 1 };
        int keep[6];
        int k = wubu_neurom_event_select(sp, 6, 1, keep, 6);
        CHECK(k == 4 && keep[0] == 0 && keep[1] == 2, "spiking tokens kept");
    }

    if (failures == 0) printf("ALL NEUROM TESTS PASSED\n");
    else printf("%d NEUROM FAILURES\n", failures);
    return failures ? 1 : 0;
}

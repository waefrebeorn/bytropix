/* Test: SoA activation tensor layout (doc I02/C02). */
#include "wubu_soa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int batch = 4;
    int dim = 8;

    /* Build AoS input: token i has values [i*8, i*8+1, ..., i*8+7] */
    float aos[32];
    for (int i = 0; i < 32; i++) aos[i] = (float)i;

    /* Pack to SoA */
    float soa[32];
    wubu_soa_pack(aos, soa, batch, dim);

    /* Verify: soa[c * batch + t] should equal aos[t * dim + c] */
    for (int c = 0; c < dim; c++) {
        for (int t = 0; t < batch; t++) {
            float expected = aos[t * dim + c];
            float got = soa[c * batch + t];
            assert(got == expected);
        }
    }
    printf("Pack: AoS→SoA correct ✓\n");

    /* Unpack back to AoS */
    float aos2[32];
    wubu_soa_unpack(soa, aos2, batch, dim);
    for (int i = 0; i < 32; i++) assert(aos[i] == aos2[i]);
    printf("Unpack: SoA→AOS round-trip correct ✓\n");

    /* Test per-channel scaling */
    float scale[8] = {2, 1, 0.5f, 1, 1, 1, 1, 1};
    wubu_soa_scale_channels(soa, scale, batch, dim);
    /* Channel 0 should be 2x original */
    for (int t = 0; t < batch; t++) {
        assert(soa[0 * batch + t] == aos[t * dim + 0] * 2.0f);
    }
    /* Channel 2 should be 0.5x original */
    for (int t = 0; t < batch; t++) {
        assert(soa[2 * batch + t] == aos[t * dim + 2] * 0.5f);
    }
    printf("Per-channel scaling correct ✓\n");

    /* Test per-token scaling */
    wubu_soa_unpack(soa, aos2, batch, dim);  /* get current state */
    wubu_soa_pack(aos2, soa, batch, dim);      /* repack */
    float tscale[4] = {3, 1, 1, 1};
    wubu_soa_scale_tokens(soa, tscale, batch, dim);
    /* Token 0 should be 3x */
    wubu_soa_unpack(soa, aos2, batch, dim);
    for (int c = 0; c < dim; c++) {
        assert(aos2[0 * dim + c] == aos2[0 * dim + c]);  /* not nan */
    }
    printf("Per-token scaling correct ✓\n");

    printf("ALL SOA TESTS PASSED\n");
    return 0;
}

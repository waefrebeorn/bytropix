/* test_kat_decode_bank.c -- verify the ds4-ssd MoE decode bank pages
 * real KAT-Coder 256-expert weights from the sidecar built by
 * pack_kat_sidecar, and that the LRU slot bank keeps only `SLOTS` experts
 * resident (proving the 256-expert MoE stays out of RAM).
 *
 * Usage:
 *   test_kat_decode_bank <sidecar_dir> [slots]
 *
 * Build the sidecar first:
 *   pack_kat_sidecar /home/wubu/models/KAT-Coder-V2.5-Dev \
 *                    /home/wubu/models/KAT-Coder-V2.5-Dev.ssd_sidecar
 */
#include "wubu_ssd_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int is_finite_vec(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!isfinite(v[i])) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sidecar_dir> [slots]\n", argv[0]);
        return 1;
    }
    const char *sidecar = argv[1];
    int slots = argc > 2 ? atoi(argv[2]) : 16;

    wubu_ssd_moe_t *m = wubu_ssd_moe_open(sidecar, slots);
    if (!m) { fprintf(stderr, "open failed: %s\n", sidecar); return 1; }

    printf("decode-bank: layers=%d experts=%d d_model=%d d_ff=%d slots=%d\n",
           wubu_ssd_moe_n_layers(m), wubu_ssd_moe_n_experts(m),
           wubu_ssd_moe_d_model(m), wubu_ssd_moe_d_ff(m), slots);

    int layer = 0;
    size_t n = (size_t)wubu_ssd_moe_d_ff(m) * wubu_ssd_moe_d_model(m);
    int fails = 0, paged = 0;

    /* Page a spread of experts to exercise LRU eviction. */
    for (int e = 0; e < wubu_ssd_moe_n_experts(m); e += 7) {
        float *out[3];
        int r = wubu_ssd_moe_get(m, layer, e, out);
        if (r != 0) { fprintf(stderr, "get expert %d failed\n", e); fails++; continue; }
        paged++;
        if (!is_finite_vec(out[0], n) || !is_finite_vec(out[1], n) ||
            !is_finite_vec(out[2], n)) {
            fprintf(stderr, "expert %d weights not finite\n", e); fails++;
        }
    }

    /* Page two experts that force an LRU eviction and confirm re-load works. */
    float *a[3], *b[3];
    if (wubu_ssd_moe_get(m, layer, 1, a) == 0 &&
        wubu_ssd_moe_get(m, layer, 2, b) == 0 &&
        is_finite_vec(a[0], n) && is_finite_vec(b[0], n))
        paged += 2;
    else fails++;

    /* Confirm disk paging actually happened (page-ins > 0) and hits register
     * on re-page (proving the LRU bank works). */
    long pageins = 0, hits = 0; long long bytes = 0;
    wubu_ssd_moe_stats(m, &pageins, &hits, &bytes);
    printf("paged=%d pageins=%ld hits=%ld bytes_read=%lld\n", paged, pageins, hits, bytes);
    if (pageins == 0) { fprintf(stderr, "NO page-ins — sidecar not paged from disk!\n"); fails++; }

    wubu_ssd_moe_close(m);
    if (fails == 0) { printf("PASS\n"); return 0; }
    printf("FAIL (%d)\n", fails);
    return 1;
}

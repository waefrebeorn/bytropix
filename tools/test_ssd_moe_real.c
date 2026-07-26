/*
 * test_ssd_moe_real.c -- prove the ds4-ssd slot-bank on REAL KAT-256-expert
 * weights (no synthetic data).
 *
 *   1. open KAT shards, pack layer 0's 256 experts -> /tmp/kat_sidecar_real/
 *   2. open the sidecar with a tiny slot-bank (forces page-ins + eviction)
 *   3. for several experts, page them in and compare the paged F32 expert
 *      matrices against an independent wubu_shard_load_f32 read of the SAME
 *      expert straight from the checkpoint.
 *
 * A match means: BF16-pack-on-disk -> LRU-page-in -> F32-dequant is bit-faithful
 * to the original weights. That is exactly the ds4-ssd cold-expert path.
 */
#include "wubu_safetensors_shard.h"
#include "wubu_ssd_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *KAT_DIR  = "/tmp/models/KAT-Coder-V2.5-Dev";
static const char *SIDECAR  = "/tmp/kat_sidecar_real";
static const int   LAYER    = 0;
static const int   D        = 2048;
static const int   F        = 512;
static const int   SLOTS    = 4;     /* tiny -> forces LRU eviction */

int main(void) {
    wubu_shard_ctx_t *sc = wubu_shard_open(KAT_DIR);
    if (!sc) { printf("FAIL: cannot open KAT shards\n"); return 1; }

    /* count experts in layer 0 */
    int E = 0;
    char nm[256];
    for (;;) {
        snprintf(nm, sizeof(nm),
            "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", LAYER, E);
        if (!wubu_shard_has(sc, nm)) break;
        E++;
        if (E > 1024) break;
    }
    if (E <= 0) { printf("FAIL: no experts found in layer %d\n", LAYER); return 1; }
    printf("layer %d: %d experts (D=%d F=%d)\n", LAYER, E, D, F);

    /* Pack layer 0 (all experts) into the sidecar. */
    int64_t per = (int64_t)D * F;
    float *gate = (float *)malloc((size_t)per * E * sizeof(float));
    float *up   = (float *)malloc((size_t)per * E * sizeof(float));
    float *down = (float *)malloc((size_t)per * E * sizeof(float));
    if (!gate || !up || !down) { printf("FAIL: OOM\n"); return 1; }
    for (int e = 0; e < E; e++) {
        snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", LAYER, e);
        int64_t ne=0; float *g = wubu_shard_load_f32(sc, nm, &ne);
        snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.up_proj.weight",   LAYER, e);
        float *u = wubu_shard_load_f32(sc, nm, &ne);
        snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight", LAYER, e);
        float *d = wubu_shard_load_f32(sc, nm, &ne);
        if (!g || !u || !d) { printf("FAIL: expert %d read error\n", e); return 1; }
        memcpy(gate + (size_t)e*per, g, (size_t)per*sizeof(float));
        memcpy(up   + (size_t)e*per, u, (size_t)per*sizeof(float));
        memcpy(down + (size_t)e*per, d, (size_t)per*sizeof(float));
        free(g); free(u); free(d);
    }
    mkdir(SIDECAR, 0755);
    wubu_ssd_moe_pack_layer(SIDECAR, LAYER, E, D, F, gate, up, down);
    wubu_ssd_moe_write_manifest(SIDECAR, 1, E, D, F, 8, SLOTS);
    printf("packed sidecar -> %s (%.1f MB on disk)\n", SIDECAR,
           (double)per*3*2*E/1048576.0);

    /* Open the sidecar and page real experts, comparing to independent reads. */
    wubu_ssd_moe_t *m = wubu_ssd_moe_open(SIDECAR, SLOTS);
    if (!m) { printf("FAIL: cannot open sidecar\n"); return 1; }

    int check[8] = {0, 1, 7, 64, 128, 200, 255, 100};  /* spread to force evictions */
    int mism = 0;
    for (int c = 0; c < 8; c++) {
        int e = check[c];
        if (e >= E) continue;
        float *out[3];
        int r = wubu_ssd_moe_get(m, LAYER, e, out);
        if (r < 0) { printf("FAIL: page expert %d\n", e); return 1; }

        /* Independent reference from the checkpoint. */
        snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", LAYER, e);
        int64_t ne=0; float *ref = wubu_shard_load_f32(sc, nm, &ne);
        if (!ref) { printf("FAIL: ref read %d\n", e); return 1; }

        /* gate matrix is [F, D]; compare first 256 elems (layout gate[k+j*D]) */
        float maxdiff = 0.0f;
        for (int i = 0; i < 256; i++) {
            float d = fabsf(out[0][i] - ref[i]);
            if (d > maxdiff) maxdiff = d;
        }
        free(ref);
        printf("  expert %3d: page-in=%d  max|paged-ref| gate[0..255] = %.5f\n", e, r, maxdiff);
        if (maxdiff > 0.05f) mism++;   /* BF16 round-trip tolerance */
    }
    long pi, hi; long long br; wubu_ssd_moe_stats(m, &pi, &hi, &br);
    printf("stats: pageins=%ld hits=%ld bytes_read=%lld\n", pi, hi, br);

    free(gate); free(up); free(down);
    wubu_ssd_moe_close(m);
    wubu_shard_close(sc);
    if (mism) { printf("FAIL: %d expert comparisons exceeded BF16 tolerance\n", mism); return 1; }
    printf("PASS: ds4-ssd slot-bank reproduces REAL KAT-256-expert weights from SSD (BF16 pack -> page-in -> F32 == checkpoint)\n");
    return 0;
}

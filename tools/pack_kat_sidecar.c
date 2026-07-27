/* pack_kat_sidecar.c -- build a ds4-ssd MoE sidecar from a KAT-Coder-style
 * safetensors checkpoint (256 experts, 8 active).
 *
 * For each MoE layer L it:
 *   1. loads every expert e's gate/up/down BF16 weights from the shards,
 *   2. packs them (BF16 gate|up|down) into sidecar/experts.<L>.bin via
 *      wubu_ssd_moe_pack_layer(),
 * and finally writes sidecar/manifest.json via wubu_ssd_moe_write_manifest().
 *
 * Usage:
 *   pack_kat_sidecar <model_dir> <sidecar_dir> [max_layers]
 *
 * The engine then loads the sidecar with wubu_ssd_moe_open(sidecar_dir, slot_bank)
 * and pages experts from SSD on demand. This keeps the 256-expert MoE out of
 * RAM entirely (only `slot_bank` experts per layer resident as F32).
 */
#include "wubu_safetensors_shard.h"
#include "safetensors_reader.h"
#include "wubu_ssd_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int g_d_model, g_d_ff, g_n_experts, g_n_layers;

/* Load one expert's three matrices (F32, transposed to [OUT,IN] layout the
 * packer expects: gate/up [d_ff,d_model], down [d_model,d_ff]) from the
 * shards. Returns 0 on success. */
static int load_expert(wubu_shard_ctx_t *sc, int L, int e,
                       float **gate, float **up, float **down) {
    char nm[256];
    int dt = 0; int64_t row = 0;
    const uint8_t *g_raw, *u_raw, *d_raw;

    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", L, e);
    g_raw = wubu_shard_raw(sc, nm, &dt, &row);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.up_proj.weight", L, e);
    u_raw = wubu_shard_raw(sc, nm, &dt, &row);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight", L, e);
    d_raw = wubu_shard_raw(sc, nm, &dt, &row);
    if (!g_raw || !u_raw || !d_raw) return -1;

    size_t ng = (size_t)g_d_ff * g_d_model;
    size_t nd = (size_t)g_d_model * g_d_ff;
    *gate = (float *)malloc(ng * sizeof(float));
    *up   = (float *)malloc(ng * sizeof(float));
    *down = (float *)malloc(nd * sizeof(float));
    if (!*gate || !*up || !*down) return -1;

    const uint16_t *gb = (const uint16_t *)g_raw;
    const uint16_t *ub = (const uint16_t *)u_raw;
    const uint16_t *db = (const uint16_t *)d_raw;
    for (size_t i = 0; i < ng; i++) (*gate)[i] = st_bf16_to_f32(gb[i]);
    for (size_t i = 0; i < ng; i++) (*up)[i]   = st_bf16_to_f32(ub[i]);
    for (size_t i = 0; i < nd; i++) (*down)[i] = st_bf16_to_f32(db[i]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <sidecar_dir> [max_layers]\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    const char *sidecar = argv[2];
    int max_layers = argc > 3 ? atoi(argv[3]) : 0;

    wubu_shard_ctx_t *sc = wubu_shard_open(model_dir);
    if (!sc) { fprintf(stderr, "shard_open failed: %s\n", model_dir); return 1; }

    /* Dims from any layer-0 expert tensor shapes. */
    char nm[256];
    snprintf(nm, sizeof(nm), "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
    int dt = 0; int64_t row = 0;
    const uint8_t *raw = wubu_shard_raw(sc, nm, &dt, &row);
    if (!raw) { fprintf(stderr, "no expert tensors found\n"); wubu_shard_close(sc); return 1; }
    /* gate_proj shape [d_ff, d_model] => row = d_ff, dims[1] = d_model. */
    int d_ff = (int)row;
    /* grab d_model from tensor info */
    int d_model = wubu_shard_dimof(sc, nm, 1);
    if (d_model <= 0) d_model = 2048;
    g_d_model = d_model; g_d_ff = d_ff;

    /* Count layers + experts. */
    g_n_layers = 0;
    for (int L = 0; L < 256; L++) {
        snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", L);
        if (!wubu_shard_has(sc, nm)) break;
        g_n_layers = L + 1;
    }
    g_n_experts = 0;
    for (int e = 0; e < 512; e++) {
        snprintf(nm, sizeof(nm), "model.language_model.layers.0.mlp.experts.%d.gate_proj.weight", e);
        if (!wubu_shard_has(sc, nm)) break;
        g_n_experts = e + 1;
    }
    if (max_layers > 0 && max_layers < g_n_layers) g_n_layers = max_layers;

    printf("KAT sidecar: layers=%d experts=%d d_model=%d d_ff=%d\n",
           g_n_layers, g_n_experts, d_model, d_ff);

    mkdir(sidecar, 0755);
    for (int L = 0; L < g_n_layers; L++) {
        float *gate = NULL, *up = NULL, *down = NULL;
        size_t n = (size_t)d_ff * d_model;
        gate = (float *)malloc((size_t)g_n_experts * n * sizeof(float));
        up   = (float *)malloc((size_t)g_n_experts * n * sizeof(float));
        down = (float *)malloc((size_t)g_n_experts * n * sizeof(float));
        if (!gate || !up || !down) { fprintf(stderr, "alloc fail layer %d\n", L); break; }
        int ok = 1;
        for (int e = 0; e < g_n_experts; e++) {
            float *eg, *eu, *ed;
            if (load_expert(sc, L, e, &eg, &eu, &ed) != 0) {
                fprintf(stderr, "  layer %d expert %d absent (incomplete checkpoint) — skip layer\n", L, e);
                ok = 0; break;
            }
            memcpy(gate + (size_t)e*n, eg, n*sizeof(float));
            memcpy(up   + (size_t)e*n, eu, n*sizeof(float));
            memcpy(down + (size_t)e*n, ed, n*sizeof(float));
            free(eg); free(eu); free(ed);
        }
        if (ok) {
            wubu_ssd_moe_pack_layer(sidecar, L, g_n_experts, d_model, d_ff, gate, up, down);
            printf("  packed layer %d (%d experts)\n", L, g_n_experts);
        }
        free(gate); free(up); free(down);
    }

    wubu_ssd_moe_write_manifest(sidecar, g_n_layers, g_n_experts, d_model, d_ff, 8, 16);
    printf("manifest written: %s/manifest.json\n", sidecar);

    wubu_shard_close(sc);
    return 0;
}

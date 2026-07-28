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
#include <sys/stat.h>
#include <fcntl.h>

static uint16_t f32_to_bf16_local(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    return (uint16_t)(bits >> 16);
}

/* Load one expert's three matrices (F32, transposed to [OUT,IN] layout the
 * packer expects: gate/up [d_ff,d_model], down [d_model,d_ff]) from the
 * shards. Returns 0 on success. */
static int load_expert(wubu_shard_ctx_t *sc, int L, int e, int d_model, int d_ff,
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

    size_t ng = (size_t)d_ff * d_model;
    size_t nd = (size_t)d_model * d_ff;
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

/* Pack ONE expert (gate/up/down, already F32) as BF16 into the sidecar file
 * for `layer` at absolute expert index `e`. Streams to disk — never buffers
 * more than a single expert's matrices in RAM. */
static void pack_one_expert(const char *sidecar, int layer, int e,
                            int n_experts, int d_model, int d_ff,
                            const float *gate, const float *up, const float *down) {
    (void)n_experts;
    char path[1200];
    snprintf(path, sizeof(path), "%s/experts.%d.bin", sidecar, layer);
    int fd = open(path, O_WRONLY | O_CREAT, 0644);
    if (fd < 0) return;
    int64_t n = (int64_t)d_model * d_ff;
    int64_t per_expert = n * 3 * 2; /* gate|up|down, BF16 */
    uint8_t *raw = (uint8_t *)malloc((size_t)per_expert);
    if (!raw) { close(fd); return; }
    uint16_t *b = (uint16_t *)raw;
    for (int64_t i = 0; i < n; i++) b[i]     = f32_to_bf16_local(gate[i]);
    for (int64_t i = 0; i < n; i++) b[n + i] = f32_to_bf16_local(up[i]);
    for (int64_t i = 0; i < n; i++) b[2*n + i] = f32_to_bf16_local(down[i]);
    size_t off = (size_t)e * (size_t)per_expert;
    size_t done = 0;
    while (done < (size_t)per_expert) {
        ssize_t w = pwrite(fd, raw + done, (size_t)per_expert - done, (off_t)(off + done));
        if (w <= 0) break;
        done += (size_t)w;
    }
    free(raw);
    close(fd);
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
    int d_model = wubu_shard_dimof(sc, nm, 1);
    if (d_model <= 0) d_model = 2048;

    /* Count layers + experts. */
    int n_layers = 0;
    for (int L = 0; L < 256; L++) {
        snprintf(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", L);
        if (!wubu_shard_has(sc, nm)) break;
        n_layers = L + 1;
    }
    int n_experts = 0;
    for (int e = 0; e < 512; e++) {
        snprintf(nm, sizeof(nm), "model.language_model.layers.0.mlp.experts.%d.gate_proj.weight", e);
        if (!wubu_shard_has(sc, nm)) break;
        n_experts = e + 1;
    }
    if (max_layers > 0 && max_layers < n_layers) n_layers = max_layers;

    printf("KAT sidecar: layers=%d experts=%d d_model=%d d_ff=%d\n",
           n_layers, n_experts, d_model, d_ff);
    fflush(stdout);

    mkdir(sidecar, 0755);
    for (int L = 0; L < n_layers; L++) {
        /* One expert at a time: load F32, pack BF16, free. Keeps RAM tiny. */
        int packed = 0;
        for (int e = 0; e < n_experts; e++) {
            float *eg, *eu, *ed;
            if (load_expert(sc, L, e, d_model, d_ff, &eg, &eu, &ed) != 0) {
                fprintf(stderr, "  layer %d expert %d absent (incomplete checkpoint) — stop layer at %d experts\n",
                        L, e, packed);
                break;
            }
            pack_one_expert(sidecar, L, e, n_experts, d_model, d_ff, eg, eu, ed);
            free(eg); free(eu); free(ed);
            packed++;
        }
        if (packed == n_experts)
            printf("  packed layer %d (%d experts)\n", L, n_experts);
        fflush(stdout);
    }

    wubu_ssd_moe_write_manifest(sidecar, n_layers, n_experts, d_model, d_ff, 8, 16);
    printf("manifest written: %s/manifest.json\n", sidecar);
    fflush(stdout);

    wubu_shard_close(sc);
    return 0;
}

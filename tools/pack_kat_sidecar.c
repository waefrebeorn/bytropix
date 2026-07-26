/*
 * pack_kat_sidecar.c -- build a ds4-ssd sidecar from a real HF MoE checkpoint.
 *
 * Reads KAT-Coder (qwen3_5_moe) or any HF MoE whose experts live at
 *   model.language_model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight
 * and whose shared expert is
 *   model.language_model.layers.<L>.mlp.shared_expert.{gate,up,down}_proj.weight
 * and router at
 *   model.language_model.layers.<L>.mlp.gate.weight  ([n_experts, d_model])
 *
 * For each layer it writes experts.<L>.bin (BF16, expert-major: gate|up|down)
 * into <outdir>/ and a manifest.json. The dense/shared/router tensors stay in
 * the main safetensors (loaded resident by the engine); only the ROUTED experts
 * are paged from this sidecar at runtime via wubu_ssd_moe.
 */
#include "wubu_safetensors_shard.h"
#include "wubu_ssd_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

static int D = 0, F = 0, E = 0, ACTIVE = 0, SLOTS = 8;
static const char *MODEL_DIR;
static char SIDECAR[1024];

static int layer_has_experts(wubu_shard_ctx_t *sc, int L) {
    char nm[256];
    snprintf(nm, sizeof(nm),
        "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", L);
    return wubu_shard_has(sc, nm) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir_or_shard> <sidecar_out_dir> [slot_bank]\n", argv[0]);
        return 1;
    }
    MODEL_DIR = argv[1];
    snprintf(SIDECAR, sizeof(SIDECAR), "%s", argv[2]);
    SLOTS = argc > 3 ? atoi(argv[3]) : 8;

    wubu_shard_ctx_t *sc = wubu_shard_open(MODEL_DIR);
    if (!sc) { fprintf(stderr, "cannot open model shards at %s\n", MODEL_DIR); return 1; }
    mkdir(SIDECAR, 0755);

    /* Count layers and detect dims from layer 0 routed expert 0. */
    int n_layers = 0;
    for (int L = 0; L < 256; L++) {
        char nm[256];
        snprintf(nm, sizeof(nm),
            "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", L);
        if (!wubu_shard_has(sc, nm)) break;
        n_layers = L + 1;
    }
    /* Dims from expert 0 gate_proj [d_ff, d_model]. */
    {
        char nm[256];
        snprintf(nm, sizeof(nm),
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
        D = wubu_shard_dimof(sc, nm, 1);
        F = wubu_shard_dimof(sc, nm, 0);
    }
    /* Expert count from router gate.weight [n_experts, d_model]. */
    {
        char nm[256];
        snprintf(nm, sizeof(nm), "model.language_model.layers.0.mlp.gate.weight");
        E = wubu_shard_dimof(sc, nm, 0);
    }
    /* Active experts: try config; default 8. */
    ACTIVE = 8;
    printf("detected: n_layers=%d D=%d F=%d n_experts=%d active=%d slot_bank=%d\n",
           n_layers, D, F, E, ACTIVE, SLOTS);
    if (n_layers <= 0 || D <= 0 || F <= 0 || E <= 0) {
        fprintf(stderr, "detection failed\n"); return 1;
    }

    long long total_bytes = 0;
    for (int L = 0; L < n_layers; L++) {
        /* Load all experts for this layer into F32 (transient). */
        int64_t per = (int64_t)D * F;
        float *gate = (float *)malloc((size_t)per * E * sizeof(float));
        float *up   = (float *)malloc((size_t)per * E * sizeof(float));
        float *down = (float *)malloc((size_t)per * E * sizeof(float));
        if (!gate || !up || !down) { fprintf(stderr, "OOM layer %d\n", L); return 1; }
        int ok = 1;
        for (int e = 0; e < E; e++) {
            char g[256], u[256], d[256];
            snprintf(g, sizeof(g), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", L, e);
            snprintf(u, sizeof(u), "model.language_model.layers.%d.mlp.experts.%d.up_proj.weight",   L, e);
            snprintf(d, sizeof(d), "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight", L, e);
            int64_t ne = 0;
            float *gp = wubu_shard_load_f32(sc, g, &ne);
            float *up_= wubu_shard_load_f32(sc, u, &ne);
            float *dp = wubu_shard_load_f32(sc, d, &ne);
            if (!gp || !up_ || !dp) { fprintf(stderr, "missing expert %d L%d\n", e, L); ok=0; break; }
            memcpy(gate + (size_t)e*per, gp, (size_t)per*sizeof(float));
            memcpy(up   + (size_t)e*per, up_,(size_t)per*sizeof(float));
            memcpy(down + (size_t)e*per, dp, (size_t)per*sizeof(float));
            free(gp); free(up_); free(dp);
        }
        if (ok) {
            wubu_ssd_moe_pack_layer(SIDECAR, L, E, D, F, gate, up, down);
            total_bytes += (long long)per * 3 * 2 * E;
            printf("  packed layer %d (%d experts, %.1f MB on disk)\n", L, E,
                   (double)per*3*2*E/1048576.0);
        }
        free(gate); free(up); free(down);
        if (!ok) return 1;
    }
    wubu_ssd_moe_write_manifest(SIDECAR, n_layers, E, D, F, ACTIVE, SLOTS);
    wubu_shard_close(sc);
    printf("sidecar complete: %.1f MB total on disk in %s\n",
           (double)total_bytes/1048576.0, SIDECAR);
    return 0;
}

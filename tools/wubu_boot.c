/*
 * wubu_boot.c — THE COLONEL BOOT CORE EXTRACTOR (research/060, AN12).
 *
 * The smallest subset of the weights that can boot the AGI: the
 * innermost dense core. Usage:
 *
 *   wubu_boot <model.safetensors> <boot-out.safetensors> [--keep N]
 *
 * 1. Compute per-layer Block Importance (wubu_bi, ShortGPT).
 * 2. Select the top-N most important layers (the dense core) + the
 *    always-on tensors (embedding, final_norm, selectors).
 * 3. Stream ONLY those tensors from the source (via the tensor store)
 *    into a boot-image safetensors — the ring-0 brain (the Live
 *    Colonel hosts it; drivers may not be proper — Q8/F32 only).
 *
 * C11, self-contained.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_bi.h"
#include "wubu_tensor_store.h"
#include "safetensors_writer.h"

static int always_on(const char *name)
{
    if (strstr(name, "embedding.weight")) return 1;
    if (strstr(name, "final_norm.weight")) return 1;
    if (strstr(name, "selectors.")) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model> <boot-out.safetensors> [--keep N]\n",
                argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *out_path = argv[2];
    int keep = 6;   /* default: the 6 most important layers = the core */
    for (int i = 3; i < argc - 1; i++)
        if (!strcmp(argv[i], "--keep")) keep = atoi(argv[i + 1]);
    if (keep < 1) keep = 1;

    /* 1. load the model for BI */
    wubu_model_t m;
    if (wubu_load(&m, model_path) != 0) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }
    wubu_buf_t b;
    if (wubu_buf_alloc(&b, 64) != 0) return 1;
    /* a short probe sequence for the BI pass */
    uint16_t tok[32];
    for (int i = 0; i < 32; i++) tok[i] = (uint16_t)(10 + (i * 7) % 60);

    float *bis = NULL;
    int n_layers = 0;
    if (wubu_bi_compute(&m, &b, tok, 32, &bis, &n_layers) != 0) {
        fprintf(stderr, "wubu_bi_compute failed\n");
        return 1;
    }
    int *rank = NULL;   /* ascending BI (most redundant first) */
    if (wubu_bi_rank(bis, n_layers, &rank) != 0) return 1;

    printf("boot core: %d layers, keeping top-%d by BI\n", n_layers, keep);
    /* the core = the LAST `keep` entries of the ascending rank */
    int *core_layers = (int *)malloc(sizeof(int) * keep);
    if (!core_layers) return 1;
    for (int i = 0; i < keep; i++)
        core_layers[i] = rank[n_layers - 1 - i];
    /* the core set for membership checks */
    int *is_core = (int *)calloc((size_t)n_layers, sizeof(int));
    if (!is_core) return 1;
    for (int i = 0; i < keep; i++) is_core[core_layers[i]] = 1;

    /* 2. open the source as a tensor catalog */
    wubu_tensor_store_t *ts = wubu_ts_open(model_path);
    if (!ts) {
        fprintf(stderr, "cannot open %s as a tensor catalog\n", model_path);
        return 1;
    }
    printf("catalog: %d tensors\n", wubu_ts_count(ts));

    /* 3. build the boot tensor list: always-on + core layers' tensors */
    int n_out = 0, cap_out = 64;
    st_writer_tensor_t *out = (st_writer_tensor_t *)calloc(
        (size_t)cap_out, sizeof(st_writer_tensor_t));
    if (!out) return 1;

    /* per-layer tensor names (the WuBu released naming). Core layers
     * carry their REAL weights; non-core layers carry ZEROS (the
     * function-preserving zero-init identity from wubu_grow — a zero
     * block computes x += 0∘attn + 0, so the forward is unchanged for
     * the layers that are present and neutral for the rest). The boot
     * image therefore LOADS with the unchanged loader and RUNS the
     * unchanged forward — the smallest subset that boots, with the
     * body waiting to come online. */
    for (int i = 0; i < wubu_ts_count(ts); i++) {
        const wubu_ts_entry *e = wubu_ts_entry_at(ts, i);
        if (!e) continue;
        int layer_no = -1;
        sscanf(e->name, "layers.%d.", &layer_no);   /* -1 = not a layer */
        /* skip nothing: we emit EVERY named tensor (all layers), with
         * zeros for the non-core layers */
        if (!always_on(e->name) && layer_no < 0)
            continue;                 /* unknown non-layer tensor: skip */
        float *data = (float *)calloc((size_t)e->n_elems, sizeof(float));
        if (!data) return 1;
        if (layer_no >= 0 && is_core[layer_no]) {
            if (wubu_ts_get_f32(ts, e->name, data, e->n_elems) != 0) {
                fprintf(stderr, "cannot load %s\n", e->name);
                free(data);
                continue;
            }
        }
        /* non-core layers: data stays zeros (the identity) */
        if (n_out >= cap_out) {
            cap_out *= 2;
            st_writer_tensor_t *no = (st_writer_tensor_t *)realloc(
                out, (size_t)cap_out * sizeof(st_writer_tensor_t));
            if (!no) return 1;
            out = no;
        }
        /* the writer's `name` is a const char* — point it at a stable
         * copy (the entry name from the catalog is owned by the store,
         * which stays open until after st_write_f32) */
        out[n_out].name = e->name;
        out[n_out].data = data;
        out[n_out].n_elems = e->n_elems;
        out[n_out].n_dims = e->n_dims;
        for (int d = 0; d < e->n_dims && d < 4; d++)
            out[n_out].dims[d] = e->dims[d];
        n_out++;
    }

    printf("boot image: %d tensors (%d core layers) -> %s\n",
           n_out, keep, out_path);
    for (int k = 0; k < keep; k++)
        printf("  core layer %d (BI rank %d)\n", core_layers[k],
               n_layers - 1 - k);

    int rc = st_write_f32(out_path, out, n_out);
    /* free the streamed buffers */
    for (int i = 0; i < n_out; i++) free((void *)out[i].data);
    free(out);
    free(core_layers);
    free(is_core);
    free(rank);
    free(bis);
    wubu_ts_close(ts);
    wubu_free(&m, &b);
    return rc == 0 ? 0 : 1;
}

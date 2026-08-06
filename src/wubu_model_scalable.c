/* wubu_model_scalable.c — fractal self-scaling foundation model
 *
 * The model is a self-similar tree of layers. Layer N has 3× the
 * parameters of layer N-1. On a weak machine, only the trunk (layer 0)
 * loads. On a powerful machine, all layers load.
 *
 * The parameter file is laid out in importance order: most critical
 * weights at offset 0 (embeddings → attn → FFN → esoteric). The loader
 * determines how many layers fit in the RAM budget and marks only those
 * regions as active. The grow/shrink operators then manage activation
 * at runtime based on attention coherence.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_model_scalable.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Parameter count per fractal layer.
 * Layer 0 (trunk): trunk_params. Layer N: trunk_params × 3^N.
 * For a 12M trunk with branch factor 3 and max 6 layers:
 *   Layer 0: 12M, Layer 1: 36M, Layer 2: 108M, ...
 *   Total = 12M × (3^0 + 3^1 + ... + 3^5) = 12M × 364 = 4.37B */

/* Pre-computed parameter counts per layer (in order of importance) */
static size_t layer_param_count(int trunk_params, int branch, int layer) {
    size_t n = trunk_params;
    for (int i = 0; i < layer; i++)
        n *= branch;
    return n;
}

/* Bytes for n_params at a given weight format */
static size_t fmt_bytes(wubu_weight_format_t fmt, size_t n_params) {
    switch (fmt) {
        case WUBU_WT_F32:  return n_params * 4;
        case WUBU_WT_F16:  return n_params * 2;
        case WUBU_WT_Q8_K: return (n_params * 8 + 7) / 8;  /* ~1 byte */
        case WUBU_WT_Q2_K: return (n_params * 2 + 7) / 8;  /* ~0.25 byte */
        case WUBU_WT_Q4_K: return (n_params * 4 + 7) / 8;  /* ~0.5 byte */
        default:           return n_params * 4;
    }
}

/* Precision allocation within a layer (by importance):
 * - First 20%: embeddings → F32
 * - Next 25%: attn QKV → F16
 * - Next 40%: FFN → F16/Q8_K
 * - Next 15%: norms/scales → F32 (small but critical)
 * - Remainder (esoteric): Q4_K/Q2_K */
static wubu_weight_format_t layer_precision(int layer, size_t offset_in_layer,
                                             size_t layer_n_params) {
    /* Trunk layer (0) stays high precision */
    if (layer == 0) {
        size_t emb_end = layer_n_params / 5;  /* 20% embeddings */
        size_t norm_end = emb_end + layer_n_params / 7;  /* ~14% norms */
        if (offset_in_layer < emb_end) return WUBU_WT_F32;
        if (offset_in_layer < norm_end) return WUBU_WT_F32;
        return WUBU_WT_F16;
    }
    /* Deeper layers: more aggressive quantization */
    size_t emb_end = layer_n_params / 5;
    size_t attn_end = emb_end + layer_n_params * 25 / 100;
    size_t ffn_end = attn_end + layer_n_params * 40 / 100;
    size_t norm_end = ffn_end + layer_n_params * 15 / 100;
    if (offset_in_layer < emb_end) return WUBU_WT_F32;
    if (offset_in_layer < attn_end) return WUBU_WT_F16;
    if (offset_in_layer < ffn_end) return WUBU_WT_Q8_K;
    if (offset_in_layer < norm_end) return WUBU_WT_F32;
    return WUBU_WT_Q4_K;  /* esoteric tail */
}

struct wubu_scalable_model {
    char *param_path;
    wubu_scalable_cfg_t cfg;
    wubu_weight_region_t *regions;
    size_t n_regions;
    size_t cap_regions;
    size_t total_bytes;        /* sum of all region bytes */
    size_t active_bytes;       /* sum of active region bytes */
    int   budget_depth;        /* how many layers fit in budget */
};

wubu_scalable_cfg_t wubu_scalable_default_cfg(void) {
    wubu_scalable_cfg_t cfg;
    cfg.trunk_params = 12000000;       /* 12M */
    cfg.branch_factor = 3;            /* 3× per layer */
    cfg.max_layers = 6;              /* 6 layers → 4.37B total */
    cfg.ram_budget = 256 * 1024 * 1024; /* 256MB */
    cfg.kv_cache_bytes = 64 * 1024 * 1024; /* 64MB KV cache */
    cfg.min_precision = WUBU_WT_F16;
    return cfg;
}

wubu_scalable_model_t *wubu_scalable_model_create(const char *param_path,
                                                    const wubu_scalable_cfg_t *cfg) {
    if (!param_path || !cfg) return NULL;
    wubu_scalable_model_t *m = (wubu_scalable_model_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->param_path = strdup(param_path);
    m->cfg = *cfg;
    m->budget_depth = 0;

    if (!m->param_path) { free(m); return NULL; }

    /* Build the region table: for each fractal layer, create weight
     * regions in importance order. Each layer has sub-regions:
     * embeddings, attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up,
     * ffn_down, norms. */
    static const char *subregion_names[] = {
        "embed", "attn_q", "attn_k", "attn_v", "attn_o",
        "ffn_gate", "ffn_up", "ffn_down", "norm_1", "norm_2"
    };
    int n_sub = (int)(sizeof(subregion_names) / sizeof(subregion_names[0]));

    /* Estimate params per sub-region within a layer.
     * Rough split: embed 20%, attn 25%, ffn 50%, norms 5% */
    int emb_frac_num = 20, emb_frac_den = 100;
    int attn_frac_num = 25, attn_frac_den = 100;
    int ffn_frac_num = 50, ffn_frac_den = 100;
    int norm_frac_num = 5, norm_frac_den = 100;

    uint64_t offset = 0;
    size_t avail_budget = cfg->ram_budget - cfg->kv_cache_bytes;
    if (avail_budget == 0) avail_budget = cfg->ram_budget;

    for (int layer = 0; layer < cfg->max_layers; layer++) {
        size_t layer_params = layer_param_count(cfg->trunk_params,
                                                  cfg->branch_factor, layer);

        /* Create sub-regions for this layer */
        size_t emb_params = layer_params * emb_frac_num / emb_frac_den;
        size_t attn_params = layer_params * attn_frac_num / attn_frac_den;
        size_t ffn_params = layer_params * ffn_frac_num / ffn_frac_den;
        size_t norm_params = layer_params - emb_params - attn_params - ffn_params;

        /* Embed region */
        size_t emb_bytes = fmt_bytes(WUBU_WT_F32, emb_params);
        wubu_weight_region_t *r = NULL;
        if (m->n_regions >= m->cap_regions) {
            size_t newcap = m->cap_regions ? m->cap_regions * 2 : 64;
            wubu_weight_region_t *p = (wubu_weight_region_t *)
                realloc(m->regions, newcap * sizeof(*p));
            if (!p) break;
            m->regions = p;
            m->cap_regions = newcap;
        }
        r = &m->regions[m->n_regions];
        snprintf(r->name, sizeof(r->name), "layer.%d.%s", layer, subregion_names[0]);
        r->offset = offset;
        r->n_bytes = emb_bytes;
        r->fmt = WUBU_WT_F32;
        r->layer = layer;
        r->priority = (int)m->n_regions;
        r->active = (offset + emb_bytes <= avail_budget) ? 1 : 0;
        offset += emb_bytes;
        m->n_regions++;
        m->total_bytes += emb_bytes;
        if (r->active) m->active_bytes += emb_bytes;

        /* Attn sub-regions (4: q, k, v, o) */
        size_t attn_each = attn_params / 4;
        for (int j = 1; j <= 4; j++) {
            wubu_weight_format_t fmt = (layer == 0) ? WUBU_WT_F16 : WUBU_WT_F16;
            size_t b = fmt_bytes(fmt, attn_each);
            r = &m->regions[m->n_regions];
            snprintf(r->name, sizeof(r->name), "layer.%d.%s", layer, subregion_names[j]);
            r->offset = offset;
            r->n_bytes = b;
            r->fmt = fmt;
            r->layer = layer;
            r->priority = (int)m->n_regions;
            r->active = (offset + b <= avail_budget) ? 1 : 0;
            offset += b;
            m->n_regions++;
            m->total_bytes += b;
            if (r->active) m->active_bytes += b;
        }

        /* FFN sub-regions (3: gate, up, down) */
        size_t ffn_each = ffn_params / 3;
        wubu_weight_format_t ffn_fmt = (layer == 0) ? WUBU_WT_F16 : WUBU_WT_Q8_K;
        if (layer == 0) ffn_fmt = WUBU_WT_F16;
        for (int j = 5; j <= 7; j++) {
            size_t b = fmt_bytes(ffn_fmt, ffn_each);
            r = &m->regions[m->n_regions];
            snprintf(r->name, sizeof(r->name), "layer.%d.%s", layer, subregion_names[j]);
            r->offset = offset;
            r->n_bytes = b;
            r->fmt = ffn_fmt;
            r->layer = layer;
            r->priority = (int)m->n_regions;
            r->active = (offset + b <= avail_budget) ? 1 : 0;
            offset += b;
            m->n_regions++;
            m->total_bytes += b;
            if (r->active) m->active_bytes += b;
        }

        /* Norm sub-regions (2: norm_1, norm_2) — F32 */
        size_t norm_each = norm_params / 2;
        for (int j = 8; j <= 9; j++) {
            size_t b = fmt_bytes(WUBU_WT_F32, norm_each);
            r = &m->regions[m->n_regions];
            snprintf(r->name, sizeof(r->name), "layer.%d.%s", layer, subregion_names[j]);
            r->offset = offset;
            r->n_bytes = b;
            r->fmt = WUBU_WT_F32;
            r->layer = layer;
            r->priority = (int)m->n_regions;
            r->active = (offset + b <= avail_budget) ? 1 : 0;
            offset += b;
            m->n_regions++;
            m->total_bytes += b;
            if (r->active) m->active_bytes += b;
        }

        /* Track budget depth: this layer is fully loaded */
        int layer_fully_active = 1;
        for (size_t i = 0; i < m->n_regions; i++) {
            if (m->regions[i].layer == layer && !m->regions[i].active) {
                layer_fully_active = 0;
                break;
            }
        }
        if (layer_fully_active) m->budget_depth = layer + 1;
    }

    return m;
}

size_t wubu_scalable_region_count(wubu_scalable_model_t *m) {
    if (!m) return 0;
    return m->n_regions;
}

const wubu_weight_region_t *
wubu_scalable_get_region(wubu_scalable_model_t *m, size_t i) {
    if (!m || i >= m->n_regions) return NULL;
    return &m->regions[i];
}

int wubu_scalable_budget_depth(wubu_scalable_model_t *m) {
    if (!m) return 0;
    return m->budget_depth;
}

int wubu_scalable_get_f32(wubu_scalable_model_t *m,
                           const char *name,
                           float **out_data, size_t *out_n_elems) {
    if (!m || !name || !out_data || !out_n_elems) return -1;
    for (size_t i = 0; i < m->n_regions; i++) {
        if (strcmp(m->regions[i].name, name) == 0) {
            if (!m->regions[i].active) return -1;
            /* For this demo, we don't have a real parameter file.
             * Return the region's precision info as metadata.
             * In production, this would mmap + dequantize. */
            *out_data = NULL;  /* no actual data without a real file */
            *out_n_elems = m->regions[i].n_bytes / 4;  /* upper bound */
            return 0;
        }
    }
    return -1;
}

int wubu_scalable_mark_hot(wubu_scalable_model_t *m, const char *name) {
    if (!m || !name) return -1;
    for (size_t i = 0; i < m->n_regions; i++) {
        if (strcmp(m->regions[i].name, name) == 0) {
            m->regions[i].priority = 0;  /* hottest */
            return 0;
        }
    }
    return -1;
}

int wubu_scalable_prune_cold(wubu_scalable_model_t *m) {
    if (!m) return 0;
    int pruned = 0;
    /* Don't prune the trunk layer (layer 0) */
    for (size_t i = 0; i < m->n_regions; i++) {
        if (!m->regions[i].active) continue;
        if (m->regions[i].layer >= m->budget_depth) {
            /* Beyond budget depth — prune */
            m->regions[i].active = 0;
            m->active_bytes -= m->regions[i].n_bytes;
            pruned++;
        }
    }
    return pruned;
}

void wubu_scalable_memory_stats(wubu_scalable_model_t *m,
                                 size_t *out_active_bytes,
                                 size_t *out_total_bytes,
                                 int *out_active_regions,
                                 int *out_total_regions) {
    if (!m) {
        if (out_active_bytes) *out_active_bytes = 0;
        if (out_total_bytes) *out_total_bytes = 0;
        if (out_active_regions) *out_active_regions = 0;
        if (out_total_regions) *out_total_regions = 0;
        return;
    }
    int active = 0;
    for (size_t i = 0; i < m->n_regions; i++)
        if (m->regions[i].active) active++;
    if (out_active_bytes) *out_active_bytes = m->active_bytes;
    if (out_total_bytes) *out_total_bytes = m->total_bytes;
    if (out_active_regions) *out_active_regions = active;
    if (out_total_regions) *out_total_regions = (int)m->n_regions;
}

void wubu_scalable_model_free(wubu_scalable_model_t *m) {
    if (!m) return;
    free(m->param_path);
    free(m->regions);
    free(m);
}

/* lfm2_load.c -- LFM2.5 loader (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_load.h"
#include "safetensors_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>

/* ---- multi-shard safetensors context (self-scanned) ---- */
static st_ctx *g_shards[8];
static int g_nsh = 0;

int lfm2_open_shards(const char *model_dir) {
    g_nsh = 0;
    DIR *d = opendir(model_dir);
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d)) && g_nsh < 8) {
        const char *nm = e->d_name;
        if (strstr(nm, "model-") && strstr(nm, "-of-") && strstr(nm, ".safetensors")) {
            char fp[2048]; snprintf(fp, sizeof(fp), "%s/%s", model_dir, nm);
            st_ctx *s = st_open(fp);
            if (s) g_shards[g_nsh++] = s;
        }
    }
    closedir(d);
    return g_nsh;
}

/* Load a BF16/F32 tensor by exact name across opened shards -> malloc'd F32. */
static float *load_bf16_f32(const char *name) {
    for (int s = 0; s < g_nsh; s++) {
        const st_tensor_info *t = st_find_tensor(g_shards[s], name);
        if (!t) continue;
        int64_t ne = t->n_elems;
        float *f = (float *)malloc((size_t)ne * sizeof(float));
        if (st_read_tensor_f32(g_shards[s], t, f, ne) != ne) { free(f); continue; }
        return f;
    }
    return NULL;
}

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "lfm2 oom\n"); exit(1); } return p; }

bool lfm2_load(const char *model_dir, lfm2_model_t *m) {
    memset(m, 0, sizeof(*m));
    if (lfm2_open_shards(model_dir) == 0) {
        fprintf(stderr, "lfm2: no safetensors shards in %s\n", model_dir);
        return false;
    }
    fprintf(stderr, "[lfm2] opened %d shard(s)\n", g_nsh);

    /* ---- config.json: dims ---- */
    char cfg[2048]; snprintf(cfg, sizeof(cfg), "%s/config.json", model_dir);
    FILE *f = fopen(cfg, "rb");
    if (!f) { fprintf(stderr, "lfm2: no config.json\n"); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)xmalloc((size_t)sz + 1); size_t _r1 = fread(buf, 1, sz, f); (void)_r1; buf[sz] = 0; fclose(f);

    m->d_model = 0; m->n_layers = 0; m->ff_dim = 0; m->n_q_heads = 0;
    m->n_kv_heads = 0; m->vocab_size = 0; m->conv_dim = 0;
#define GETI(key, field) do { const char *p = strstr(buf, key); if (p) { p = strchr(p, ':'); if (p) m->field = (int)atoll(p + 1); } } while (0)
    GETI("\"hidden_size\"", d_model);
    GETI("\"num_hidden_layers\"", n_layers);
    GETI("\"intermediate_size\"", ff_dim);
    GETI("\"num_attention_heads\"", n_q_heads);
    GETI("\"num_key_value_heads\"", n_kv_heads);
    GETI("\"vocab_size\"", vocab_size);
    GETI("\"conv_dim\"", conv_dim);
#undef GETI
    free(buf);

    m->head_dim = m->d_model / m->n_q_heads;
    /* rope_theta: prefer nested rope_parameters.rope_theta (LFM2.5 = 1e7) */
    m->rope_theta = 10000.0f;
    {
        FILE *fp = fopen(cfg, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END); long z = ftell(fp); fseek(fp, 0, SEEK_SET);
            char *b = (char *)xmalloc((size_t)z + 1); size_t _r2 = fread(b, 1, z, fp); (void)_r2; b[z] = 0; fclose(fp);
            const char *rp = strstr(b, "\"rope_parameters\"");
            const char *th = rp ? strstr(rp, "\"rope_theta\"") : strstr(b, "\"rope_theta\"");
            if (th) { th = strchr(th, ':'); if (th) m->rope_theta = (float)atof(th + 1); }
            free(b);
        }
    }
    if (!m->d_model || !m->n_layers) { fprintf(stderr, "lfm2: bad config\n"); return false; }

    m->is_conv = (bool *)xmalloc(m->n_layers * sizeof(bool));
    m->layers = (lfm2_layer_t *)xmalloc(m->n_layers * sizeof(lfm2_layer_t));
    memset(m->layers, 0, m->n_layers * sizeof(lfm2_layer_t));

    /* layer_types from config.json */
    {
        FILE *fc = fopen(cfg, "rb"); fseek(fc, 0, SEEK_END); long csz = ftell(fc); fseek(fc, 0, SEEK_SET);
        char *cb = (char *)xmalloc((size_t)csz + 1); size_t _r3 = fread(cb, 1, csz, fc); (void)_r3; cb[csz] = 0; fclose(fc);
        const char *lt = strstr(cb, "\"layer_types\"");
        int li = 0;
        if (lt) {
            const char *p = strchr(lt, '[');
            if (p) while (*++p && *p != ']' && li < m->n_layers) {
                if (*p == '"') {
                    if (!strncmp(p + 1, "conv", 4)) m->is_conv[li++] = true;
                    else if (!strncmp(p + 1, "full_attention", 14)) m->is_conv[li++] = false;
                    while (*p && *p != '"') p++;
                }
            }
        }
        free(cb);
        if (li != m->n_layers) fprintf(stderr, "lfm2: warn layer_types count %d != %d\n", li, m->n_layers);
    }

    /* ---- per-layer weights ---- */
    for (int l = 0; l < m->n_layers; l++) {
        lfm2_layer_t *L = &m->layers[l];
        L->conv_k = 3;
        char nm[160];
#define LOAD(var, fmt) do { snprintf(nm, sizeof(nm), fmt, l); \
            L->var = load_bf16_f32(nm); \
            if (!L->var) fprintf(stderr, "lfm2: missing %s\n", nm); } while (0)
        if (m->is_conv[l]) {
            LOAD(in_proj, "model.layers.%d.conv.in_proj.weight");
            LOAD(conv_w,   "model.layers.%d.conv.conv.weight");
            LOAD(out_proj, "model.layers.%d.conv.out_proj.weight");
        } else {
            LOAD(q_proj, "model.layers.%d.self_attn.q_proj.weight");
            LOAD(k_proj, "model.layers.%d.self_attn.k_proj.weight");
            LOAD(v_proj, "model.layers.%d.self_attn.v_proj.weight");
            LOAD(o_proj, "model.layers.%d.self_attn.out_proj.weight");
            LOAD(q_ln,   "model.layers.%d.self_attn.q_layernorm.weight");
            LOAD(k_ln,   "model.layers.%d.self_attn.k_layernorm.weight");
        }
        LOAD(w1,      "model.layers.%d.feed_forward.w1.weight");
        LOAD(w2,      "model.layers.%d.feed_forward.w2.weight");
        LOAD(w3,      "model.layers.%d.feed_forward.w3.weight");
        LOAD(ffn_norm, "model.layers.%d.ffn_norm.weight");
        LOAD(op_norm,  "model.layers.%d.operator_norm.weight");
#undef LOAD
    }

    /* embeddings + norms (tied lm_head) */
    m->embed = load_bf16_f32("model.embed_tokens.weight");
    m->embed_norm = load_bf16_f32("model.embedding_norm.weight");
    m->kv_max_t = 8192;
    size_t kv_bytes = (size_t)m->n_layers * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
    m->kv_cache = (float *)xmalloc(kv_bytes * sizeof(float));
    memset(m->kv_cache, 0, kv_bytes * sizeof(float));

    if (!m->embed || !m->embed_norm) { fprintf(stderr, "lfm2: missing embed/embed_norm\n"); return false; }
    fprintf(stderr, "[lfm2] loaded d=%d layers=%d q=%d kv=%d hd=%d ff=%d vocab=%d conv_dim=%d rope_theta=%.0f\n",
            m->d_model, m->n_layers, m->n_q_heads, m->n_kv_heads, m->head_dim, m->ff_dim, m->vocab_size, m->conv_dim, m->rope_theta);
    return true;
}

void lfm2_free(lfm2_model_t *m) {
    if (m->layers) {
        for (int l = 0; l < m->n_layers; l++) {
            lfm2_layer_t *L = &m->layers[l];
            free(L->in_proj); free(L->conv_w); free(L->out_proj);
            free(L->q_proj); free(L->k_proj); free(L->v_proj); free(L->o_proj);
            free(L->q_ln); free(L->k_ln);
            free(L->w1); free(L->w2); free(L->w3);
            free(L->ffn_norm); free(L->op_norm);
        }
        free(m->layers);
    }
    free(m->is_conv); free(m->embed); free(m->embed_norm); free(m->kv_cache);
    for (int s = 0; s < g_nsh; s++) if (g_shards[s]) st_close(g_shards[s]);
    g_nsh = 0;
    memset(m, 0, sizeof(*m));
}

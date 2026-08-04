/*
 * wubu_tensor_store.c -- the uniform tensor catalog (anti-waste interchange).
 *
 * One catalog over ALL formats (safetensors / GGUF / .st dump). Opening a
 * model file never loads weights -- it builds a name->(offset,dtype,shape)
 * table. Tensors are LIVE-LOADED on demand (one at a time) and EXPORTED
 * streaming (one tensor at a time, bounded RAM). This replaces the
 * load-everything-then-save conversion waste.
 *
 * C11, self-contained; wraps the existing st_ctx (safetensors) and gguf_ctx
 * readers. The .st dump layout is the WuBu-35M trainer's save_checkpoint
 * format: magic(0xBA000001/2) + n_layers + param_count + 137 f32 tensors
 * in the fixed release order.
 */
#include "wubu_tensor_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "safetensors_reader.h"
#include "gguf_reader.h"

/* ---- WuBu-35M .st dump layout constants (the release tensor order) ---- */
#define ST_EMB_ELEMS   (16384LL * 448)
#define ST_FN_ELEMS    448LL
#define ST_Q_ELEMS     (448LL * 448)
#define ST_K_ELEMS     (448LL * 64)
#define ST_V_ELEMS     (448LL * 64)
#define ST_O_ELEMS     (448LL * 448)
#define ST_G_ELEMS     (448LL * 448)
#define ST_QN_ELEMS    64LL
#define ST_KN_ELEMS    64LL
#define ST_AN_ELEMS    448LL
#define ST_GU_ELEMS    (448LL * 2456)
#define ST_DN_ELEMS    (1228LL * 448)
#define ST_FNN_ELEMS   448LL
#define ST_SEL_ELEMS   448LL
#define ST_N_LAYERS    12
#define ST_N_SELECTORS 3

/* the fixed .st write order constants are the ST_* macros above; the
 * trainer's load_checkpoint reads exactly: embedding, final_norm, then
 * 12 layers x 11 tensors, then 3 selectors (137 total). */

struct wubu_tensor_store {
    wubu_ts_fmt fmt;
    char path[512];
    /* backing readers (one is live per format) */
    st_ctx *st;
    gguf_ctx *gg;
    FILE *f;            /* .st dump */
    int64_t st_nl;      /* .st active layer count (v2) */
    /* catalog */
    int n;
    int cap;
    wubu_ts_entry *entries;
    int64_t file_size;
};

/* ---------------------------------------------------------------- sniff */

wubu_ts_fmt wubu_ts_sniff(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return WUBU_TS_UNKNOWN;
    uint8_t b[16];
    size_t got = fread(b, 1, 16, f);
    fclose(f);
    if (got < 8) return WUBU_TS_UNKNOWN;
    if (memcmp(b, "GGUF", 4) == 0) return WUBU_TS_GGUF;
    uint32_t m;
    memcpy(&m, b, 4);
    if (m == 0xBA000001u || m == 0xBA000002u) return WUBU_TS_STDUMP;
    /* safetensors: 8-byte LE header length, then JSON -- a sane header is
     * between 8 bytes and 256 MB */
    uint64_t hlen;
    memcpy(&hlen, b, 8);
    if (hlen > 8 && hlen < (256ULL << 20)) return WUBU_TS_SAFETENSORS;
    return WUBU_TS_UNKNOWN;
}

/* ------------------------------------------------------------- catalog */

static int ts_reserve(wubu_tensor_store_t *ts, int n)
{
    if (n <= ts->cap) return 0;
    int ncap = ts->cap ? ts->cap * 2 : 64;
    while (ncap < n) ncap *= 2;
    wubu_ts_entry *e = (wubu_ts_entry *)realloc(ts->entries,
                                                (size_t)ncap * sizeof(wubu_ts_entry));
    if (!e) return -1;
    ts->entries = e;
    ts->cap = ncap;
    return 0;
}

static wubu_ts_entry *ts_push(wubu_tensor_store_t *ts)
{
    if (ts_reserve(ts, ts->n + 1) != 0) return NULL;
    wubu_ts_entry *e = &ts->entries[ts->n++];
    memset(e, 0, sizeof(*e));
    e->offset = -1;
    return e;
}

static int ts_open_safetensors(wubu_tensor_store_t *ts, const char *path)
{
    ts->st = st_open(path);
    if (!ts->st) return -1;
    int64_t n = st_n_tensors(ts->st);
    for (int64_t i = 0; i < n; i++) {
        const st_tensor_info *info = st_tensor_info_by_index(ts->st, i);
        if (!info) continue;
        wubu_ts_entry *e = ts_push(ts);
        if (!e) return -1;
        snprintf(e->name, sizeof(e->name), "%s", info->name);
        e->n_elems = info->n_elems;
        e->offset = (int64_t)info->data_begin;
        e->ggml_type = (info->dtype == ST_DTYPE_F32) ? 0 : -1;
        e->n_dims = info->n_dims > 4 ? 4 : info->n_dims;
        for (int d = 0; d < e->n_dims; d++) e->dims[d] = info->dims[d];
    }
    return 0;
}

static int ts_open_gguf(wubu_tensor_store_t *ts, const char *path)
{
    ts->gg = gguf_open(path);
    if (!ts->gg) return -1;
    for (int64_t i = 0; i < ts->gg->n_tensors; i++) {
        gguf_tensor_info *ti = &ts->gg->tensors[i];
        wubu_ts_entry *e = ts_push(ts);
        if (!e) return -1;
        snprintf(e->name, sizeof(e->name), "%s", ti->name);
        e->n_elems = 1;
        for (int d = 0; d < ti->n_dims; d++) e->n_elems *= ti->dims[d];
        e->offset = -1; /* addressed via gguf reader */
        e->ggml_type = ti->ggml_type;
        e->n_dims = ti->n_dims > 4 ? 4 : ti->n_dims;
        for (int d = 0; d < e->n_dims; d++) e->dims[d] = ti->dims[d];
    }
    return 0;
}

/* build the fixed 137-entry .st catalog + byte offsets */
static int ts_open_stdump(wubu_tensor_store_t *ts, const char *path)
{
    ts->f = fopen(path, "rb");
    if (!ts->f) return -1;
    uint32_t magic = 0;
    int nl = 0;
    if (fread(&magic, 4, 1, ts->f) != 1 ||
        (magic != 0xBA000001u && magic != 0xBA000002u)) { return -1; }
    if (magic == 0xBA000002u) {
        if (fread(&nl, 4, 1, ts->f) != 1) return -1;
        if (nl < 1 || nl > ST_N_LAYERS) return -1;
    }
    long n = 0;
    if (fread(&n, sizeof(long), 1, ts->f) != 1) return -1;
    ts->st_nl = (magic == 0xBA000002u) ? nl : ST_N_LAYERS;

    int64_t off = (int64_t)ftell(ts->f);
    char name[192];

    snprintf(name, sizeof(name), "embedding.weight");
    { wubu_ts_entry *e = ts_push(ts); if (!e) return -1;
      snprintf(e->name, sizeof(e->name), "%s", name);
      e->n_elems = ST_EMB_ELEMS; e->offset = off; e->ggml_type = 0;
      e->n_dims = 2; e->dims[0] = 16384; e->dims[1] = 448;
      off += ST_EMB_ELEMS * 4; }

    snprintf(name, sizeof(name), "final_norm.weight");
    { wubu_ts_entry *e = ts_push(ts); if (!e) return -1;
      snprintf(e->name, sizeof(e->name), "%s", name);
      e->n_elems = ST_FN_ELEMS; e->offset = off; e->ggml_type = 0;
      e->n_dims = 1; e->dims[0] = 448;
      off += ST_FN_ELEMS * 4; }

    static const char *L[11] = {
        "layers.%d.attn.q_proj.weight", "layers.%d.attn.k_proj.weight",
        "layers.%d.attn.v_proj.weight", "layers.%d.attn.o_proj.weight",
        "layers.%d.attn.g_proj.weight", "layers.%d.attn.q_norm.weight",
        "layers.%d.attn.k_norm.weight", "layers.%d.attn_norm.weight",
        "layers.%d.ffn.gate_up.weight", "layers.%d.ffn.down.weight",
        "layers.%d.ffn_norm.weight" };
    static const int64_t LE[11] = { ST_Q_ELEMS, ST_K_ELEMS, ST_V_ELEMS,
        ST_O_ELEMS, ST_G_ELEMS, ST_QN_ELEMS, ST_KN_ELEMS, ST_AN_ELEMS,
        ST_GU_ELEMS, ST_DN_ELEMS, ST_FNN_ELEMS };
    for (int layer = 0; layer < ST_N_LAYERS; layer++) {
        for (int t = 0; t < 11; t++) {
            wubu_ts_entry *e = ts_push(ts); if (!e) return -1;
            snprintf(e->name, sizeof(e->name), L[t], layer);
            e->n_elems = LE[t]; e->offset = off; e->ggml_type = 0;
            switch (t) {
                case 1: case 2:              /* k_proj, v_proj: [448,64] */
                    e->n_dims = 2; e->dims[0] = 448; e->dims[1] = 64; break;
                case 5: case 6:              /* q_norm, k_norm: [64] */
                    e->n_dims = 1; e->dims[0] = 64; break;
                case 7: case 10:             /* attn_norm, ffn_norm: [448] */
                    e->n_dims = 1; e->dims[0] = 448; break;
                case 8:                      /* gate_up: [448,2456] */
                    e->n_dims = 2; e->dims[0] = 448; e->dims[1] = 2456; break;
                case 9:                      /* down: [1228,448] */
                    e->n_dims = 2; e->dims[0] = 1228; e->dims[1] = 448; break;
                default:                     /* q/k/v/o/g_proj: [448,448] */
                    e->n_dims = 2; e->dims[0] = 448; e->dims[1] = 448; break;
            }
            off += LE[t] * 4;
        }
    }
    for (int s = 0; s < ST_N_SELECTORS; s++) {
        wubu_ts_entry *e = ts_push(ts); if (!e) return -1;
        snprintf(e->name, sizeof(e->name), "selectors.%d.score.weight", s);
        e->n_elems = ST_SEL_ELEMS; e->offset = off; e->ggml_type = 0;
        e->n_dims = 1; e->dims[0] = 448;
        off += ST_SEL_ELEMS * 4;
    }
    ts->file_size = off;
    return 0;
}

wubu_tensor_store_t *wubu_ts_open(const char *path)
{
    wubu_ts_fmt fmt = wubu_ts_sniff(path);
    if (fmt == WUBU_TS_UNKNOWN) return NULL;
    wubu_tensor_store_t *ts = (wubu_tensor_store_t *)calloc(1, sizeof(*ts));
    if (!ts) return NULL;
    ts->fmt = fmt;
    snprintf(ts->path, sizeof(ts->path), "%s", path);
    int rc = -1;
    if (fmt == WUBU_TS_SAFETENSORS) rc = ts_open_safetensors(ts, path);
    else if (fmt == WUBU_TS_GGUF)    rc = ts_open_gguf(ts, path);
    else if (fmt == WUBU_TS_STDUMP)  rc = ts_open_stdump(ts, path);
    if (rc != 0) { wubu_ts_close(ts); return NULL; }
    return ts;
}

wubu_ts_fmt wubu_ts_format(const wubu_tensor_store_t *ts) { return ts ? ts->fmt : WUBU_TS_UNKNOWN; }
int wubu_ts_count(const wubu_tensor_store_t *ts) { return ts ? ts->n : 0; }
const wubu_ts_entry *wubu_ts_entry_at(const wubu_tensor_store_t *ts, int i)
{ return (ts && i >= 0 && i < ts->n) ? &ts->entries[i] : NULL; }

const wubu_ts_entry *wubu_ts_find(const wubu_tensor_store_t *ts, const char *name)
{
    if (!ts || !name) return NULL;
    for (int i = 0; i < ts->n; i++)
        if (strcmp(ts->entries[i].name, name) == 0) return &ts->entries[i];
    return NULL;
}

/* --------------------------------------------------------- live load */

int wubu_ts_get_f32(const wubu_tensor_store_t *ts, const char *name,
                    float *out, int64_t max_elems)
{
    if (!ts || !name || !out) return -1;
    const wubu_ts_entry *e = wubu_ts_find(ts, name);
    if (!e) return -1;
    if (max_elems < e->n_elems) return -1;

    if (ts->fmt == WUBU_TS_SAFETENSORS) {
        const st_tensor_info *info = st_find_tensor(ts->st, name);
        if (!info) return -1;
        return st_read_tensor_f32(ts->st, info, out, e->n_elems) ? 0 : -1;
    }
    if (ts->fmt == WUBU_TS_GGUF) {
        gguf_tensor_info *ti = gguf_find_tensor(ts->gg, name);
        if (!ti) return -1;
        int64_t raw = gguf_raw_size(ti->ggml_type, e->n_elems);
        if (raw <= 0) return -1;
        if (fseek(ts->gg->file, (long)(ts->gg->data_blob_offset + ti->data_offset),
                  SEEK_SET) != 0) return -1;
        uint8_t *buf = (uint8_t *)malloc((size_t)raw);
        if (!buf) return -1;
        size_t got = fread(buf, 1, (size_t)raw, ts->gg->file);
        if (got != (size_t)raw) { free(buf); return -1; }
        gguf_dequantize(buf, ti->ggml_type, e->n_elems, out);
        free(buf);
        return 0;
    }
    if (ts->fmt == WUBU_TS_STDUMP) {
        if (fseek(ts->f, (long)e->offset, SEEK_SET) != 0) return -1;
        size_t got = fread(out, 4, (size_t)e->n_elems, ts->f);
        return (int64_t)got == e->n_elems ? 0 : -1;
    }
    return -1;
}

/* ------------------------------------------------------------ export */

static int st_fixed_name(char *buf, size_t cap, int idx)
{
    /* the fixed .st order: 0 = embedding, 1 = final_norm,
     * then 12 layers x 11, then 3 selectors */
    if (idx == 0) { snprintf(buf, cap, "embedding.weight"); return 0; }
    if (idx == 1) { snprintf(buf, cap, "final_norm.weight"); return 0; }
    int rest = idx - 2;
    int n_sel = 12 * 11;
    if (rest < n_sel) {
        int layer = rest / 11, t = rest % 11;
        static const char *L[11] = {
            "layers.%d.attn.q_proj.weight", "layers.%d.attn.k_proj.weight",
            "layers.%d.attn.v_proj.weight", "layers.%d.attn.o_proj.weight",
            "layers.%d.attn.g_proj.weight", "layers.%d.attn.q_norm.weight",
            "layers.%d.attn.k_norm.weight", "layers.%d.attn_norm.weight",
            "layers.%d.ffn.gate_up.weight", "layers.%d.ffn.down.weight",
            "layers.%d.ffn_norm.weight" };
        snprintf(buf, cap, L[t], layer);
        return 0;
    }
    int s = rest - n_sel;
    if (s < ST_N_SELECTORS) { snprintf(buf, cap, "selectors.%d.score.weight", s); return 0; }
    return -1;
}

static int64_t st_fixed_elems(int idx)
{
    if (idx == 0) return ST_EMB_ELEMS;
    if (idx == 1) return ST_FN_ELEMS;
    int rest = idx - 2;
    if (rest < 12 * 11) {
        static const int64_t LE[11] = { ST_Q_ELEMS, ST_K_ELEMS, ST_V_ELEMS,
            ST_O_ELEMS, ST_G_ELEMS, ST_QN_ELEMS, ST_KN_ELEMS, ST_AN_ELEMS,
            ST_GU_ELEMS, ST_DN_ELEMS, ST_FNN_ELEMS };
        return LE[rest % 11];
    }
    int s = rest - 12 * 11;
    return (s >= 0 && s < ST_N_SELECTORS) ? ST_SEL_ELEMS : -1;
}

static int ts_export_stdump(const wubu_tensor_store_t *ts, const char *out)
{
    FILE *w = fopen(out, "wb");
    if (!w) return -1;
    uint32_t magic = 0xBA000002u;
    int nl = (int)((ts->fmt == WUBU_TS_STDUMP && ts->st_nl) ? ts->st_nl : ST_N_LAYERS);
    fwrite(&magic, 4, 1, w);
    fwrite(&nl, 4, 1, w);
    long n = 0;
    for (int i = 0; i < ts->n; i++) n += (long)ts->entries[i].n_elems;
    fwrite(&n, sizeof(long), 1, w);
    /* fixed .st order (load_checkpoint compatibility) */
    for (int idx = 0; idx < 2 + 12 * 11 + ST_N_SELECTORS; idx++) {
        char name[192];
        if (st_fixed_name(name, sizeof(name), idx) != 0) { fclose(w); return -1; }
        int64_t elems = st_fixed_elems(idx);
        float *buf = (float *)malloc((size_t)elems * sizeof(float));
        if (!buf) { fclose(w); return -1; }
        if (wubu_ts_get_f32(ts, name, buf, elems) != 0) { free(buf); fclose(w); return -1; }
        fwrite(buf, sizeof(float), (size_t)elems, w);
        free(buf);
    }
    fclose(w);
    return 0;
}

static int ts_export_safetensors(const wubu_tensor_store_t *ts, const char *out)
{
    /* build the JSON header: {"name":{"dtype":"F32","shape":[...],"data_offsets":[b,e]},...} */
    size_t hcap = 4096 + (size_t)ts->n * 256;
    char *json = (char *)malloc(hcap);
    if (!json) return -1;
    size_t p = 0;
    p += (size_t)snprintf(json + p, hcap - p, "{");
    int64_t blob = 0;
    for (int i = 0; i < ts->n; i++) {
        const wubu_ts_entry *e = &ts->entries[i];
        char shape[256]; shape[0] = 0;
        int nd = e->n_dims ? e->n_dims : 1;
        size_t sp = 0;
        for (int d = 0; d < nd; d++) {
            int64_t dim = e->dims[d] ? e->dims[d] : e->n_elems;
            sp += (size_t)snprintf(shape + sp, sizeof(shape) - sp, "%s%lld",
                                   d ? "," : "", (long long)dim);
        }
        p += (size_t)snprintf(json + p, hcap - p,
              "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%s],\"data_offsets\":[%lld,%lld]}",
              i ? "," : "", e->name, shape, (long long)blob,
              (long long)(blob + e->n_elems * 4));
        blob += e->n_elems * 4;
    }
    p += (size_t)snprintf(json + p, hcap - p, "}");
    /* header length padded so (8 + len) % 8 == 0 */
    size_t jlen = p;
    size_t pad = (8 + jlen) % 8 ? 8 - (8 + jlen) % 8 : 0;
    while (p < jlen + pad && p + 1 < hcap) json[p++] = ' ';

    FILE *w = fopen(out, "wb");
    if (!w) { free(json); return -1; }
    uint64_t hlen = jlen + pad;
    fwrite(&hlen, 8, 1, w);
    fwrite(json, 1, jlen + pad, w);
    free(json);
    for (int i = 0; i < ts->n; i++) {
        const wubu_ts_entry *e = &ts->entries[i];
        float *buf = (float *)malloc((size_t)e->n_elems * sizeof(float));
        if (!buf) { fclose(w); return -1; }
        if (wubu_ts_get_f32(ts, e->name, buf, e->n_elems) != 0) {
            free(buf); fclose(w); return -1;
        }
        fwrite(buf, sizeof(float), (size_t)e->n_elems, w);
        free(buf);
    }
    fclose(w);
    return 0;
}

/* minimal GGUF v3 writer: all tensors as F32, one KV (general.name). */
static int ts_export_gguf(const wubu_tensor_store_t *ts, const char *out)
{
    FILE *w = fopen(out, "wb");
    if (!w) return -1;
    fwrite("GGUF", 4, 1, w);
    uint32_t ver = 3;
    fwrite(&ver, 4, 1, w);
    uint64_t n_t = (uint64_t)ts->n, n_kv = 1;
    fwrite(&n_t, 8, 1, w);
    fwrite(&n_kv, 8, 1, w);
    /* kv: general.name = the source path base */
    const char *name = strrchr(ts->path, '/');
    name = name ? name + 1 : ts->path;
    uint64_t slen = strlen(name);
    uint64_t klen = strlen("general.name");
    fwrite(&klen, 8, 1, w); fwrite("general.name", 1, klen, w);
    uint32_t vtype = 8; /* GGUF_TYPE_STRING */
    fwrite(&vtype, 4, 1, w);
    fwrite(&slen, 8, 1, w); fwrite(name, 1, slen, w);
    /* NOTE: GGUF does NOT store an alignment field here (the spec keeps
     * it implicit -- llama.cpp defaults to 32 and derives the blob start
     * by aligning the end of the tensor table). Writing one shifts the
     * tensor table and misaligns every entry. The reader pads itself. */
    const uint32_t alignment = 32;
    /* tensor table: name, n_dims, dims[], type, offset (computed) */
    int64_t data_off = 0;
    for (int i = 0; i < ts->n; i++) {
        const wubu_ts_entry *e = &ts->entries[i];
        uint64_t tlen = strlen(e->name);
        fwrite(&tlen, 8, 1, w); fwrite(e->name, 1, tlen, w);
        uint32_t nd = e->n_dims ? e->n_dims : 1;
        fwrite(&nd, 4, 1, w);
        for (int d = 0; d < (int)nd; d++) {
            uint64_t dim = e->dims[d] ? e->dims[d] : e->n_elems;
            fwrite(&dim, 8, 1, w);
        }
        uint32_t gtype = 0; /* F32 */
        fwrite(&gtype, 4, 1, w);
        uint64_t off = (uint64_t)data_off;
        fwrite(&off, 8, 1, w);
        int64_t bytes = e->n_elems * 4;
        data_off += bytes;
        if (data_off % alignment) data_off += alignment - data_off % alignment;
    }
    /* pad to alignment, then the data blob */
    long pos = ftell(w);
    long pad_to = ((pos + alignment - 1) / alignment) * alignment;
    while (ftell(w) < pad_to) fputc(0, w);
    for (int i = 0; i < ts->n; i++) {
        const wubu_ts_entry *e = &ts->entries[i];
        float *buf = (float *)malloc((size_t)e->n_elems * sizeof(float));
        if (!buf) { fclose(w); return -1; }
        if (wubu_ts_get_f32(ts, e->name, buf, e->n_elems) != 0) {
            free(buf); fclose(w); return -1;
        }
        fwrite(buf, sizeof(float), (size_t)e->n_elems, w);
        free(buf);
        long rem = (long)(e->n_elems * 4) % alignment;
        if (rem) { long need = alignment - rem; while (need--) fputc(0, w); }
    }
    fclose(w);
    return 0;
}

int wubu_ts_export(const wubu_tensor_store_t *ts, wubu_ts_fmt target,
                   const char *out_path)
{
    if (!ts || !out_path) return -1;
    if (target == WUBU_TS_STDUMP)      return ts_export_stdump(ts, out_path);
    if (target == WUBU_TS_SAFETENSORS) return ts_export_safetensors(ts, out_path);
    if (target == WUBU_TS_GGUF)        return ts_export_gguf(ts, out_path);
    return -1;
}

void wubu_ts_close(wubu_tensor_store_t *ts)
{
    if (!ts) return;
    if (ts->st) st_close(ts->st);
    if (ts->gg) gguf_close(ts->gg);
    if (ts->f) fclose(ts->f);
    free(ts->entries);
    free(ts);
}

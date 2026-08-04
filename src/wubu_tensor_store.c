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

/* IEEE-754 f32 -> fp16 (round-to-nearest-even not required; the Q8_0
 * scale is only 8-bit precision, so truncation-level is fine). */
static uint16_t f32_to_f16(float x)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xffu);
    uint32_t mant = u & 0x7fffffu;
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00u);      /* inf/nan */
    int32_t e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);          /* overflow -> inf */
    if (e <= 0) {                                             /* subnormal half */
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = 14u - (uint32_t)e;
        return (uint16_t)(sign | (mant >> shift));
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | (mant >> 13));
}

static void iq2xxs_encode_block(const float *v, uint8_t *blk);

/* minimal GGUF v3 writer: all tensors as F32, one KV (general.name). */
static int ts_export_gguf_typed(const wubu_tensor_store_t *ts,
                                const char *out, int ggml_type);
static int ts_export_gguf(const wubu_tensor_store_t *ts, const char *out)
{
    return ts_export_gguf_typed(ts, out, 0 /* F32 */);
}

/* Q8_0 GGUF export: the storage-reduction path. Each 32-element block is
 * [d: f32][qs: 32 x int8] with d = amax/127 (36 bytes vs 128 -> 3.55x).
 * Block-major over the flat tensor (ceil(n/32) blocks; the tail block is
 * zero-padded). gguf_open + gguf_dequantize read it back (type 7). */
static int ts_export_gguf_typed(const wubu_tensor_store_t *ts,
                                const char *out, int ggml_type)
{
    FILE *w = fopen(out, "wb");
    if (!w) return -1;
    const int q8 = (ggml_type == GGML_TYPE_Q8_0);
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
        uint32_t gtype = (uint32_t)ggml_type;
        fwrite(&gtype, 4, 1, w);
        uint64_t off = (uint64_t)data_off;
        fwrite(&off, 8, 1, w);
        int64_t bytes = q8 ? ((e->n_elems + 31) / 32) * 34
                           : e->n_elems * 4;
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
        if (q8) {
            /* Q8_0: per-32-block [d: fp16][32 x int8] = 34 bytes, d = amax/127
             * (the llama.cpp block_q8_0 layout -- the reader's gguf_raw_size
             * returns 34/block, so we MUST match it, not 36). */
            int64_t nb = (e->n_elems + 31) / 32;
            uint16_t *dbuf = (uint16_t *)malloc((size_t)nb * sizeof(uint16_t));
            int8_t *qbuf = (int8_t *)malloc((size_t)nb * 32);
            if (!dbuf || !qbuf) { free(buf); free(dbuf); free(qbuf); fclose(w); return -1; }
            for (int64_t b = 0; b < nb; b++) {
                float amax = 0.0f;
                for (int j = 0; j < 32; j++) {
                    int64_t idx = b * 32 + j;
                    float v = (idx < e->n_elems) ? buf[idx] : 0.0f;
                    float a = fabsf(v);
                    if (a > amax) amax = a;
                }
                float d = (amax > 0.0f) ? amax / 127.0f : 0.0f;
                dbuf[b] = f32_to_f16(d);
                for (int j = 0; j < 32; j++) {
                    int64_t idx = b * 32 + j;
                    float v = (idx < e->n_elems) ? buf[idx] : 0.0f;
                    qbuf[b * 32 + j] = (d > 0.0f)
                        ? (int8_t)(v / d < 0 ? -1 - (int)(-v / d) : (int)(v / d + 0.5f))
                        : 0;
                }
            }
            /* INTERLEAVED: [d16][32 x int8] per block (34 B) */
            {
                uint8_t *blk = (uint8_t *)malloc((size_t)nb * 34);
                if (!blk) { free(buf); free(dbuf); free(qbuf); fclose(w); return -1; }
                for (int64_t b = 0; b < nb; b++) {
                    memcpy(blk + b * 34, &dbuf[b], 2);
                    memcpy(blk + b * 34 + 2, qbuf + b * 32, 32);
                }
                fwrite(blk, 1, (size_t)nb * 34, w);
                free(blk);
            }
            free(dbuf); free(qbuf);
        } else {
            fwrite(buf, sizeof(float), (size_t)e->n_elems, w);
        }
        free(buf);
        long rem = (long)((q8 ? ((e->n_elems + 31) / 32) * 34 : e->n_elems * 4)) % alignment;
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

/* Q8_0-quantized GGUF export: the storage-reduction path (~3.55x smaller
 * than F32). The output is a valid GGUF v3 (tensors typed Q8_0) readable
 * by gguf_open/gguf_dequantize and the engine's GGUF loader. */
int wubu_ts_export_q8(const wubu_tensor_store_t *ts, const char *out_path)
{
    if (!ts || !out_path) return -1;
    return ts_export_gguf_typed(ts, out_path, GGML_TYPE_Q8_0);
}

/* ------------------------------------------------------ MIXED EXPORT
 * The Unsloth/quality-density doctrine (research/057): compression is a
 * LADDER over roles, never a uniform bit-width. Keep maximum elements
 * where signal lives (embeddings, attention, head), minimize where
 * saturation eats the bits (expert weights), exact for norms/routers.
 * Quant encoders owned here: F32, Q8_0 (fp16 d + int8, 34 B/32 el),
 * Q4_0 (fp16 d + 4-bit nibbles, 18 B/32 el). IQ2_XXS/IQ3_XXS/IQ4_NL
 * encoders are the next wave (the grids + dequants are in-tree). */

typedef enum {
    ROLE_EMBED, ROLE_HEAD, ROLE_ATTN, ROLE_EXPERT_GU, ROLE_EXPERT_DOWN,
    ROLE_SHARED, ROLE_EXACT
} wubu_ts_role;

static wubu_ts_role role_of(const wubu_ts_entry *e)
{
    const char *n = e->name;
    /* exact-first: tiny tensors + norms + routers stay F32 */
    if (e->n_elems < 4096) return ROLE_EXACT;
    if (strstr(n, "norm") || strstr(n, "gate_inp") || strstr(n, "router"))
        return ROLE_EXACT;
    if (strstr(n, "embed") || strstr(n, "embd")) return ROLE_EMBED;
    if (strstr(n, "head") || strstr(n, "output")) return ROLE_HEAD;
    if (strstr(n, "attn") || strstr(n, "qkv")) return ROLE_ATTN;
    if (strstr(n, "shexp")) return ROLE_SHARED;
    if (strstr(n, "down")) return ROLE_EXPERT_DOWN;
    if (strstr(n, "gate") || strstr(n, "up")) return ROLE_EXPERT_GU;
    return ROLE_ATTN; /* default: keep max */
}

/* per-role quant: the Unsloth ladder shape (Q8_0/Q4_0 are the encoders
 * we own; IQ2_XXS/IQ3_XXS slot into GU/SHARED when the encoders land) */
static int quant_for_role(wubu_ts_role r)
{
    switch (r) {
        case ROLE_EXACT:      return 0;  /* F32 */
        case ROLE_EMBED:      return GGML_TYPE_Q8_0;
        case ROLE_HEAD:       return GGML_TYPE_Q8_0;
        case ROLE_ATTN:       return GGML_TYPE_Q8_0;
        case ROLE_EXPERT_GU:  return GGML_TYPE_IQ2_XXS;
        case ROLE_EXPERT_DOWN:return GGML_TYPE_Q4_0;
        case ROLE_SHARED:     return GGML_TYPE_IQ2_XXS;
    }
    return 0;
}

static int64_t q_bytes(int type, int64_t n_elems)
{
    if (type == 0) return n_elems * 4;
    if (type == GGML_TYPE_Q8_0) return ((n_elems + 31) / 32) * 34;
    if (type == GGML_TYPE_Q4_0) return ((n_elems + 31) / 32) * 18;
    if (type == GGML_TYPE_IQ2_XXS) return ((n_elems + 255) / 256) * 66;
    return -1;
}

static int write_quant_block(FILE *w, int type, const float *buf, int64_t n_elems)
{
    int64_t nb = (n_elems + 31) / 32;
    if (type == 0) {
        return fwrite(buf, sizeof(float), (size_t)n_elems, w) == (size_t)n_elems ? 0 : -1;
    }
    if (type == GGML_TYPE_Q8_0) {
        uint16_t *d16 = (uint16_t *)malloc((size_t)nb * 2);
        int8_t *qs = (int8_t *)malloc((size_t)nb * 32);
        uint8_t *blk = (uint8_t *)malloc((size_t)nb * 34);
        if (!d16 || !qs || !blk) { free(d16); free(qs); free(blk); return -1; }
        for (int64_t b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = (b * 32 + j < n_elems) ? buf[b * 32 + j] : 0.0f;
                float a = fabsf(v); if (a > amax) amax = a;
            }
            float d = (amax > 0.0f) ? amax / 127.0f : 0.0f;
            d16[b] = f32_to_f16(d);
            for (int j = 0; j < 32; j++) {
                float v = (b * 32 + j < n_elems) ? buf[b * 32 + j] : 0.0f;
                qs[b * 32 + j] = (d > 0.0f)
                    ? (int8_t)(v / d < 0 ? -1 - (int)(-v / d) : (int)(v / d + 0.5f)) : 0;
            }
        }
        /* INTERLEAVED blocks: [d16][32 x int8] per block -- writing all
         * d16s then all qs as two chunks drifts every block after 0. */
        for (int64_t b = 0; b < nb; b++) {
            memcpy(blk + b * 34, &d16[b], 2);
            memcpy(blk + b * 34 + 2, qs + b * 32, 32);
        }
        int rc = (fwrite(blk, 1, (size_t)nb * 34, w) == (size_t)nb * 34) ? 0 : -1;
        free(d16); free(qs); free(blk);
        return rc;
    }
    if (type == GGML_TYPE_IQ2_XXS) {
        int64_t nb = (n_elems + 255) / 256;
        uint8_t *blk = (uint8_t *)malloc((size_t)nb * 66);
        if (!blk) return -1;
        for (int64_t b = 0; b < nb; b++) {
            float tmp[256];
            for (int j = 0; j < 256; j++)
                tmp[j] = (b * 256 + j < n_elems) ? buf[b * 256 + j] : 0.0f;
            iq2xxs_encode_block(tmp, blk + b * 66);
        }
        int rc = (fwrite(blk, 1, (size_t)nb * 66, w) == (size_t)nb * 66) ? 0 : -1;
        free(blk);
        return rc;
    }
    if (type == GGML_TYPE_Q4_0) {
        uint16_t *d16 = (uint16_t *)malloc((size_t)nb * 2);
        uint8_t *q4 = (uint8_t *)malloc((size_t)nb * 16);
        uint8_t *blk = (uint8_t *)malloc((size_t)nb * 18);
        if (!d16 || !q4 || !blk) { free(d16); free(q4); free(blk); return -1; }
        for (int64_t b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = (b * 32 + j < n_elems) ? buf[b * 32 + j] : 0.0f;
                float a = fabsf(v); if (a > amax) amax = a;
            }
            float d = (amax > 0.0f) ? amax / 8.0f : 0.0f;
            d16[b] = f32_to_f16(d);
            for (int j = 0; j < 32; j += 2) {
                float v0 = (b * 32 + j     < n_elems) ? buf[b * 32 + j] : 0.0f;
                float v1 = (b * 32 + j + 1 < n_elems) ? buf[b * 32 + j + 1] : 0.0f;
                int q0 = (d > 0.0f) ? (int)(v0 / d + (v0 >= 0 ? 0.5f : -0.5f)) : 0;
                int q1 = (d > 0.0f) ? (int)(v1 / d + (v1 >= 0 ? 0.5f : -0.5f)) : 0;
                if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
                /* reader: even j -> HIGH nibble, odd j -> LOW nibble,
                 * value = (nibble) - 8  => store nibble = q + 8 */
                q4[b * 16 + j / 2] = (uint8_t)(((q0 + 8) << 4) | ((q1 + 8) & 0xF));
            }
        }
        /* INTERLEAVED: [d16][16 nibbles] per block (18 bytes) */
        for (int64_t b = 0; b < nb; b++) {
            memcpy(blk + b * 18, &d16[b], 2);
            memcpy(blk + b * 18 + 2, q4 + b * 16, 16);
        }
        int rc = (fwrite(blk, 1, (size_t)nb * 18, w) == (size_t)nb * 18) ? 0 : -1;
        free(d16); free(q4); free(blk);
        return rc;
    }
    return -1;
}

int wubu_ts_export_mixed(const wubu_tensor_store_t *ts, const char *out_path)
{
    if (!ts || !out_path) return -1;
    const uint32_t alignment = 32;
    int n = ts->n;
    int *types = (int *)malloc((size_t)n * sizeof(int));
    int64_t *bytes = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!types || !bytes) { free(types); free(bytes); return -1; }
    int64_t data_off = 0;
    for (int i = 0; i < n; i++) {
        types[i] = quant_for_role(role_of(&ts->entries[i]));
        bytes[i] = q_bytes(types[i], ts->entries[i].n_elems);
        if (bytes[i] < 0) { free(types); free(bytes); return -1; }
        data_off += bytes[i];
        if (data_off % alignment) data_off += alignment - data_off % alignment;
    }
    FILE *w = fopen(out_path, "wb");
    if (!w) { free(types); free(bytes); return -1; }
    fwrite("GGUF", 4, 1, w);
    uint32_t ver = 3; fwrite(&ver, 4, 1, w);
    uint64_t n_t = (uint64_t)n, n_kv = 1;
    fwrite(&n_t, 8, 1, w); fwrite(&n_kv, 8, 1, w);
    const char *name = strrchr(ts->path, '/');
    name = name ? name + 1 : ts->path;
    uint64_t klen = strlen("general.name"), slen = strlen(name);
    fwrite(&klen, 8, 1, w); fwrite("general.name", 1, klen, w);
    uint32_t vtype = 8; fwrite(&vtype, 4, 1, w);
    fwrite(&slen, 8, 1, w); fwrite(name, 1, slen, w);
    data_off = 0;
    for (int i = 0; i < n; i++) {
        const wubu_ts_entry *e = &ts->entries[i];
        uint64_t tlen = strlen(e->name);
        fwrite(&tlen, 8, 1, w); fwrite(e->name, 1, tlen, w);
        uint32_t nd = e->n_dims ? e->n_dims : 1;
        fwrite(&nd, 4, 1, w);
        for (int d = 0; d < (int)nd; d++) {
            uint64_t dim = e->dims[d] ? e->dims[d] : e->n_elems;
            fwrite(&dim, 8, 1, w);
        }
        uint32_t gtype = (uint32_t)types[i];
        fwrite(&gtype, 4, 1, w);
        fwrite(&data_off, 8, 1, w);
        data_off += bytes[i];
        if (data_off % alignment) data_off += alignment - data_off % alignment;
    }
    long pos = ftell(w);
    long pad_to = ((pos + alignment - 1) / alignment) * alignment;
    while (ftell(w) < pad_to) fputc(0, w);
    for (int i = 0; i < n; i++) {
        const wubu_ts_entry *e = &ts->entries[i];
        float *buf = (float *)malloc((size_t)e->n_elems * sizeof(float));
        if (!buf) { fclose(w); free(types); free(bytes); return -1; }
        if (wubu_ts_get_f32(ts, e->name, buf, e->n_elems) != 0) {
            free(buf); fclose(w); free(types); free(bytes); return -1;
        }
        if (write_quant_block(w, types[i], buf, e->n_elems) != 0) {
            free(buf); fclose(w); free(types); free(bytes); return -1;
        }
        free(buf);
        long rem = (long)(bytes[i] % alignment);
        if (rem) { long need = alignment - rem; while (need--) fputc(0, w); }
    }
    fclose(w);
    free(types); free(bytes);
    return 0;
}

/* iq2xxs_grid[256] -- the IQ2_XXS 8-dim codebook (copied from
 * src/dequant_iq2_xxs.c; each uint64 packs 8 small-int magnitudes). */
static const uint64_t IQ2XXS_GRID[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};



/* ------------------------------------------------------ IQ2_XXS (2.06 bpw)
 * The 2-bit slot of the mixed ladder (research/057-058). Block layout
 * (66 bytes -> 256 floats, matching dequant_iq2_xxs.c):
 *   [0:2]  d  fp16  (the base scale)
 *   [2:66] qs 64 bytes = 8 sub-blocks of 32 values:
 *          per 32: aux32[0] = 4 x 1-byte grid indices (l=0..3, 8 values each)
 *                  aux32[1] = (scale_factor<<28) | sign3<<21 | sign2<<14 |
 *                             sign1<<7 | sign0   (7 sign bits per 8-group)
 *   value = +- d * (0.5+sf)*0.25 * grid[g][j]
 * ENCODE (scale-first + sign-folded): d from block amax; per 8-group the
 * optimal sign is sign(v) (grid magnitudes are non-negative), leaving a
 * pure magnitude search over the 256-entry codebook; per 32 sub-block a
 * 4-bit scale sweep. Deterministic, self-contained. */
static void iq2xxs_encode_block(const float *v, uint8_t *blk)
{
    /* d = amax / (max grid 43 * max scale 3.875) */
    float amax = 0.0f;
    for (int i = 0; i < 256; i++) { float a = fabsf(v[i]); if (a > amax) amax = a; }
    float d = (amax > 0.0f) ? amax / (43.0f * 3.875f) : 0.0f;
    uint16_t d16 = f32_to_f16(d);
    memcpy(blk, &d16, 2);
    /* per sub-block: choose sf in 0..15 minimizing reconstruction error */
    for (int ib = 0; ib < 8; ib++) {
        const float *s = v + ib * 32;
        float mags[32];
        for (int j = 0; j < 32; j++) mags[j] = fabsf(s[j]);
        int best_sf = 0; float best_err = 1e30f;
        uint8_t best_idx[4]; uint32_t best_signs = 0;
        for (int sf = 0; sf < 16; sf++) {
            float db = (d > 0.0f) ? d * (0.5f + (float)sf) * 0.25f : 0.0f;
            uint8_t idx[4]; uint32_t signs = 0; float err = 0.0f;
            if (db > 0.0f) {
                for (int l = 0; l < 4; l++) {
                    /* sign-folded magnitude search over the 256 codebook */
                    float bg = 1e30f; int bg_idx = 0; uint32_t sg = 0;
                    for (int g = 0; g < 256; g++) {
                        const uint8_t *cv = (const uint8_t *)(&IQ2XXS_GRID[g]);
                        float e = 0.0f;
                        for (int j = 0; j < 8; j++) {
                            float diff = mags[l * 8 + j] - db * (float)cv[j];
                            e += diff * diff;
                        }
                        if (e < bg) { bg = e; bg_idx = g; }
                    }
                    idx[l] = (uint8_t)bg_idx;
                    for (int j = 0; j < 8; j++)
                        if (s[l * 8 + j] < 0.0f) sg |= (1u << j);
                    /* re-encode error with the chosen codebook + signs */
                    const uint8_t *cv = (const uint8_t *)(&IQ2XXS_GRID[bg_idx]);
                    for (int j = 0; j < 8; j++) {
                        float rec = db * (float)cv[j];
                        if (sg & (1u << j)) rec = -rec;
                        float diff = s[l * 8 + j] - rec;
                        err += diff * diff;
                    }
                    signs |= (sg << (7 * l));
                }
            } else {
                memset(idx, 0, 4);
            }
            if (err < best_err) {
                best_err = err; best_sf = sf;
                memcpy(best_idx, idx, 4);
                best_signs = signs;
            }
        }
        uint32_t aux0 = (uint32_t)best_idx[0] | ((uint32_t)best_idx[1] << 8) |
                        ((uint32_t)best_idx[2] << 16) | ((uint32_t)best_idx[3] << 24);
        uint32_t aux1 = ((uint32_t)best_sf << 28) | (best_signs & 0x0FFFFFFFu);
        memcpy(blk + 2 + ib * 8, &aux0, 4);
        memcpy(blk + 2 + ib * 8 + 4, &aux1, 4);
    }
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

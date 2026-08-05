/*
 * safetensors_reader.c -- HuggingFace safetensors loader (C11, self-contained).
 *
 * Parses the JSON header, exposes an opaque tensor table, and dequantizes
 * F32 / F16 / BF16 / I8..I64 tensors to float32.
 *
 * C11: no VLAs, no compound literals, opaque struct, minimal includes.
 */

#include "safetensors_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ---- dtype string -> enum ---- */
static st_dtype_t st_dtype_from_str(const char *s) {
    if (!s) return ST_DTYPE_UNKNOWN;
    if (strcmp(s, "F32")  == 0) return ST_DTYPE_F32;
    if (strcmp(s, "F16")  == 0) return ST_DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return ST_DTYPE_BF16;
    if (strcmp(s, "F8")   == 0) return ST_DTYPE_F8;
    if (strcmp(s, "I8")   == 0) return ST_DTYPE_I8;
    if (strcmp(s, "I16")  == 0) return ST_DTYPE_I16;
    if (strcmp(s, "I32")  == 0) return ST_DTYPE_I32;
    if (strcmp(s, "I64")  == 0) return ST_DTYPE_I64;
    if (strcmp(s, "BOOL") == 0) return ST_DTYPE_BOOL;
    return ST_DTYPE_UNKNOWN;
}

/* ---- tiny JSON tokenizer (header is flat-ish, hand-parsed) ---- */
/* Returns pointer to the value string for key "key" within [p,end).
 * The value is the quoted string following the first "key" : "value" pair. */
static const char *st_json_find_string(const char *p, const char *end,
                                      const char *key) {
    size_t klen = strlen(key);
    for (; p + klen + 2 < end; p++) {
        if (p[0] == '"' && strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
            const char *q = p + 1 + klen + 1;
            while (q < end && *q != ':') q++;
            if (q >= end) return NULL;
            q++;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\n')) q++;
            if (q < end && *q == '"') return q + 1;
            return NULL;
        }
    }
    return NULL;
}

/* Returns pointer to the first char AFTER the ':' for key "key"
 * (i.e. the raw value start: '[', '{', '"', or a digit). NULL if absent. */
static const char *st_json_value_ptr(const char *p, const char *end,
                                      const char *key) {
    size_t klen = strlen(key);
    for (; p + klen + 1 < end; p++) {
        if (p[0] == '"' && strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
            const char *q = p + 1 + klen + 1;
            while (q < end && *q != ':') q++;
            if (q >= end) return NULL;
            q++;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\n')) q++;
            return q;
        }
    }
    return NULL;
}

/* Parse a JSON integer starting at *pp (advances *pp past it). */
static int64_t st_json_int(const char **pp, const char *end) {
    const char *p = *pp;
    while (p < end && (*p < '0' || *p > '9') && *p != '-') p++;
    if (p >= end) { *pp = p; return 0; }
    int64_t sign = 1, val = 0;
    if (*p == '-') { sign = -1; p++; }
    while (p < end && *p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
    *pp = p;
    return val * sign;
}

/* Parse a tensor's data_offsets [begin,end] pair at *pp, advance past ']'. */
static void st_json_offsets(const char **pp, const char *end,
                           uint64_t *begin, uint64_t *e) {
    const char *p = *pp;
    while (p < end && *p != '[') p++;
    if (p >= end) { *pp = p; return; }
    p++;
    *begin = (uint64_t)st_json_int(&p, end);
    while (p < end && *p != ',') p++;
    if (p < end) p++;
    *e = (uint64_t)st_json_int(&p, end);
    while (p < end && *p != ']') p++;
    if (p < end) p++;
    *pp = p;
}

/* Parse the integer shape array at *pp ("shape":[a,b,c]) -> dims, return n_dims. */
static int st_json_shape(const char **pp, const char *end, int64_t *dims, int cap) {
    const char *p = *pp;
    while (p < end && *p != '[') p++;
    if (p >= end) { *pp = p; return 0; }
    p++;
    int n = 0;
    while (p < end && *p != ']' && n < cap) {
        int64_t v = st_json_int(&p, end);
        dims[n++] = v;
        while (p < end && *p != ',' && *p != ']') p++;
        if (p < end && *p == ',') p++;
    }
    if (p < end && *p == ']') p++;
    *pp = p;
    return n;
}

/* ---- F16 / BF16 -> F32 (used by st_read_tensor_f32) ---- */
float st_f16_to_f32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp  = (v >> 10) & 0x1F;
    int mant = v & 0x03FF;
    if (exp == 0) {
        float s = (sign ? -1.0f : 1.0f);
        return ldexpf((float)mant / 1024.0f, -14) * s;
    }
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    float s = (sign ? -1.0f : 1.0f);
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * s;
}

float st_bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;   // bf16 is the top 16 bits of f32
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---- opaque context ---- */
struct st_ctx {
    FILE    *file;           // NULL when mmap'd
    uint8_t *blob;          // mmap or malloc'd header+pad+raw
    int       blob_owned;     // 1 if we malloc'd (fallback); 0 if mmap'd
    uint64_t header_len;     // JSON length (from first 8 bytes)
    uint64_t raw_off;        // byte offset of raw tensor data = 8 + align8(8+header_len)
    uint8_t *raw;           // pointer to raw data start
    int64_t  n_tensors;
    st_tensor_info *tensors; // heap array
    /* mmap bookkeeping (zero-copy load path) */
    int       fd;            // -1 if not mmap'd
    uint8_t  *map_base;      // MAP_FAILED if not mmap'd
    uint64_t  map_size;      // total mapped bytes
};

int64_t st_n_tensors(const st_ctx *ctx) { return ctx ? ctx->n_tensors : 0; }

const st_tensor_info *st_tensor_info_by_index(const st_ctx *ctx, int64_t idx) {
    if (!ctx || idx < 0 || idx >= ctx->n_tensors) return NULL;
    return &ctx->tensors[idx];
}

const st_tensor_info *st_find_tensor(const st_ctx *ctx, const char *name) {
    if (!ctx) return NULL;
    for (int64_t i = 0; i < ctx->n_tensors; i++)
        if (strcmp(ctx->tensors[i].name, name) == 0) return &ctx->tensors[i];
    return NULL;
}

int st_dtype_size(st_dtype_t dt) {
    switch (dt) {
        case ST_DTYPE_F32:  return 4;
        case ST_DTYPE_F16:  return 2;
        case ST_DTYPE_BF16: return 2;
        case ST_DTYPE_F8:   return 1;
        case ST_DTYPE_I8:   return 1;
        case ST_DTYPE_I16:  return 2;
        case ST_DTYPE_I32:  return 4;
        case ST_DTYPE_I64:  return 8;
        case ST_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

static uint64_t st_align8(uint64_t v) { return (v + 7u) & ~7u; }

st_ctx *st_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    uint8_t hdr_len_buf[8];
    if (fread(hdr_len_buf, 1, 8, f) != 8) { fclose(f); return NULL; }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) header_len |= (uint64_t)hdr_len_buf[i] << (8 * i);

    uint64_t raw_off = st_align8(8 + header_len);
    _fseeki64(f, 0, SEEK_END);
    uint64_t file_sz = (uint64_t)_ftelli64(f);
    fseek(f, 0, SEEK_SET);
    if (raw_off > file_sz) { fclose(f); return NULL; }

    /* Zero-copy path: mmap the whole file read-only. The header + raw tensor
     * bytes stay in the page cache (shared, demand-paged) — we never copy the
     * 55 GB of shards into a malloc'd buffer. Per-tensor dequant still happens
     * into caller buffers, but big tensors (embed_tokens / lm_head) can be
     * accessed directly from the map via st_tensor_raw_ptr() for lazy use. */
    uint8_t *blob = NULL;
    int   fd = -1;
    int   blob_owned = 0;
    uint8_t *map_base = MAP_FAILED;

    fd = open(path, O_RDONLY);
    if (fd >= 0) {
        map_base = mmap(NULL, (size_t)file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map_base != MAP_FAILED) {
            blob = map_base;
        } else {
            close(fd); fd = -1;
        }
    }
    if (blob == NULL) {
        /* Fallback: buffered read (small shards / mmap unavailable). */
        blob = (uint8_t *)malloc(file_sz);
        if (!blob) { fclose(f); return NULL; }
        blob_owned = 1;
        if (fread(blob, 1, file_sz, f) != file_sz) { free(blob); fclose(f); return NULL; }
    }
    fclose(f);

    const char *json = (const char *)(blob + 8);
    const char *json_end = (const char *)(blob + 8 + header_len);

    /* ---- Robust single-pass parse of the JSON header ---- */
    /* First pass: count real tensor entries (skip "__"-prefixed keys). */
    int64_t n = 0;
    {
        const char *p = json;
        while (p < json_end) {
            if (*p != '"') { p++; continue; }
            /* read key name (find matching close quote) */
            const char *ks = p + 1;
            const char *ke = ks;
            while (ke < json_end && *ke != '"') ke++;
            if (ke >= json_end) break;
            size_t klen = (size_t)(ke - ks);
            int is_meta = (klen >= 2 && ks[0] == '_' && ks[1] == '_');
            if (!is_meta) n++;
            /* skip the value to find the next key */
            const char *v = ke + 1;
            while (v < json_end && *v != ':') v++;
            if (v >= json_end) break;
            v++;
            /* skip value (string | number | array | object) */
            if (*v == '"') {
                v++;
                while (v < json_end && *v != '"') v++;
                if (v < json_end) v++;
            } else if (*v == '[' || *v == '{') {
                char open = *v, close = (*v == '[') ? ']' : '}';
                int depth = 0;
                for (; v < json_end; v++) {
                    if (*v == open) depth++;
                    else if (*v == close) { depth--; if (depth == 0) { v++; break; } }
                }
            } else {
                while (v < json_end && *v != ',' && *v != '}') v++;
            }
            p = v;
        }
    }

    st_tensor_info *tensors = (st_tensor_info *)calloc((size_t)(n > 0 ? n : 1),
                                                     sizeof(st_tensor_info));
    if (!tensors) { free(blob); return NULL; }

    int64_t filled = 0;
    {
        const char *p = json;
        while (p < json_end && filled < n) {
            if (*p != '"') { p++; continue; }
            const char *ks = p + 1;
            const char *ke = ks;
            while (ke < json_end && *ke != '"') ke++;
            if (ke >= json_end) break;
            size_t klen = (size_t)(ke - ks);
            /* skip "__"-prefixed metadata keys entirely */
            if (klen >= 2 && ks[0] == '_' && ks[1] == '_') {
                const char *v = ke + 1;
                while (v < json_end && *v != ':') v++;
                if (v < json_end) v++;
                if (*v == '[' || *v == '{') {
                    char open = *v, close = (*v == '[') ? ']' : '}';
                    int depth = 0;
                    for (; v < json_end; v++) {
                        if (*v == open) depth++;
                        else if (*v == close) { depth--; if (depth == 0) { v++; break; } }
                    }
                }
                p = v;
                continue;
            }

            st_tensor_info *t = &tensors[filled];
            if (klen > 255) klen = 255;
            memcpy(t->name, ks, klen);
            t->name[klen] = '\0';

            /* value object for this tensor */
            const char *body = ke + 1;
            while (body < json_end && *body != '{') body++;
            if (body >= json_end) break;
            const char *body_end = body;
            int depth = 0;
            for (; body_end < json_end; body_end++) {
                if (*body_end == '{') depth++;
                else if (*body_end == '}') { depth--; if (depth == 0) break; }
            }

            const char *dts = st_json_find_string(body, body_end, "dtype");
            if (dts) {
                char dtbuf[16];
                int di = 0;
                while (dts[di] && dts[di] != '"' && di < 15) { dtbuf[di] = dts[di]; di++; }
                dtbuf[di] = '\0';
                t->dtype = st_dtype_from_str(dtbuf);
            } else {
                t->dtype = ST_DTYPE_UNKNOWN;
            }

            const char *shp = st_json_value_ptr(body, body_end, "shape");
            if (shp) t->n_dims = st_json_shape(&shp, body_end, t->dims, 8);

            const char *off = st_json_value_ptr(body, body_end, "data_offsets");
            if (off) st_json_offsets(&off, body_end, &t->data_begin, &t->data_end);

            t->n_elems = 1;
            for (int d = 0; d < t->n_dims; d++) {
                if (t->dims[d] <= 0) { t->n_elems = 0; break; }
                t->n_elems *= t->dims[d];
            }

            filled++;
            p = body_end + 1;
        }
    }

    st_ctx *ctx = (st_ctx *)calloc(1, sizeof(st_ctx));
    if (!ctx) {
        if (map_base != MAP_FAILED) munmap(map_base, (size_t)file_sz);
        if (fd >= 0) close(fd);
        if (blob_owned) free(blob);
        return NULL;
    }
    ctx->file = NULL;
    ctx->blob = blob;
    ctx->blob_owned = (map_base == MAP_FAILED) ? 1 : 0;
    ctx->header_len = header_len;
    ctx->raw_off = raw_off;
    ctx->raw = blob + raw_off;
    ctx->n_tensors = filled;
    ctx->tensors = tensors;
    ctx->fd = (map_base != MAP_FAILED) ? fd : -1;
    ctx->map_base = map_base;
    ctx->map_size = (map_base != MAP_FAILED) ? file_sz : 0;
    return ctx;
}

int st_read_tensor_f32(const st_ctx *ctx, const st_tensor_info *info,
                       float *output, int64_t max_elems) {
    if (!ctx || !info || !output) return 0;
    int esz = st_dtype_size(info->dtype);
    if (esz == 0) return 0;
    int64_t n = info->n_elems;
    if (n > max_elems) n = max_elems;
    const uint8_t *src = ctx->raw + info->data_begin;

    switch (info->dtype) {
        case ST_DTYPE_F32:
            memcpy(output, src, (size_t)n * 4);
            return (int)n;
        case ST_DTYPE_F16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int64_t i = 0; i < n; i++) output[i] = st_f16_to_f32(s[i]);
            return (int)n;
        }
        case ST_DTYPE_BF16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int64_t i = 0; i < n; i++) output[i] = st_bf16_to_f32(s[i]);
            return (int)n;
        }
        case ST_DTYPE_I8:
        case ST_DTYPE_I16:
        case ST_DTYPE_I32:
        case ST_DTYPE_I64:
        case ST_DTYPE_BOOL:
        case ST_DTYPE_F8:
        default:
            /* integer / unknown -> reject (caller can use raw path) */
            return 0;
    }
}

int64_t st_read_tensor_raw(const st_ctx *ctx, const st_tensor_info *info,
                           void *output, int64_t max_bytes) {
    if (!ctx || !info || !output) return 0;
    uint64_t want = info->data_end - info->data_begin;
    if (want > (uint64_t)max_bytes) want = (uint64_t)max_bytes;
    memcpy(output, ctx->raw + info->data_begin, (size_t)want);
    return (int64_t)want;
}

const uint8_t *st_tensor_raw_ptr(const st_ctx *ctx, const st_tensor_info *info) {
    if (!ctx || !info) return NULL;
    return ctx->raw + info->data_begin;
}

int st_dequant_row(const st_tensor_info *info, const uint8_t *raw_base,
                   int64_t row, float *out) {
    if (!info || !raw_base || !out || row < 0) return 0;
    if (info->n_dims < 2) return 0;
    int64_t row_elems = 1;
    for (int d = 1; d < info->n_dims; d++) row_elems *= info->dims[d];
    if (row_elems <= 0 || row >= info->dims[0]) return 0;
    int esz = st_dtype_size(info->dtype);
    if (esz == 0) return 0;
    const uint8_t *src = raw_base + (size_t)row * (size_t)row_elems * esz;
    switch (info->dtype) {
        case ST_DTYPE_F32:
            memcpy(out, src, (size_t)row_elems * 4);
            return 1;
        case ST_DTYPE_F16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int64_t i = 0; i < row_elems; i++) out[i] = st_f16_to_f32(s[i]);
            return 1;
        }
        case ST_DTYPE_BF16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int64_t i = 0; i < row_elems; i++) out[i] = st_bf16_to_f32(s[i]);
            return 1;
        }
        default:
            return 0;  /* integer/unknown: unsupported for lazy dequant */
    }
}

void st_close(st_ctx *ctx) {
    if (!ctx) return;
    if (ctx->map_base != MAP_FAILED && ctx->map_size > 0) {
        munmap(ctx->map_base, (size_t)ctx->map_size);
        if (ctx->fd >= 0) close(ctx->fd);
    } else if (ctx->blob_owned && ctx->blob) {
        free(ctx->blob);
    }
    if (ctx->tensors) free(ctx->tensors);
    free(ctx);
}

/*
 * wubu_tensor_store.h -- the uniform tensor catalog (anti-waste interchange).
 *
 * The user's directive (2026-08-04): "we need to be able to live load
 * convert and be able to interchange between all of the formats and the
 * way we do it right now is incredibly wasteful." A model format is just
 * a CATALOG over the same bytes. This store opens ANY of the three native
 * formats (safetensors / GGUF / .st dump) WITHOUT loading the weights,
 * serves tensors by name via the existing zero-copy readers, and exports
 * STREAMING (one tensor at a time, bounded RAM) -- never load-all-then-save.
 *
 * Formats (sniffed by magic):
 *   WUBU_TS_SAFETENSORS  -- 8-byte LE header-len + JSON table (mmap'd)
 *   WUBU_TS_GGUF         -- "GGUF" magic + tensor table
 *   WUBU_TS_STDUMP       -- 0xBA000001/2 + param count + fixed 137-tensor
 *                           float dump (the trainer's save_checkpoint)
 *
 * C11, self-contained; wraps the existing st_ctx / gguf_ctx readers.
 */
#ifndef WUBU_TENSOR_STORE_H
#define WUBU_TENSOR_STORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_TS_UNKNOWN = 0,
    WUBU_TS_SAFETENSORS,
    WUBU_TS_GGUF,
    WUBU_TS_STDUMP
} wubu_ts_fmt;

/* one catalog entry: name + element count + byte offset in the backing
 * file (safetensors/.st) or -1 (gguf -- addressed via its own reader). */
typedef struct wubu_ts_entry {
    char name[192];
    int64_t n_elems;
    int64_t offset;      /* byte offset of the raw data (f32 for st, raw for st) */
    int ggml_type;       /* 0 = f32 (safetensors/.st); ggml enum for gguf */
    int n_dims;
    int64_t dims[4];
} wubu_ts_entry;

typedef struct wubu_tensor_store wubu_tensor_store_t;

/* magic-based format detection; no allocation. */
wubu_ts_fmt wubu_ts_sniff(const char *path);

/* open a model file as a tensor catalog. Does NOT load weights. NULL on error. */
wubu_tensor_store_t *wubu_ts_open(const char *path);

/* format + entry access */
wubu_ts_fmt  wubu_ts_format(const wubu_tensor_store_t *ts);
int          wubu_ts_count(const wubu_tensor_store_t *ts);
const wubu_ts_entry *wubu_ts_entry_at(const wubu_tensor_store_t *ts, int i);
const wubu_ts_entry *wubu_ts_find(const wubu_tensor_store_t *ts, const char *name);

/* LIVE LOAD: materialize ONE tensor as f32 (the caller owns `out`,
 * max_elems must be >= entry->n_elems). 0 on success, -1 on error.
 * This is the interchange primitive: any tensor, any format, on demand. */
int wubu_ts_get_f32(const wubu_tensor_store_t *ts, const char *name,
                    float *out, int64_t max_elems);

/* STREAMING EXPORT: convert the whole catalog to another format, one
 * tensor at a time (bounded RAM -- never holds more than one tensor).
 * target: WUBU_TS_SAFETENSORS | WUBU_TS_GGUF | WUBU_TS_STDUMP.
 * For WUBU_TS_GGUF the source must already be f32-compatible (or gguf
 * with dequant); all tensors are written as f32. 0 on success. */
int wubu_ts_export(const wubu_tensor_store_t *ts, wubu_ts_fmt target,
                   const char *out_path);

void wubu_ts_close(wubu_tensor_store_t *ts);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TENSOR_STORE_H */

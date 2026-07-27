/*
 * wubu_safetensors_shard.h -- multi-shard safetensors loader.
 *
 * HF releases split large models across many files:
 *   model-00000-of-00013.safetensors, model-00001-of-00013.safetensors, ...
 * Each shard is a valid safetensors file with its OWN header. This module
 * opens the whole set and routes a tensor name to the shard that owns it,
 * so callers load real multi-GB Colonel checkpoints transparently.
 *
 * Self-contained: depends only on safetensors_reader.h. No god headers.
 * C11, opaque ctx.
 */
#ifndef WUBU_SAFETENSORS_SHARD_H
#define WUBU_SAFETENSORS_SHARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_shard_ctx wubu_shard_ctx_t;

/* Open all shards of a model given ONE shard path (e.g. the -00000- file)
 * or a directory containing model-000NN-of-000MM.safetensors. Returns NULL
 * on failure. Scans for sibling shards automatically. */
wubu_shard_ctx_t *wubu_shard_open(const char *path_or_dir);

/* Number of shards opened. */
int wubu_shard_count(const wubu_shard_ctx_t *sc);

/* Total tensors across all shards. */
int64_t wubu_shard_n_tensors(const wubu_shard_ctx_t *sc);

/* Load a named tensor as a freshly malloc'd F32 buffer (caller frees).
 * On success returns the pointer and writes the element count to
 * *n_elems_out; on failure returns NULL. Mirrors st_load_f32's
 * allocate-and-return convention so callers need no scratch buffers.
 * For transposed loads (HF [out,in] -> bytropix [in,out]) use _t. */
float *wubu_shard_load_f32(const wubu_shard_ctx_t *sc, const char *name,
                           int64_t *n_elems_out);

/* Load a named tensor as F32 and transpose from [rows,cols] to [cols,rows]
 * (HF Linear weight layout -> bytropix forward layout). Returns a freshly
 * malloc'd buffer of rows*cols floats (caller frees), or NULL on error. */
float *wubu_shard_load_f32_t(const wubu_shard_ctx_t *sc, const char *name,
                             int rows, int cols);

/* Raw pointer to a tensor's F32 data (no copy). Valid until wubu_shard_close.
 * Convenient when the caller will free via the shard. Returns NULL if absent.
 * The returned buffer is owned by the shard; do NOT free it. */
const float *wubu_shard_data_f32(const wubu_shard_ctx_t *sc, const char *name,
                                 int64_t *n_elems_out);

/* Close + free all shards. */
void wubu_shard_close(wubu_shard_ctx_t *sc);

/* Read dimension `i` of a named tensor (across all shards), or -1 if absent. */
int wubu_shard_dimof(const wubu_shard_ctx_t *sc, const char *name, int i);

/* Whether a named tensor exists across the shards. */
int wubu_shard_has(const wubu_shard_ctx_t *sc, const char *name);

/* Zero-copy raw accessor: returns a pointer to a tensor's still-encoded bytes
 * (in-place in the mmap'd file) plus its dtype + row stride in bytes. The
 * caller dequantizes only the rows it needs (e.g. one embedding row / one
 * lm_head row). Returns the pointer, or NULL if absent. `out_dtype`/`out_row`
 * (row length in elements) let the caller dequantize a single row cheaply.
 * The pointer is owned by the shard; do NOT free it. */
const uint8_t *wubu_shard_raw(const wubu_shard_ctx_t *sc, const char *name,
                              int *out_dtype, int64_t *out_row);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SAFETENSORS_SHARD_H */

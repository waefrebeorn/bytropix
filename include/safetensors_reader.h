#ifndef SAFETENSORS_READER_H
#define SAFETENSORS_READER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * safetensors_reader.h -- read HuggingFace safetensors files (the format ALL
 * of the new bytropix models ship in: Qwen3.6-27B, Agents-A1-4B,
 * KAT-Coder-V2.5-Dev, BTL-3).
 *
 * safetensors layout:
 *   [ uint64 LE header_len ][ header JSON (header_len bytes) ]
 *   [ padding to 8-byte boundary ]
 *   [ raw tensor bytes ]
 *
 * The header is a JSON object mapping tensor-name -> {
 *     "dtype": "F32"|"F16"|"BF16"|"I64"|...,
 *     "shape": [ d0, d1, ... ],
 *     "data_offsets": [ begin, end ]   // relative to START of raw blob
 * }.
 *
 * Opaque struct: callers never touch the mmap'd buffer directly.
 */

typedef enum {
    ST_DTYPE_F32  = 0,
    ST_DTYPE_F16  = 1,
    ST_DTYPE_BF16 = 2,
    ST_DTYPE_F8   = 3,
    ST_DTYPE_I8   = 4,
    ST_DTYPE_I16  = 5,
    ST_DTYPE_I32  = 6,
    ST_DTYPE_I64  = 7,
    ST_DTYPE_BOOL = 8,
    ST_DTYPE_UNKNOWN = 99
} st_dtype_t;

typedef struct {
    char      name[256];
    st_dtype_t dtype;
    int       n_dims;
    int64_t   dims[8];
    int64_t   n_elems;
    uint64_t  data_begin;   // relative to raw blob start
    uint64_t  data_end;
} st_tensor_info;

typedef struct st_ctx st_ctx;   // opaque

// Open a safetensors file and parse the header. Returns NULL on failure.
st_ctx *st_open(const char *path);

// Number of tensors declared in the header.
int64_t st_n_tensors(const st_ctx *ctx);

// Fetch tensor info by index (0..n-1). Returns NULL if out of range.
const st_tensor_info *st_tensor_info_by_index(const st_ctx *ctx, int64_t idx);

// Find a tensor by name. Returns NULL if not present.
const st_tensor_info *st_find_tensor(const st_ctx *ctx, const char *name);

// Return number of bytes per element for a dtype (0 for unknown).
int st_dtype_size(st_dtype_t dt);

// F16 / BF16 -> F32 conversion helpers (also used by lazy embed/lm_head).
float st_f16_to_f32(uint16_t v);
float st_bf16_to_f32(uint16_t v);

// Dequantize one tensor's raw bytes to F32 into caller buffer.
// output must hold at least info->n_elems floats.
// Returns number of floats written, or 0 on error.
int st_read_tensor_f32(const st_ctx *ctx, const st_tensor_info *info,
                        float *output, int64_t max_elems);

// Copy one tensor's raw bytes verbatim (for re-packing / LoRA merges).
// Returns bytes copied, or 0 on error.
int64_t st_read_tensor_raw(const st_ctx *ctx, const st_tensor_info *info,
                           void *output, int64_t max_bytes);

// Zero-copy access to a tensor's still-encoded bytes, in-place in the
// (possibly mmap'd) file. Caller must NOT free. NULL if absent. Use for big
// tensors (embed_tokens / lm_head): dequantize only the rows you need.
const uint8_t *st_tensor_raw_ptr(const st_ctx *ctx, const st_tensor_info *info);

// Dequantize a single row [row] (length info->dims[1..]) of an F32/F16/BF16
// tensor into a caller F32 buffer. Returns 1 on success, 0 if dtype/row bad.
// Used for lazy, per-row embedding / lm_head access (no full-tensor copy).
int st_dequant_row(const st_tensor_info *info, const uint8_t *raw_base,
                   int64_t row, float *out);

// Close and free.
void st_close(st_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif // SAFETENSORS_READER_H

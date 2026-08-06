// include/wubu_weight.h
// Universal weight descriptor — ONE representation for every weight of
// every model (GGUF, safetensors, raw bin; F32/F16/BF16/Q4_0/Q8_0/K-quants/
// IQ/TQ; dense, MoE, SSM, GQA, embeddings, lm_head).
//
// Design doctrine (Hivemind AGI — "load every model, train them a little"):
//   * A weight is (data pointer, ggml type tag, element count). Nothing else.
//   * ANY loader fills the descriptor identically. There is no per-format
//     field (no attn_q_weight_q vs attn_q_weight_raw vs attn_q_weight_f32).
//   * ONE materializer turns it into F32 on demand (lazy, zero-copy until
//     the layer is active).
//   * ONE matmul entry point dispatches on the type tag internally —
//     quantized dot when the type + alignment allow it, else dequant+SGEMM.
//   * Callers never branch on storage type. The engine is type-agnostic
//     BY CONSTRUCTION, which is what lets one loader handle every file
//     and one forward handle every checkpoint.
#ifndef WUBU_WEIGHT_H
#define WUBU_WEIGHT_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_reader.h"   // GGML_TYPE_* tags are the canonical type space

typedef struct {
    const uint8_t *data;    // raw bytes: mmap'd GGUF blob or heap (never freed here)
    int            type;    // GGML_TYPE_* (F32/F16/BF16/Q4_0/Q8_0/Q*_K/IQ*/TQ)
    int64_t        n_elems; // total scalar elements (product of dims)
} wubu_weight_t;

// Bytes occupied by n_elems of `type` (from the file's own layout rules).
// Returns 0 for unknown types. Used by loaders to bound-check blob spans.
int64_t wubu_weight_nbytes(int type, int64_t n_elems);

// True if the type is float storage (F32/F16/BF16) vs quantized blocks.
int wubu_weight_is_float(int type);

// True if quantized matmul can run directly on this type + row count
// (K-quants need rows % 256 == 0; legacy Q4_0/Q8_0 need rows % 32 == 0;
// float types always fine).
int wubu_weight_direct_ok(int type, int64_t n_rows);

// Universal materializer: dequant `w` into `out` (n_elems floats).
// Handles every known type (F32 copy, F16/BF16 convert, Q4_0/Q8_0,
// Q2_K..Q8_K, IQ*, TQ* when the reader knows them). Returns 0 on success,
// -1 on unknown type. Thread-safe (reads only).
int wubu_weight_to_f32(const wubu_weight_t *w, float *out);

// Universal matmul: y = x @ W.
//   x   : [n_rows] F32 input (activation row)
//   w   : weight descriptor; W is [out=n_cols, in=n_rows] (col-major per col)
//   n_rows, n_cols : logical dims of W
//   y   : [n_cols] F32 output
// Dispatches: float types -> direct SGEMM (via quantized_matmul's F32/F16/BF16
// kernels); quantized + aligned -> quantized dot; otherwise dequant+SGEMM.
// Thread-safe (OpenMP inside). Never asserts on alignment — degrades.
void wubu_weight_matmul(const float *x, const wubu_weight_t *w,
                        int64_t n_rows, int64_t n_cols, float *y);

// Convenience: fill descriptor from a GGUF tensor (blob base + offset).
// Returns the descriptor (or {0} if t/blob are NULL).
wubu_weight_t wubu_weight_from_gguf(const uint8_t *blob,
                                    const void *t /* gguf_tensor_info* */);

#endif // WUBU_WEIGHT_H

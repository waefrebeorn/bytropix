// src/wubu_weight.c
// Universal weight descriptor implementation.
//
// ONE representation for every weight of every model; ONE materializer
// (delegates to gguf_dequantize, the file-truth dequantizer); ONE matmul
// dispatcher (delegates to quantized_matmul, which already handles
// F32/F16/BF16/Q8_0/K-quants internally and falls back to dequant+SGEMM).
//
// Because EVERY loader fills wubu_weight_t identically and EVERY consumer
// reads it through wubu_weight_to_f32 / wubu_weight_matmul, there is no
// per-format branch anywhere in the engine. Adding a new quant type means
// touching gguf_dequantize / quantized_matmul — not the model structs, not
// the forwards, not the loaders. That is the Hivemind universal-manifold
// contract: any checkpoint, any storage, one path.

#include "wubu_weight.h"
#include <string.h>

int64_t wubu_weight_nbytes(int type, int64_t n_elems) {
    if (n_elems <= 0) return 0;
    int64_t sz = gguf_raw_size(type, n_elems);
    if (sz <= 0) {
        /* gguf_raw_size returns 0 for types it does not know — try the
         * per-element fallback for float storage. */
        switch (type) {
            case GGML_TYPE_F32: sz = n_elems * 4; break;
            case GGML_TYPE_F16: case GGML_TYPE_BF16: sz = n_elems * 2; break;
            default: break;
        }
    }
    return sz;
}

int wubu_weight_is_float(int type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16;
}

int wubu_weight_direct_ok(int type, int64_t n_rows) {
    if (n_rows <= 0) return 0;
    /* Float storage: direct SGEMM always fine. */
    if (wubu_weight_is_float(type)) return 1;
    /* Legacy block types: 32-element blocks (Q4_0/Q8_0/Q4_1/Q5_0/Q5_1). */
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1:
            return (n_rows % 32 == 0);
        default:
            break;
    }
    /* K-quants + IQ + TQ: 256-element blocks. */
    return (n_rows % 256 == 0);
}

int wubu_weight_to_f32(const wubu_weight_t *w, float *out) {
    if (!w || !w->data || !out || w->n_elems <= 0) return -1;
    if (wubu_weight_is_float(w->type)) {
        switch (w->type) {
            case GGML_TYPE_F32:
                memcpy(out, w->data, (size_t)w->n_elems * sizeof(float));
                return 0;
            case GGML_TYPE_F16: {
                const uint16_t *src = (const uint16_t *)w->data;
                for (int64_t i = 0; i < w->n_elems; i++) {
                    uint32_t sign = (src[i] >> 15) & 1, exp = (src[i] >> 10) & 0x1F,
                             mant = src[i] & 0x03FF, f32;
                    if (exp == 0) f32 = (sign<<31)|((uint32_t)(127-15+1)<<23)|(mant<<13);
                    else if (exp == 31) f32 = (sign<<31)|(0xFF<<23)|(mant<<13);
                    else f32 = (sign<<31)|((uint32_t)(127-15+exp)<<23)|(mant<<13);
                    memcpy(&out[i], &f32, 4);
                }
                return 0;
            }
            case GGML_TYPE_BF16: {
                const uint16_t *src = (const uint16_t *)w->data;
                for (int64_t i = 0; i < w->n_elems; i++) {
                    uint32_t f32 = (uint32_t)src[i] << 16;
                    memcpy(&out[i], &f32, 4);
                }
                return 0;
            }
            default: return -1;
        }
    }
    /* Quantized / int / fp64: the reader's universal dequantizer. */
    gguf_dequantize(w->data, w->type, w->n_elems, out);
    return 0;
}

void wubu_weight_matmul(const float *x, const wubu_weight_t *w,
                        int64_t n_rows, int64_t n_cols, float *y) {
    if (!x || !w || !w->data || !y || n_rows <= 0 || n_cols <= 0) return;
    /* quantized_matmul is itself type-agnostic: it dispatches on the tag
     * and dequant+SGEMM-falls-back when a type can't run directly. Pass
     * the descriptor straight through. */
    quantized_matmul(x, w->data, w->type, n_rows, n_cols, 0, y);
}

wubu_weight_t wubu_weight_from_gguf(const uint8_t *blob, const void *tv) {
    wubu_weight_t w = {0};
    if (!blob || !tv) return w;
    const gguf_tensor_info *t = (const gguf_tensor_info *)tv;
    w.data   = blob + t->data_offset;
    w.type   = t->ggml_type;
    w.n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) w.n_elems *= t->dims[d];
    return w;
}

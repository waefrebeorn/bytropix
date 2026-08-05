/*
 * wubu_h3_norm.h — MiniMax H3 hyperbolic neural network normalization kernel.
 *
 * H3 (Hyperbolic) activation replaces the SiLU/Swish gate in standard MLP
 * blocks with a tanh-based hyperbolic modulation:
 *   gate = SiLU(W_gate * x)     // standard Swish/SiLU gate
 *   up   = tanh (W_up  * x)     // hyperbolic branch — the H3 innovation
 *   out  = gate * up            // elementwise multiply
 *
 * NF4 support: if weights are NF4-quantized (bitsandbytes format), a
 * companion FP32 scale tensor is looked up by the convention
 * "<weight_tensor_name>.scaling_factor".
 *
 * Opaque struct, minimal includes, C11 only.
 */
#ifndef WUBU_H3_NORM_H
#define WUBU_H3_NORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context — callers never touch the internal state. */
typedef struct wubu_h3_norm_ctx wubu_h3_norm_t;

/* Initialize from F32 weights (caller retains ownership of weight buffers).
 * gate_w: [out_dim * in_dim] row-major
 * gate_b: [out_dim] or NULL
 * up_w:   [out_dim * in_dim] row-major
 * up_b:   [out_dim] or NULL
 * Returns NULL on allocation failure or invalid args. */
wubu_h3_norm_t *wubu_h3_norm_init(const float *gate_w, const float *gate_b,
                                 const float *up_w, const float *up_b,
                                 long in_dim, long out_dim);

/* Initialize from NF4 packed weights (bitsandbytes format).
 * gate_raw / up_raw: packed 4-bit codes (2 codes/byte, high nibble first)
 * gate_scale / up_scale: per-tensor FP32 scale values
 * in_dim: number of input features (elements in source tensor)
 * out_dim: number of output features (rows in weight matrix) */
wubu_h3_norm_t *wubu_h3_norm_init_nf4(const uint8_t *gate_raw, float gate_scale,
                                    const uint8_t *up_raw, float up_scale,
                                    long in_dim, long out_dim);

/* Apply H3 activation: out[o] = SiLU(gate_row · x + gate_b[o]) * tanh(up_row · x + up_b[o])
 * x: [in_dim]  input vector
 * out: [out_dim] output vector (caller-allocated) */
void wubu_h3_norm_apply(const wubu_h3_norm_t *ctx,
                        const float *x,
                        float *out);

/* Free internal buffers (does NOT free caller-provided weight pointers). */
void wubu_h3_norm_close(wubu_h3_norm_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_H3_NORM_H */

/*
 * wubu_ops.h — Low-level numerical operators (activations, norm, conv).
 *
 * Extracted from wubu_ssm.c (Strangler Fig, research 066-A1/E2):
 * these pure numerical primitives are used by SSM, GQA, MoE, and backward
 * passes. They live in their own TU (wubu_ops.c) so that changing one
 * activation does not recompile the 3263-line wubu_ssm.c.
 *
 * ADR-002: these are leaf-level primitives, not module seams — they are
 * exposed directly (no opaque wrapping) because they are leaf operations
 * with no internal state. Higher-level modules (wubu_ssm, wubu_gqa)
 * remain behind their own interfaces.
 */
#ifndef WUBU_OPS_H
#define WUBU_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Activations (elementwise) ---- */
void wubu_softplus(int n, const float *x, float *out);
void wubu_silu(int n, const float *x, float *out);
void wubu_sigmoid(int n, const float *x, float *out);

/* ---- Activations (backward) ---- */
void wubu_silu_backward(int n, const float *x, const float *y,
                        const float *dy, float *dx);

/* ---- Normalization ---- */
void wubu_l2_norm(int B, int T, int n_heads, int d,
                  const float *x, float eps, float *out);
void wubu_rms_norm(int B, int T, int d,
                   const float *x, const float *weight, float eps, float *out);
void wubu_l2_norm_backward(int B, int T, int n_heads, int d,
                           const float *x, float eps,
                           const float *d_out, float *d_x);
void wubu_rms_norm_backward(int B, int T, int d,
                            const float *x, const float *weight, float eps,
                            const float *d_out, float *d_x);

/* ---- Convolution ---- */
void wubu_conv1d(int B, int T, int C, int k,
                 const float *input, const float *kernel,
                 float *output);

/* ---- Utility ---- */
int wubu_is_ssm_layer(int layer_idx);

/* ---- SSM L2 norm epsilon (global, set from GGUF config) ---- */
extern float g_ssm_l2_eps;

#ifdef __cplusplus
}
#endif

#endif /* WUBU_OPS_H */

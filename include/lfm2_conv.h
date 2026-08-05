#ifndef LFM2_CONV_H
#define LFM2_CONV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LFM2.5 gated depthwise-causal-conv block (the "conv" / SSM operator).
 * Self-contained: owns the B/C/h_tilde gate + depthwise conv math.
 *
 * Block (LFM2.5 technical report, arxiv 2511.23404):
 *   (B, C, h_tilde) = in_proj(h)      // [3*conv_dim, d_model]
 *   y = B * h_tilde                     // input gate
 *   z = depthwise_causal_conv_k(y)     // Conv1d(groups=conv_dim), kernel k
 *   o = out_proj(C * z)                // output gate + linear -> [d_model]
 *
 * conv_w layout is PyTorch Conv1d weight [out_c, in_c, k] = [conv_dim, conv_dim, k].
 * Depthwise (groups=conv_dim) => output channel c reads input channel c:
 *   w index = ((c * conv_dim) + c) * k + j  =  c * (conv_dim + 1) * k + j
 */

/* in_proj: [3*cd, d] ; conv_w: [cd, cd, k] (F32) ; out_proj: [d, cd].
 * x: [T, d] input (already operator_norm'd). Writes op_out: [T, d]. */
void lfm2_conv(const float *in_proj, const float *conv_w, const float *out_proj,
               int conv_k, int conv_dim, int d_model,
               const float *x, int T, float *op_out);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_CONV_H */

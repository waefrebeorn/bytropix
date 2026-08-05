/* lfm2_conv.c -- LFM2.5 gated depthwise-causal-conv block (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_conv.h"
#include "lfm2_math.h"
#include <stdlib.h>
#include <string.h>

static void depthwise_causal_conv(const float *x, const float *w,
                                  int T, int C, int k, float *y) {
    /* PyTorch Conv1d(hidden, hidden, k, padding=k-1, groups=hidden):
     *   out[t,c] = sum_{j=0..k-1} w[c,(c),k-1-j] * in[t-j, c]   (in padded 0 for t<0)
     * Depthwise: output channel c == input channel c.
     * w layout [C, C, k]: element (oc, ic, j) at ((oc*C)+ic)*k + j. */
    int last = k - 1;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            /* PyTorch depthwise Conv1d(groups=C) stores weight [out_c, 1, k]
             * (in_c compressed to 1 since each out channel uses 1 in channel).
             * So kernel for channel c is at w[c*k + (k-1-j)]. */
            const float *wc = w + (size_t)c * k;
            for (int j = 0; j < k; j++) {
                int tt = t - j;
                if (tt >= 0) s += wc[last - j] * x[(size_t)tt * C + c];
            }
            y[(size_t)t * C + c] = s;
        }
    }
}

void lfm2_conv(const float *in_proj, const float *conv_w, const float *out_proj,
               int conv_k, int conv_dim, int d_model,
               const float *x, int T, float *op_out) {
    int cd = conv_dim, d = d_model, k = conv_k;
    float *proj = (float *)malloc((size_t)T * 3 * cd * sizeof(float));
    lfm2_matmul_f32(x, in_proj, T, d, 3 * cd, proj);

    /* split B, C, h_tilde -- each [T, cd] */
    const float *Bp = proj;
    const float *Cp = proj + (size_t)T * cd;
    const float *Hp = proj + (size_t)T * 2 * cd;

    float *y = (float *)malloc((size_t)T * cd * sizeof(float));
    for (size_t i = 0; i < (size_t)T * cd; i++) y[i] = Bp[i] * Hp[i]; /* input gate */

    float *z = (float *)malloc((size_t)T * cd * sizeof(float));
    depthwise_causal_conv(y, conv_w, T, cd, k, z);

    float *gated = (float *)malloc((size_t)T * cd * sizeof(float));
    for (size_t i = 0; i < (size_t)T * cd; i++) gated[i] = Cp[i] * z[i]; /* output gate */

    lfm2_matmul_f32(gated, out_proj, T, cd, d, op_out);

    free(proj); free(y); free(z); free(gated);
}

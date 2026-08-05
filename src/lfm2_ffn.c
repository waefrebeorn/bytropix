/* lfm2_ffn.c -- LFM2.5 SwiGLU feed-forward (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_ffn.h"
#include "lfm2_math.h"
#include <stdlib.h>
#include <math.h>

void lfm2_ffn(const float *w1, const float *w2, const float *w3,
              int ff, int d, const float *x, int T, float *out) {
    float *g = (float *)malloc((size_t)T * ff * sizeof(float)); /* w1(x) */
    float *u = (float *)malloc((size_t)T * ff * sizeof(float)); /* w3(x) */
    lfm2_matmul_f32(x, w1, T, d, ff, g);
    lfm2_matmul_f32(x, w3, T, d, ff, u);
    for (size_t i = 0; i < (size_t)T * ff; i++) {
        float v = g[i];
        g[i] = v / (1.0f + expf(-v)) * u[i]; /* silu(g) * u */
    }
    lfm2_matmul_f32(g, w2, T, ff, d, out);
    free(g); free(u);
}

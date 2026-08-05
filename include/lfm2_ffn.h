#ifndef LFM2_FFN_H
#define LFM2_FFN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LFM2.5 SwiGLU FFN: h = w2( silu(w1(x)) * w3(x) ).
 * Self-contained. x: [T, d_model]. Writes out: [T, d_model]. */
void lfm2_ffn(const float *w1, const float *w2, const float *w3,
              int ff_dim, int d_model, const float *x, int T, float *out);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_FFN_H */

#ifndef WUBU_ROPE_SCALING_H
#define WUBU_ROPE_SCALING_H
/* wubu_rope_scaling.h — RoPE frequency scaling for context extension (BRAIN #231).
 *
 * Extends a model trained at L_train to run at L_target by scaling the rotary
 * inverse frequencies. Three methods:
 *   - LINEAR:  inv_freq' = inv_freq / s            (s = L_target/L_train)
 *   - NTK:     inv_freq' = inv_freq^(1/s) reshaped  (NTK-aware, no dim ramp)
 *   - YARN:    per-dim scale from wubu_yarn_scales  (NTK + dimensional ramp)
 *
 * C11, self-contained. The forward (wubu_rope_scaling_freqs) is covered by
 * test_rope_scaling with measured assertions (dim-0 invariant, monotonic
 * decrease, exact linear identity, NTK high-freq preservation).
 */
#include <stddef.h>

typedef enum {
    WUBU_ROPE_SCALE_NONE = 0,  /* no scaling (identity) */
    WUBU_ROPE_SCALE_LINEAR,    /* inv_freq / s */
    WUBU_ROPE_SCALE_NTK,       /* NTK-aware: (base*freq_i)^(1/s) */
    WUBU_ROPE_SCALE_YARN       /* YaRN per-dim (needs wubu_yarn) */
} wubu_rope_scaling_t;

/* Compute scaled inverse frequencies for `half` = d/2 dimensions.
 * base: rope base (e.g. 10000.0). s = L_target/L_train (>=1).
 * method: one of the enum above. out_inv_freq: caller array of `half` doubles.
 * Returns 0 on ok, -1 on bad args (half<=0, s<1, method unknown). */
int wubu_rope_scaling_freqs(int half, double base, double s,
                            wubu_rope_scaling_t method,
                            double *out_inv_freq);

/* Human-readable method name (for logs/tests). */
const char *wubu_rope_scaling_name(wubu_rope_scaling_t method);

#endif

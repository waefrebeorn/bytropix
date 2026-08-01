/*
 * wubu_dn2.c -- Gated DeltaNet-2 (S02) + ternary STE proxy (T04). C11.
 *
 * Convergence (Gated DeltaNet-2 decoupled erase/write / BitNet STE 7-hop):
 *   - S02 Gated DeltaNet-2: decouples erase and write. Two gates e (erase) and
 *        w (write). Step 1 (erase): S_e = (1 - e) * S + e * (S - beta*(S k - v) k^T)
 *        -- i.e. erase removes the old association. Step 2 (write): S' = S_e +
 *        w * beta * (v - S_e k) k^T  -- writes the new association. Decoupling
 *        lets erase and write be controlled independently (the GDN-2 paper).
 *   - T04 ternary Straight-Through Estimator proxy: forward ternaryizes
 *        (sign clip); the STE backward passes gradient unchanged for |x|<=1 and
 *        0 otherwise (hard-tanh surrogate). We return the forward ternary value
 *        and a flag whether gradient passes (|x|<=1).
 *
 * Triple-DA: dims/gates clamped to [0,1]; null -> 0; deterministic.
 */
#include "wubu_dn2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void mat_scale(float *S, int n, float a) { for (int i=0;i<n;i++) S[i]*=a; }

/* S02 Gated DeltaNet-2 decoupled erase/write. */
int wubu_dn2_update(const float *S, const float *k, const float *v,
                    int d, float beta, float erase_g, float write_g, float *Sout) {
    if (!S || !k || !v || !Sout || d <= 0) return 0;
    if (beta < 0.0f) beta = 0.0f; if (beta > 1.0f) beta = 1.0f;
    if (erase_g < 0.0f) erase_g = 0.0f; if (erase_g > 1.0f) erase_g = 1.0f;
    if (write_g < 0.0f) write_g = 0.0f; if (write_g > 1.0f) write_g = 1.0f;

    memcpy(Sout, S, (size_t)d*d*sizeof(float));
    /* delta-rule association delta = (S k - v) */
    float *delta = (float *)calloc((size_t)d, sizeof(float));
    if (!delta) return 0;
    for (int i = 0; i < d; i++) {
        float sk = 0.0f; for (int j = 0; j < d; j++) sk += Sout[i*d+j]*k[j];
        delta[i] = sk - v[i];
    }
    /* erase step: Sout = (1-e)*S + e*(S - beta*delta k^T) */
    float *Se = (float *)malloc((size_t)d*d*sizeof(float));
    if (!Se) { free(delta); return 0; }
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Se[i*d+j] = (1.0f - erase_g)*Sout[i*d+j]
                      + erase_g*(Sout[i*d+j] - beta*delta[i]*k[j]);
    /* write step: Sout = Se + w*beta*(v - Se k) k^T */
    for (int i = 0; i < d; i++) {
        float sek = 0.0f; for (int j = 0; j < d; j++) sek += Se[i*d+j]*k[j];
        float tgt = v[i] - sek;
        for (int j = 0; j < d; j++)
            Sout[i*d+j] = Se[i*d+j] + write_g*beta*tgt*k[j];
    }
    free(delta); free(Se);
    return 1;
}

/* T04 ternary STE: forward ternary (sign clip to {-1,0,1} by thr), and report
 * whether gradient passes (|x| <= 1). out_tern holds the ternary value. */
int wubu_ternary_ste(float x, float thr, float *out_tern, int *grad_passes) {
    if (!out_tern || !grad_passes) return 0;
    if (thr < 0.0f) thr = 0.0f;
    float s = (x > thr) ? 1.0f : (x < -thr) ? -1.0f : 0.0f;
    *out_tern = s;
    *grad_passes = (x >= -1.0f && x <= 1.0f) ? 1 : 0;
    return 1;
}

/* wubu_traj_grpo.c -- the multi-turn trajectory-level GRPO (the Orchard
 * recipe core). Group-relative advantage over the G trajectories:
 *   A_g = (r_g - mean(r)) / (std(r) + eps)
 * broadcast to EVERY assistant token of trajectory g; the observation /
 * context tokens are masked out of the loss; the normalization is over
 * the masked tokens only (the Orchard "no 1/T" doctrine: longer, harder
 * tasks are not down-weighted). The loss is the advantage-weighted NLL
 * with the optional asymmetric PPO ratio clipping. */
#include <math.h>
#include <stddef.h>
#include "wubu_traj_grpo.h"

int wubu_traj_grpo(const float *logp, const float *mask, const float *r,
                   int G, int T, float clip_lo, float clip_hi,
                   const float *old_logp, float eps,
                   float *loss, float *grad)
{
    if (!logp || !mask || !r || G < 1 || T < 1 || clip_lo < 0 || clip_hi < 0)
        return 0;
    /* the group-relative advantages */
    float mean = 0;
    for (int g = 0; g < G; g++) mean += r[g];
    mean /= (float)G;
    float var = 0;
    for (int g = 0; g < G; g++) { float d = r[g] - mean; var += d * d; }
    var /= (float)G;
    float std = sqrtf(var) + (eps > 0 ? eps : 1e-6f);
    float A[64];
    if (G > 64) G = 64;
    for (int g = 0; g < G; g++) A[g] = (r[g] - mean) / std;

    double L = 0;
    for (int g = 0; g < G; g++) {
        const float *lp = logp + (size_t)g * T;
        const float *ms = mask + (size_t)g * T;
        const float *op = old_logp ? old_logp + (size_t)g * T : NULL;
        float n = 0;      /* the masked token count (the normalization) */
        double sum = 0;   /* the masked log-prob sum */
        for (int t = 0; t < T; t++) {
            if (ms[t] <= 0) continue;
            n += 1;
            double l = lp[t];
            if (op) {
                /* the ratio variant: r_t = exp(lp - old), clipped at
                 * [1-clip_lo, 1+clip_hi] (asymmetric, Orchard) */
                double ratio = expf(lp[t] - op[t]);
                double lo = 1.0 - clip_lo, hi = 1.0 + clip_hi;
                double c = ratio < lo ? lo : (ratio > hi ? hi : ratio);
                sum += (A[g] >= 0) ? (ratio < c ? ratio : c) * l
                                   : (ratio > c ? ratio : c) * l;
            } else {
                sum += l;
            }
        }
        if (n > 0) L += (double)A[g] * sum / n;
    }
    L /= (double)G;
    if (loss) *loss = (float)(-L);

    if (grad) {
        for (int g = 0; g < G; g++) {
            const float *ms = mask + (size_t)g * T;
            const float *op = old_logp ? old_logp + (size_t)g * T : NULL;
            float n = 0;
            for (int t = 0; t < T; t++) if (ms[t] > 0) n += 1;
            float w = -A[g] / (n > 0 ? n * (float)G : 1.0f);
            for (int t = 0; t < T; t++) {
                if (ms[t] <= 0) { grad[(size_t)g * T + t] = 0; continue; }
                if (!op) {
                    grad[(size_t)g * T + t] = w;
                } else {
                    /* d/dlp of the clipped term: inside the clip range the
                     * term is ratio*lp (d = ratio*(1+lp)); outside it is
                     * c*lp (d = c). The FD is the oracle here. */
                    double ratio = expf(logp[(size_t)g * T + t] - op[t]);
                    double lo = 1.0 - clip_lo, hi = 1.0 + clip_hi;
                    double c = ratio < lo ? lo : (ratio > hi ? hi : ratio);
                    /* active = the ratio is INSIDE the range (the min/max
                     * picks the ratio, not the constant clip) -- comparing
                     * against c itself is wrong (c == ratio inside) */
                    int active = (A[g] >= 0) ? (ratio < hi) : (ratio > lo);
                    double d = active ? ratio * (1.0 + logp[(size_t)g * T + t])
                                      : c;
                    grad[(size_t)g * T + t] = (float)(w * d);
                }
            }
        }
    }
    return 1;
}

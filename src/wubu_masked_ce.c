/* wubu_masked_ce.c -- the masked next-token CE. */
#include <math.h>
#include <string.h>
#include "wubu_masked_ce.h"

int wubu_masked_ce(const float *logits, const uint16_t *tokens,
                   const float *mask, int seq, int vocab,
                   float *loss, float *grad)
{
    if (!logits || !tokens || !mask || seq < 1 || vocab < 1) return 0;
    /* the masked count (the normalization -- the masked-mean doctrine) */
    int n = 0;
    for (int s = 0; s < seq; s++) if (mask[s] > 0) n++;
    if (n == 0) { if (loss) *loss = 0; if (grad) memset(grad, 0, (size_t)seq * vocab * sizeof(float)); return 1; }
    /* the masked-mean CE */
    double L = 0;
    for (int s = 0; s < seq; s++) {
        if (mask[s] <= 0) continue;
        const float *row = logits + (size_t)s * vocab;
        int t = tokens[s];
        if (t < 0 || t >= vocab) return 0;
        /* the log-sum-exp + the target's logit (stable) */
        float m = row[0];
        for (int v = 1; v < vocab; v++) if (row[v] > m) m = row[v];
        double lse = 0;
        for (int v = 0; v < vocab; v++) lse += expf((double)row[v] - m);
        L += (double)(m + logf((float)lse)) - row[t];
    }
    L /= (double)n;
    if (loss) *loss = (float)L;
    if (grad) {
        float w = 1.0f / (float)n;
        for (int s = 0; s < seq; s++) {
            float *g = grad + (size_t)s * vocab;
            const float *row = logits + (size_t)s * vocab;
            if (mask[s] <= 0) {
                memset(g, 0, (size_t)vocab * sizeof(float));
                continue;
            }
            float m = row[0];
            for (int v = 1; v < vocab; v++) if (row[v] > m) m = row[v];
            double lse = 0;
            for (int v = 0; v < vocab; v++) lse += expf((double)row[v] - m);
            for (int v = 0; v < vocab; v++) {
                double p = expf((double)row[v] - m) / lse;
                g[v] = (float)(w * (p - (v == tokens[s] ? 1.0 : 0.0)));
            }
        }
    }
    return 1;
}

float wubu_masked_ce_frac(const float *mask, int seq)
{
    if (!mask || seq < 1) return 0;
    int n = 0;
    for (int s = 0; s < seq; s++) if (mask[s] > 0) n++;
    return (float)n / (float)seq;
}

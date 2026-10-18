/*
 * wubu_decision.c -- typed decision node arithmetic (C11, no deps).
 *
 * See include/wubu_decision.h for the contract and the two published
 * coherence failures this module refuses to reproduce.
 */
#include "wubu_decision.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

int wubu_decision_softmax(const float *logits, int n, float *probs)
{
    if (!logits || !probs || n <= 0) return -1;

    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = exp((double)(logits[i] - mx));
        probs[i] = (float)e;
        sum += e;
    }
    if (!(sum > 0.0) || !isfinite(sum)) return -1;

    double inv = 1.0 / sum;
    for (int i = 0; i < n; i++) probs[i] = (float)((double)probs[i] * inv);

    /* fp32 rounding: force the exact sum to 1 by adjusting the largest entry.
     * The residual is a few ulps; leaving it out is how "sum = 1.0000001"
     * claims start. */
    double s2 = 0.0;
    int big = 0;
    for (int i = 0; i < n; i++) {
        s2 += (double)probs[i];
        if (probs[i] > probs[big]) big = i;
    }
    probs[big] = (float)((double)probs[big] + (1.0 - s2));
    return 0;
}

float wubu_decision_entropy(const float *probs, int n)
{
    if (!probs || n <= 0) return -1.0f;
    double h = 0.0;
    for (int i = 0; i < n; i++) {
        double p = (double)probs[i];
        if (p > 0.0) h -= p * log(p);
    }
    return (float)h;
}

int wubu_decision_choose(const float *probs, int n, float threshold,
                         int *idx, float *conf)
{
    if (!probs || n <= 0) return 0;
    int best = 0;
    for (int i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
    if (idx) *idx = best;
    if (conf) *conf = probs[best];
    return probs[best] >= threshold ? 1 : 0;
}

int wubu_decision_complement_ok(float p, float p_not, float tol)
{
    if (!isfinite(p) || !isfinite(p_not) || tol < 0.0f) return 0;
    float d = fabsf(p + p_not - 1.0f);
    return d <= tol ? 1 : 0;
}

int wubu_decision_exclusive_ok(const float *probs, int n, float tol)
{
    if (!probs || n <= 0 || tol < 0.0f) return 0;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(probs[i]) || probs[i] < 0.0f) return 0;
        s += (double)probs[i];
    }
    return fabs(s - 1.0) <= (double)tol ? 1 : 0;
}

float wubu_decision_ece(const float *conf, const unsigned char *correct,
                        int n, int bins)
{
    if (!conf || !correct || n <= 0 || bins <= 0) return -1.0f;

    double *sum_conf = (double *)calloc((size_t)bins, sizeof(double));
    long *count = (long *)calloc((size_t)bins, sizeof(long));
    long *hits = (long *)calloc((size_t)bins, sizeof(long));
    if (!sum_conf || !count || !hits) {
        free(sum_conf); free(count); free(hits);
        return -1.0f;
    }

    for (int i = 0; i < n; i++) {
        double c = (double)conf[i];
        if (c < 0.0) c = 0.0;
        if (c > 1.0) c = 1.0;
        int b = (int)(c * bins);
        if (b >= bins) b = bins - 1;
        sum_conf[b] += c;
        count[b] += 1;
        hits[b] += correct[i] ? 1 : 0;
    }

    double ece = 0.0;
    for (int b = 0; b < bins; b++) {
        if (count[b] == 0) continue;
        double avg_conf = sum_conf[b] / (double)count[b];
        double acc = (double)hits[b] / (double)count[b];
        ece += ((double)count[b] / (double)n) * fabs(avg_conf - acc);
    }

    free(sum_conf); free(count); free(hits);
    return (float)ece;
}
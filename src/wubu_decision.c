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
#include <string.h>

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
/* ================================================================
 * Delegation economics (JevAgentHarness, arXiv 2610.21412)
 * ================================================================ */

static int cmp_sample_desc(const void *a, const void *b)
{
    const wubu_decision_sample *x = (const wubu_decision_sample *)a;
    const wubu_decision_sample *y = (const wubu_decision_sample *)b;
    if (x->conf < y->conf) return 1;
    if (x->conf > y->conf) return -1;
    return 0;
}

double wubu_decision_analytic_threshold(double llm_acc, double saved_cost,
                                        double wrong_cost)
{
    if (llm_acc < 0.0 || llm_acc > 1.0) return -1.0;
    if (saved_cost < 0.0 || wrong_cost <= 0.0) return -1.0;
    double t = llm_acc - saved_cost / wrong_cost;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return t;
}

double wubu_decision_fit_threshold(const wubu_decision_sample *s, int n,
                                   double llm_acc, double cost_probe,
                                   double cost_llm, double cost_wrong,
                                   double budget)
{
    if (!s || n <= 0) return -1.0;
    if (llm_acc < 0.0 || llm_acc > 1.0 || budget < 0.0) return -1.0;
    if (cost_probe < 0.0 || cost_llm < 0.0 || cost_wrong < 0.0) return -1.0;

    wubu_decision_sample *copy = (wubu_decision_sample *)malloc((size_t)n * sizeof *copy);
    if (!copy) return -1.0;
    memcpy(copy, s, (size_t)n * sizeof *copy);
    qsort(copy, (size_t)n, sizeof *copy, cmp_sample_desc);

    /* k = 0: probe every state, let the LLM decide everything. */
    double best_cost = (double)n * cost_probe + (double)n * cost_llm;
    double best_tau = 2.0;   /* 2.0 == delegate nothing */
    int acc_correct = 0;

    for (int k = 1; k <= n; k++) {
        acc_correct += copy[k - 1].correct ? 1 : 0;
        double success = ((double)acc_correct + llm_acc * (double)(n - k)) / (double)n;
        if (success < llm_acc - budget) continue;

        double cost = (double)n * cost_probe
                    + (double)(n - k) * cost_llm
                    + (double)(k - acc_correct) * cost_wrong;
        if (cost < best_cost - 1e-12) {
            best_cost = cost;
            best_tau = (double)copy[k - 1].conf;
        }
    }

    free(copy);
    return best_tau;
}

void wubu_decision_ledger_init(wubu_decision_ledger *l)
{
    if (!l) return;
    memset(l, 0, sizeof *l);
}

void wubu_decision_ledger_record(wubu_decision_ledger *l, int delegated,
                                 int correct, double dm_cost, double dm_time,
                                 double llm_cost, double llm_time,
                                 double wrong_cost)
{
    if (!l) return;
    if (delegated) {
        l->delegated++;
        l->cost_dm += dm_cost;
        l->time_dm_s += dm_time;
        l->cost_llm_avoided += llm_cost;
        l->time_avoided_s += llm_time;
        if (!correct) {
            l->wrong_delegated++;
            l->cost_penalty += wrong_cost;
        }
    } else {
        l->deferred++;
        l->cost_llm_spent += llm_cost;
        l->time_llm_s += llm_time;
    }
}

double wubu_decision_ledger_cost(const wubu_decision_ledger *l)
{
    if (!l) return -1.0;
    return l->cost_dm + l->cost_llm_spent + l->cost_penalty;
}

double wubu_decision_ledger_saved_pct(const wubu_decision_ledger *l,
                                      double baseline_cost)
{
    if (!l || baseline_cost <= 0.0) return 0.0;
    double saved = baseline_cost - wubu_decision_ledger_cost(l);
    return 100.0 * saved / baseline_cost;
}

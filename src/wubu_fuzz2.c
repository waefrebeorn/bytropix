/*
 * wubu_fuzz2.c -- the robustness frontier, complete (IX). C11.
 */
#include "wubu_fuzz2.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

float wubu_fz2_tradeoff(float robustness, float quality, float w)
{
    return w * robustness + (1.0f - w) * quality;
}

int wubu_fz2_archive(const char *prompt, char **archive, int n, int cap)
{
    if (!prompt || !archive || n < 0) return -1;
    if (n >= cap) return -1;
    archive[n] = (char *)prompt;   /* the caller owns the strings */
    return n + 1;
}

int wubu_fz2_heal(int stalls, int max_stalls)
{
    return stalls >= max_stalls ? 1 : 0;
}

float wubu_fz2_signal(int evaded, int tested)
{
    if (tested <= 0) return 0;
    return (float)(tested - evaded) / (float)tested;
}

int wubu_fz2_schema(const char *in, int depth, int max_depth)
{
    if (!in) return -1;
    return depth <= max_depth ? 1 : 0;
}

int wubu_fz2_depth(const int *layer_hits, int n, int th)
{
    if (!layer_hits || n <= 0) return -1;
    int hits = 0;
    for (int i = 0; i < n; i++) hits += layer_hits[i];
    return hits >= th ? 1 : 0;
}

int wubu_fz2_delta(float cur, float prev, float th)
{
    return (cur - prev) < -th ? 1 : 0;   /* the drop is the regression */
}

float wubu_fz2_coverage(long covered, long total)
{
    if (total <= 0) return 0;
    return (float)covered / (float)total;
}

float wubu_fz2_distill(float adv_loss, float clean_loss)
{
    return adv_loss / (clean_loss + 1e-9f);
}

float wubu_fz2_fp(long false_pos, long total)
{
    if (total <= 0) return 0;
    return (float)false_pos / (float)total;
}

int wubu_fz2_leak(const char *out, const char *secret)
{
    if (!out || !secret) return 0;
    return strstr(out, secret) != NULL ? 1 : 0;
}

int wubu_fz2_energy(long evals, float j_per_eval, float budget)
{
    if (budget <= 0) return 0;
    return (float)evals * j_per_eval <= budget ? 1 : 0;
}

int wubu_fz2_canon(const char *in, char *out, int cap)
{
    if (!in || !out || cap <= 0) return -1;
    int k = 0;
    for (int i = 0; in[i] && k < cap - 1; i++) {
        char c = in[i];
        if (c == '\r') continue;                 /* strip */
        if (c == '\t') c = ' ';
        if (c == ' ' && (k == 0 || out[k - 1] == ' ')) continue; /* collapse */
        out[k++] = (char)tolower((unsigned char)c);
    }
    out[k] = 0;
    return k;
}

int wubu_fz2_diff(const char *a, const char *b, float th)
{
    if (!a || !b) return -1;
    int la = (int)strlen(a), lb = (int)strlen(b);
    int same = 0, n = la < lb ? la : lb;
    while (same < n && a[same] == b[same]) same++;
    float denom = la > lb ? la : lb;
    return ((float)same / (denom > 0 ? denom : 1.0f)) >= th ? 1 : 0;
}

int wubu_fz2_repair(float weakness, float th)
{
    return weakness >= th ? 1 : 0;
}

float wubu_fz2_harness(long evaded, long total)
{
    if (total <= 0) return 1.0f;
    return 1.0f - (float)evaded / (float)total;
}

int wubu_fz2_anomaly(const uint32_t *ids, int n, float mean_len, float dev)
{
    if (!ids || n <= 0) return 0;
    float len = 0;
    for (int i = 0; i < n; i++) if (ids[i] > mean_len + dev) len += 1;
    return (len / n) > 0.3f ? 1 : 0;
}

int wubu_fz2_redundant(const int *layers, int n, int th)
{
    if (!layers || n <= 0) return -1;
    int active = 0;
    for (int i = 0; i < n; i++) active += layers[i];
    return active >= th ? 1 : 0;
}

int wubu_fz2_degraded(int core_defense, int optional_defense)
{
    return core_defense && !optional_defense ? 1 : 0;  /* degraded-but-safe */
}

int wubu_fz2_ci(float evasion, float th)
{
    return evasion <= th ? 1 : 0;
}

int wubu_fz2_gen(const char *seed, char *out, int cap, uint32_t variant)
{
    if (!seed || !out || cap <= 0) return -1;
    int n = (int)strlen(seed);
    if (n >= cap - 2) n = cap - 3;
    memcpy(out, seed, (size_t)n);
    /* append a variant tag (the auto-generated attack family) */
    out[n++] = '-';
    out[n++] = (char)('0' + (variant % 10));
    out[n] = 0;
    return n;
}

int wubu_fz2_attrib(const float *layer_scores, int n)
{
    if (!layer_scores || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (layer_scores[i] < layer_scores[best]) best = i;
    return best;   /* the weakest layer = the failure point */
}

int wubu_fz2_workers(int tasks, int cores)
{
    if (tasks <= 0 || cores <= 0) return 0;
    return tasks < cores ? tasks : cores;
}

int wubu_fz2_sla(float score, float bar)
{
    return score >= bar ? 1 : 0;
}

int wubu_fz2_verifier(int fuzz_found, int verified)
{
    return fuzz_found && verified ? 1 : 0;
}

int wubu_fz2_debt(const float *weaknesses, int n, float th, int *count)
{
    if (!weaknesses || !count || n <= 0) return -1;
    *count = 0;
    for (int i = 0; i < n; i++)
        if (weaknesses[i] >= th) (*count)++;
    return 0;
}

int wubu_fz2_entropy_guard(const uint32_t *counts, int n, float th)
{
    if (!counts || n <= 0) return 0;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) total += counts[i];
    if (total == 0) return 0;
    double h = 0;
    for (int i = 0; i < n; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / (double)total;
        h -= p * log2(p);
    }
    return h > th ? 1 : 0;
}

float wubu_fz2_transfer(float src_evasion, float dst_evasion)
{
    return dst_evasion / (src_evasion + 1e-9f);
}

float wubu_fz2_def_sampling(float logit, float defense_confidence)
{
    /* high defense confidence -> pull the logit toward the safe path */
    return logit * (1.0f - 0.3f * defense_confidence);
}

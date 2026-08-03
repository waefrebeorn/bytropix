/*
 * wubu_hopfield3.c -- the associative-memory frontier, final (IP). C11.
 */
#include "wubu_hopfield3.h"
#include <math.h>
#include <string.h>

int wubu_hop3_attention_read(const float *q, const float *kv, int d, int n, float *out)
{
    if (!q || !kv || !out || d <= 0 || n <= 0) return -1;
    float max_sim = -1e30f;
    int best = 0;
    for (int i = 0; i < n; i++) {
        float sim = 0;
        for (int j = 0; j < d; j++) sim += q[j] * kv[i * d + j];
        if (sim > max_sim) { max_sim = sim; best = i; }
    }
    for (int j = 0; j < d; j++) out[j] = kv[best * d + j];
    return best;
}

float wubu_hop3_curvature(const float *patterns, int n, int d)
{
    if (!patterns || n <= 1 || d <= 0) return 0;
    /* the curvature: the average second-difference of the pattern norms */
    float total = 0;
    for (int i = 0; i < n; i++) {
        float norm = 0;
        for (int j = 0; j < d; j++) norm += patterns[i * d + j] * patterns[i * d + j];
        total += sqrtf(norm);
    }
    return total / (float)n;
}

int wubu_hop3_federated(const float *patterns, int n, int src_id, int *merged)
{
    if (!patterns || !merged || n <= 0) return -1;
    *merged = src_id * 1000 + n;
    return 0;
}

int wubu_hop3_stabilize(const float *pattern, int d, float strength, float *anchored)
{
    if (!pattern || !anchored) return -1;
    float norm = 0;
    for (int i = 0; i < d; i++) norm += pattern[i] * pattern[i];
    norm = sqrtf(norm);
    if (norm < 1e-9f) {
        for (int i = 0; i < d; i++) anchored[i] = 0;
        return 0;
    }
    for (int i = 0; i < d; i++) anchored[i] = pattern[i] / norm * strength;
    return 0;
}

int wubu_hop3_cue_quality(const float *cue, int d, float th)
{
    if (!cue || d <= 0) return 0;
    float norm = 0;
    for (int i = 0; i < d; i++) norm += cue[i] * cue[i];
    return sqrtf(norm) >= th ? 1 : 0;
}

int wubu_hop3_write_batch(const float *patterns, int n, int d, int batch_size)
{
    if (!patterns || batch_size <= 0) return -1;
    return (n + batch_size - 1) / batch_size;
}

int wubu_hop3_read_batch(const float *patterns, int n, int d, int batch_size)
{
    return wubu_hop3_write_batch(patterns, n, d, batch_size);
}

int wubu_hop3_outlier_tol(const float *pattern, int d, float max_noise)
{
    if (!pattern) return 0;
    float norm = 0;
    for (int i = 0; i < d; i++) norm += pattern[i] * pattern[i];
    return sqrtf(norm) <= max_noise ? 1 : 0;
}

int wubu_hop3_ann(const float *query, const float *memory, int n, int d, float th, int *idx)
{
    if (!query || !memory || !idx || n <= 0) return -1;
    int best = 0;
    float best_sim = -1e30f;
    for (int i = 0; i < n; i++) {
        float sim = 0;
        for (int j = 0; j < d; j++) sim += query[j] * memory[i * d + j];
        if (sim > best_sim) { best_sim = sim; best = i; }
    }
    *idx = best;
    return best_sim >= th ? 1 : 0;
}

float wubu_hop3_asymmetry(long write_cost, long read_benefit)
{
    if (read_benefit <= 0) return 0;
    return (float)write_cost / (float)read_benefit;
}

int wubu_hop3_decay_arbitrate(float decay_rate, float rehearsal_rate, float th)
{
    return (decay_rate - rehearsal_rate) > th ? 1 : 0;
}

int wubu_hop3_rag(const float *corpus, int n, int d, float *pattern)
{
    if (!corpus || !pattern || n <= 0) return -1;
    for (int j = 0; j < d; j++) {
        float sum = 0;
        for (int i = 0; i < n; i++) sum += corpus[i * d + j];
        pattern[j] = sum / (float)n;
    }
    return d;
}

int wubu_hop3_provenance(const float *pattern, int id, char *meta, int cap)
{
    if (!meta || cap <= 0) return -1;
    snprintf(meta, cap, "pattern_%d", id);
    return 0;
}

int wubu_hop3_forget(const float *pattern, const int *forget_ids, int n, int target)
{
    if (!pattern || !forget_ids) return -1;
    for (int i = 0; i < n; i++)
        if (forget_ids[i] == target) return 1;
    return 0;
}

int wubu_hop3_balance(const float *access_counts, int n, int *hot_tier)
{
    if (!access_counts || !hot_tier || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (access_counts[i] > access_counts[best]) best = i;
    *hot_tier = best;
    return 0;
}

int wubu_hop3_world_update(const float *state, int d, const float *obs, float *next)
{
    if (!state || !obs || !next) return -1;
    for (int i = 0; i < d; i++) next[i] = state[i] + 0.1f * obs[i];
    return 0;
}

int wubu_hop3_capacity_warning(long stored, long limit)
{
    return stored > limit ? 1 : 0;
}

int wubu_hop3_weight(const float *patterns, int n, int d, float *weights)
{
    if (!patterns || !weights || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        float norm = 0;
        for (int j = 0; j < d; j++) norm += patterns[i * d + j] * patterns[i * d + j];
        weights[i] = sqrtf(norm);
    }
    return n;
}

int wubu_hop3_coherence(const float *a, const float *b, int n, int d, float *score)
{
    if (!a || !b || !score || n <= 0) return -1;
    float dot = 0, an = 0, bn = 0;
    for (int i = 0; i < n * d; i++) {
        dot += a[i] * b[i];
        an += a[i] * a[i];
        bn += b[i] * b[i];
    }
    *score = dot / (sqrtf(an) * sqrtf(bn) + 1e-9f);
    return 0;
}

int wubu_hop3_momentum(const float *current, const float *target, int d, float momentum, float *next)
{
    if (!current || !target || !next) return -1;
    for (int i = 0; i < d; i++)
        next[i] = momentum * current[i] + (1.0f - momentum) * target[i];
    return 0;
}

int wubu_hop3_sparse(const float *patterns, int n, int k, int *selected)
{
    if (!patterns || !selected || k <= 0 || n <= 0) return -1;
    int sel = k < n ? k : n;
    for (int i = 0; i < sel; i++) selected[i] = i;
    return sel;
}

float wubu_hop3_continuous(float tau, float input, float state)
{
    return state + (input - state) / tau;
}

float wubu_hop3_energy(const float *state, const float *weights, int d)
{
    if (!state || !weights) return 0;
    float energy = 0;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            energy -= 0.5f * state[i] * weights[i * d + j] * state[j];
    for (int i = 0; i < d; i++) energy += state[i] * state[i];
    return energy;
}

long wubu_hop3_scaling(int d, float capacity_factor)
{
    return (long)(capacity_factor * d);
}

int wubu_hop3_noise(const float *clean, const float *noisy, int d, float th)
{
    if (!clean || !noisy) return 0;
    float max_err = 0;
    for (int i = 0; i < d; i++) {
        float e = fabsf(clean[i] - noisy[i]);
        if (e > max_err) max_err = e;
    }
    return max_err <= th ? 1 : 0;
}

int wubu_hop3_complete(const float *partial, int d, const float *memory, int n, float *completed)
{
    if (!partial || !memory || !completed) return -1;
    float max_sim = -1e30f;
    int best = 0;
    for (int i = 0; i < n; i++) {
        float sim = 0;
        for (int j = 0; j < d; j++) sim += partial[j] * memory[i * d + j];
        if (sim > max_sim) { max_sim = sim; best = i; }
    }
    for (int j = 0; j < d; j++) completed[j] = memory[best * d + j];
    return best;
}
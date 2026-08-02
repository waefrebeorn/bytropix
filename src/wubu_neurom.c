/*
 * wubu_neurom.c -- neuromorphic / SNN frontier (Theme IW). C11.
 */
#include "wubu_neurom.h"
#include <math.h>
#include <string.h>

int wubu_neurom_encode(float value, float rate_max, float dt, int n_bins,
                       uint8_t *spikes)
{
    if (!spikes || n_bins <= 0 || value < 0) return -1;
    float rate = value * rate_max;          /* spikes per second */
    float p = rate * dt;
    int n = 0;
    for (int i = 0; i < n_bins; i++) {
        float r = ((float)(i * 2654435761u % 1000)) / 1000.0f;
        spikes[i] = (r < p) ? 1 : 0;
        n += spikes[i];
    }
    return n;
}

int wubu_neurom_lif(float *membrane, float input, float leak, float th,
                    float *spike)
{
    if (!membrane || !spike) return -1;
    *membrane = *membrane * (1.0f - leak) + input;
    if (*membrane >= th) {
        *membrane = 0;
        *spike = 1.0f;
        return 1;
    }
    *spike = 0;
    return 0;
}

float wubu_neurom_energy(long spikes, float pj_per_spike)
{
    return (float)spikes * pj_per_spike;
}

float wubu_neurom_gate(float spike_rate, float th)
{
    return spike_rate >= th ? 1.0f : spike_rate / (th > 0 ? th : 1.0f);
}

float wubu_neurom_sparsity(float active_fraction)
{
    /* the fraction of compute skipped by the spike sparsity */
    return 1.0f - active_fraction;
}

int wubu_neurom_schedule(long *spike_counts, int n_cores, int *core, int n)
{
    if (!spike_counts || !core || n_cores <= 0 || n <= 0) return -1;
    long loads[64] = { 0 };
    if (n_cores > 64) n_cores = 64;
    for (int i = 0; i < n; i++) {
        int best = 0;
        for (int c = 1; c < n_cores; c++)
            if (loads[c] < loads[best]) best = c;
        core[i] = best;
        loads[best] += spike_counts[i];
    }
    return 0;
}

int wubu_neurom_spike_attn(const uint8_t *spikes, int n, int d,
                           const float *w, float *out)
{
    if (!spikes || !w || !out || d <= 0) return -1;
    for (int i = 0; i < d; i++) out[i] = 0;
    float mass = 0;
    for (int t = 0; t < n; t++) {
        if (!spikes[t]) continue;
        for (int i = 0; i < d; i++) out[i] += w[t * d + i];
        mass += 1.0f;
    }
    if (mass > 0) for (int i = 0; i < d; i++) out[i] /= mass;
    return 0;
}

int wubu_neurom_kv(const float *k, const float *v, float *synapse, int d)
{
    if (!k || !v || !synapse) return -1;
    /* Hebbian-ish write: the synapse += k * v (the KV as weights) */
    for (int i = 0; i < d; i++) synapse[i] += k[i] * v[i];
    return 0;
}

float wubu_neurom_temporal(float value, float t_max)
{
    /* first-spike latency: higher value -> earlier spike */
    if (value < 0) value = 0;
    if (value > 1) value = 1;
    return t_max * (1.0f - value);
}

float wubu_neurom_convert(float ann_activation, float scale)
{
    /* rate-matching: the ANN activation becomes a spike rate */
    return ann_activation * scale;
}

float wubu_neurom_energy_saved(float sparsity)
{
    if (sparsity < 0) return 0;
    if (sparsity > 1) sparsity = 1;
    return sparsity;   /* 1 - active = the saved fraction */
}

int wubu_neurom_event_select(const uint8_t *spikes, int n, int th, int *keep,
                             int cap)
{
    if (!spikes || !keep || cap <= 0) return -1;
    int k = 0;
    for (int t = 0; t < n && k < cap; t++)
        if (spikes[t] >= th) keep[k++] = t;
    return k;
}

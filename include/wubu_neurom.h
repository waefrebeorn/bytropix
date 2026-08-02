/*
 * wubu_neurom.h -- neuromorphic / SNN frontier (Theme IW). C11.
 * Spike encoding, event-driven decode, energy models, brain-inspired
 * gating, spike sparsity, multi-core scheduling, LIF accumulation,
 * spike attention, neuromorphic KV, temporal coding, ANN->SNN,
 * energy-sparsity correlation, event-driven token selection.
 */
#ifndef WUBU_NEUROM_H
#define WUBU_NEUROM_H

#include <stdint.h>

/* IW01: token -> spike train (rate coding). */
int wubu_neurom_encode(float value, float rate_max, float dt, int n_bins,
                       uint8_t *spikes);

/* IW07: leaky integrate-and-fire membrane. */
int wubu_neurom_lif(float *membrane, float input, float leak, float th,
                    float *spike);

/* IW03: SNN energy model (J per spike vs the GPU baseline). */
float wubu_neurom_energy(long spikes, float pj_per_spike);

/* IW04: brain-inspired gating -- the gate opens on spike evidence. */
float wubu_neurom_gate(float spike_rate, float th);

/* IW05: spike-sparsity compute cut (the fraction of work skipped). */
float wubu_neurom_sparsity(float active_fraction);

/* IW06: multi-core spike scheduling -- the parallel core assignment. */
int wubu_neurom_schedule(long *spike_counts, int n_cores, int *core, int n);

/* IW08: spike-based attention -- the attention over spike events. */
int wubu_neurom_spike_attn(const uint8_t *spikes, int n, int d,
                           const float *w, float *out);

/* IW09: neuromorphic KV -- the KV as synaptic weights. */
int wubu_neurom_kv(const float *k, const float *v, float *synapse, int d);

/* IW10: spike-timing encoding (the first-spike latency codes the value). */
float wubu_neurom_temporal(float value, float t_max);

/* IW11: ANN-to-SNN conversion (the rate-matching scale). */
float wubu_neurom_convert(float ann_activation, float scale);

/* IW12: energy saved per sparsity level. */
float wubu_neurom_energy_saved(float sparsity);

/* IW13: event-driven token selection (spikes gate the processing). */
int wubu_neurom_event_select(const uint8_t *spikes, int n, int th, int *keep,
                             int cap);

#endif

/*
 * wubu_deltanet.h -- the Gated-DeltaNet linear mixer (research 008).
 *
 * The sparse-attention lineage (Qwen3-Next / Gated-DeltaNet / the
 * wizard's research 008): replace most attention layers with a linear
 * (recurrent) mixer that keeps a FIXED-SIZE state S updated by a delta
 * rule -- linear in context, not quadratic. The KV cache for these
 * layers is just the small state (head_dim² per head) -- essentially
 * free vs the quadratic KV of full attention.
 *
 * The delta rule (from research 008, DA-verified):
 *     S = α·S + k·(v − S·k)·β
 * where α is the decay gate (sigmoid), k the key, v the value, β the
 * scale. The out-gate is RMSNorm(SiLU(gate)) -- stabilizes, removes
 * attention sinks (ties to research 011).
 *
 * This is the phase-5 hook for the WuBu model: the hybrid rhythm
 * becomes 3 deltanet : 1 full (the reference's 3:1), so 2048-token
 * contexts cost ~linear time and the KV cache shrinks to state-sized.
 */
#ifndef WUBU_DELTANET_H
#define WUBU_DELTANET_H

#include <stdint.h>

/* the fixed state: S is [head_dim, head_dim] per head. */
typedef struct {
    float *S;             /* [n_heads, head_dim, head_dim] the delta state */
    int    n_heads;
    int    head_dim;
} wubu_deltanet_state_t;

/* D1: allocate the state (all zero -- the reference starts at 0). */
int wubu_deltanet_state_init(wubu_deltanet_state_t *st, int n_heads,
                             int head_dim);

/* D2: one step (decode): S = α·S + k·(v − S·k)·β, returns the output
 * o = S·k^T + the gated out. k/v are [head_dim] for ONE head. */
void wubu_deltanet_step(wubu_deltanet_state_t *st, int head,
                        const float *k, const float *v,
                        float alpha, float beta, float *out);

/* D2b: a PURE read -- o = S·k^T with NO state update. The DA caught
 * that step() trains AND reads, making recall untestable (a "recall"
 * call retrained with the value). This is the query-time path. */
void wubu_deltanet_read(wubu_deltanet_state_t *st, int head,
                        const float *k, float *out);

/* D3: prefill a chunk of tokens [T, head_dim] per head: runs the
 * recurrence in order (chunkwise is the next milestone; the sequential
 * scan is exact). Returns 0 on success. */
int wubu_deltanet_prefill(wubu_deltanet_state_t *st,
                          const float *K,   /* [T, n_heads, head_dim] */
                          const float *V,   /* [T, n_heads, head_dim] */
                          int T, float alpha, float beta,
                          float *outs);     /* [T, n_heads, head_dim] */

/* D4: reset the state (new sequence). */
void wubu_deltanet_state_reset(wubu_deltanet_state_t *st);

/* D5: free. */
void wubu_deltanet_state_free(wubu_deltanet_state_t *st);

#endif

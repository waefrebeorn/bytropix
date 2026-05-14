#ifndef WUBU_TST_H
#define WUBU_TST_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// TST: Token Superposition Training
// ============================================================
//
// Two-phase training protocol:
//   1. Superposition Phase (ratio r of steps):
//      - Bag s=8 contiguous tokens -> average embeddings
//      - Forward pass on T/s tokens
//      - MCE loss: (1/s) * sum CE(pred, target[i]) for i in [0..s)
//   2. Recovery Phase (ratio 1-r of steps):
//      - Standard CE next-token prediction
//      - Weights carry over from superposition phase
// ============================================================

// TST hyperparameters
typedef struct {
    int   bag_size;          // s — number of tokens to bag (default 8)
    float superp_ratio;      // r — fraction of steps in superposition phase (default 0.25)
} tst_config;

// Default TST config
static const tst_config TST_CONFIG_DEFAULT = {
    .bag_size     = 8,
    .superp_ratio = 0.25f
};

// -----------------------------------------------------------
// Superposition Phase: Embedding Bagging
// -----------------------------------------------------------
// Given input embeddings [B, T, D] and bag_size s,
// average every s contiguous embeddings into one.
//
// embeddings:  [B, T, D] input embeddings
// bagged:      [B, T/s, D] output (caller-allocated, T must be multiple of s)
// B, T, D:     batch, sequence, hidden dimensions
// s:           bag size
void tst_bag_embeddings(const float *embeddings, float *bagged,
                        int B, int T, int D, int s);

// -----------------------------------------------------------
// Superposition Phase: Target Preparation
// -----------------------------------------------------------
// Shift token IDs left by s-1, then split into non-overlapping
// bags of size s.  Returns the number of complete target bags.
//
// token_ids: [B, T] input token IDs
// targets:   [B, T_out, s] output targets (caller-allocated)
//            T_out = (T - s + 1) / s
// B, T:      batch, sequence dimensions
// s:         bag size
// Returns:   T_out (number of target bags).  Returns 0 if
//            (T - s + 1) < s (no complete target bag).
int tst_prepare_targets(const int *token_ids, int *targets,
                        int B, int T, int s);

// -----------------------------------------------------------
// Superposition Phase: MCE Loss (Mean Cross-Entropy)
// -----------------------------------------------------------
// Compute mean cross-entropy loss for superposition training.
// For each bag position i, compute CE against s targets and
// average: loss = (1/s) * sum_{k=0}^{s-1} CE(logits[i], targets[i,k])
//
// logits:    [B, T_out, V] predicted logits (one per bag)
// targets:   [B, T_out, s] ground-truth token IDs (from tst_prepare_targets)
// B, T_out:  batch and bagged sequence dimensions
// V:         vocabulary size
// s:         bag size
// loss:      scalar output loss value
// Returns:   false on invalid input
bool tst_compute_mce_loss(const float *logits, const int *targets,
                          int B, int T_out, int V, int s,
                          float *loss);

// -----------------------------------------------------------
// Superposition Phase: Per-sample gradient scaler
// -----------------------------------------------------------
// Compute the per-sample gradient scale for MCE loss backward.
// d_logits[b, t, v] = (1/s) * sum_{k=0}^{s-1} (softmax(logits[b,t])[v] - delta(v == targets[b,t,k]))
// where delta is 1 if v matches target k, 0 otherwise.
//
// logits:    [B, T_out, V] predicted logits
// targets:   [B, T_out, s] ground-truth token IDs
// B, T_out:  batch and bagged sequence dimensions
// V:         vocabulary size
// s:         bag size
// d_logits:  [B, T_out, V] gradient of loss w.r.t. logits (caller-allocated)
void tst_mce_loss_backward(const float *logits, const int *targets,
                           int B, int T_out, int V, int s,
                           float *d_logits);

// -----------------------------------------------------------
// Utility: Softmax + Cross-Entropy (used internally, exposed
// for standalone testing)
// -----------------------------------------------------------
// In-place softmax on logits of shape [N, V]
void tst_softmax_inplace(float *logits, int N, int V);

// Cross-entropy loss for integer label
// logits: [V] pre-softmax values (will be softmaxed internally)
// label: integer ground-truth class
// Returns: CE loss value
float tst_cross_entropy(const float *logits, int V, int label);

#ifdef __cplusplus
}
#endif

#endif // WUBU_TST_H

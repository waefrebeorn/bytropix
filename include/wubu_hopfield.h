/*
 * wubu_hopfield.h -- Modern Hopfield / associative memory (Theme IL).
 * C11.
 *
 * Convergence (Ramsauer et al 2020 "Hopfield Networks is All You Need":
 * the modern Hopfield update is the attention softmax; the storage
 * capacity is EXPONENTIAL in the pattern dimension):
 *   - IL01 Modern-Hopfield retrieval: xi_new = X^T softmax(beta * X *
 *         xi) -- the stored-pattern matrix X, the inverse temperature
 *         beta (higher = sharper, the attention's 1/sqrt(d)).
 *   - IL02 Attention equivalence: the attention's softmax is the
 *         Hopfield update with beta ~ 1/sqrt(d); expose the mapping.
 *   - IL03 Exponential capacity: C ~ exp(alpha * d) -- the reason the
 *         Hopfield storage beats the Hebbian O(d) capacity.
 *   - IL04 Associative recall: pattern completion from a corrupted cue
 *         (iterate the update to a fixed point).
 *   - IL05 Memory decay: stored patterns lose weight with age (the
 *         STM->LTM forgetting curve).
 *   - IL06 Consolidation: replay/reward strengthens a pattern's weight
 *         (ties the continual-learning themes).
 *   - IL07 The operator: top-k KV-slot retrieval by Hopfield overlap
 *         (the memory as the KV source for the attention).
 */
#ifndef WUBU_HOPFIELD_H
#define WUBU_HOPFIELD_H

/* IL01: one Hopfield retrieval step.
 * X = n_pat x dim row-major stored patterns; xi = the cue (dim);
 * out receives X^T softmax(beta * X * xi). Returns 0. */
int wubu_hopfield_retrieve(const float *X, int n_pat, int dim,
                           const float *xi, float beta, float *out);

/* IL02: the attention-equivalent beta for a key dimension (1/sqrt(d)). */
float wubu_hopfield_beta_attention(int dim);

/* IL03: the exponential capacity estimate: exp(alpha * dim). */
float wubu_hopfield_capacity(int dim, float alpha);

/* IL04: associative recall -- iterate the update until the change is
 * below tol or max_iter reached; the result in out (the caller's
 * buffer, dim floats; may alias xi). Returns the iterations used. */
int wubu_hopfield_denoise(const float *X, int n_pat, int dim,
                          const float *xi, float beta,
                          float tol, int max_iter, float *out);

/* IL05: memory decay -- weight * 2^(-age / halflife). */
float wubu_hopfield_decay(float weight, int age, float halflife);

/* IL06: consolidation -- weight + alpha * reward (reward >= 0). */
float wubu_hopfield_consolidate(float weight, float reward, float alpha);

/* IL07: top-k pattern indices by |X_i . xi| (the Hopfield overlap).
 * out_idx receives the indices sorted by descending overlap.
 * Returns the number written (<= k). */
int wubu_hopfield_topk(const float *X, int n_pat, int dim,
                       const float *xi, int k, int *out_idx);

#endif

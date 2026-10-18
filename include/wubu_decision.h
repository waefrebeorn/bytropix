/*
 * wubu_decision.h -- typed decision node (the SystemOne/Clef/Jev contract).
 *
 * A decision model consumes a state plus a schema of typed questions and
 * returns a probability for every allowed option, with no free-form text. That
 * is the right shape for an OS control plane: the kernel can act on a number.
 *
 * Two things the hosted models do NOT guarantee, and this module does:
 *   - per-question probabilities sum to 1 (softmax is exact, not approximate);
 *   - logically related questions are coherent (arXiv 2609.33209 measured Jev's
 *     "X" and "not X" probabilities missing 1 by 0.064 on average, and three
 *     single-label probabilities summing to 1.14).
 *
 * The gate is deliberately separate from the model: a model can be swapped,
 * the arithmetic contract cannot.
 */
#ifndef WUBU_DECISION_H
#define WUBU_DECISION_H

/* Softmax over n logits. Returns 0, or -1 on bad arguments. */
int wubu_decision_softmax(const float *logits, int n, float *probs);

/* Shannon entropy of a distribution (nats). -1.0 on bad arguments. */
float wubu_decision_entropy(const float *probs, int n);

/* Typed verdict. Returns 1 when max(probs) >= threshold (act), else 0 (defer).
 * *idx receives the argmax, *conf the max probability, on both paths. */
int wubu_decision_choose(const float *probs, int n, float threshold,
                         int *idx, float *conf);

/* Complement consistency: |p(x) + p(not x) - 1| <= tol. Returns 1 if coherent. */
int wubu_decision_complement_ok(float p, float p_not, float tol);

/* Mutual exclusivity: |sum(probs) - 1| <= tol. Returns 1 if coherent. */
int wubu_decision_exclusive_ok(const float *probs, int n, float tol);

/* Expected calibration error over confidence/correctness pairs, `bins` bins.
 * Returns the ECE in [0,1], or -1 on bad arguments. */
float wubu_decision_ece(const float *conf, const unsigned char *correct,
                        int n, int bins);

#endif /* WUBU_DECISION_H */
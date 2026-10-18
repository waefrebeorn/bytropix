/*
 * wubu_manifold_train.h — manifold-native training primitives.
 *
 * The engine already has the geometry (wubu_mobius: Möbius add, exp/log,
 * geodesic distance, all with backward passes) and a hyperbolic LM head
 * (wubu_hyperbolic_output_proj). What was missing is the TRAINING layer that
 * makes the manifold usable in pre-training and fine-tuning:
 *
 *   1. a hyperbolic classifier head (anchors on the ball, logits from
 *      tangent-space distance), with an analytic gradient — pre-training;
 *   2. a Möbius low-rank adapter (log → low-rank update → exp) — fine-tuning.
 *
 * Design notes:
 *  - Logits use the log map at the origin: u = log_map(x), v_k = log_map(a_k),
 *    logit_k = -||u - v_k||^2 / (2 R^2). This is the linearised hyperbolic
 *    classifier; its gradient is exact for this objective and reuses the
 *    existing log_map_backward.
 *  - The adapter update is applied in the tangent space at h, then mapped back,
 *    so the result never leaves the ball. B = 0 is the identity, which is the
 *    property that makes a LoRA adapter safe to attach to a trained model.
 *
 * C11, no third-party dependencies.
 */
#ifndef WUBU_MANIFOLD_TRAIN_H
#define WUBU_MANIFOLD_TRAIN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim;            /* embedding dimension                        */
    int n_classes;      /* number of anchors                          */
    float R;            /* Poincaré ball radius (curvature = -1/R^2)  */
    float *anchor;      /* [n_classes * dim], each ||a|| < R          */
} wubu_manifold_head_t;

/* Allocate a head with deterministic anchors inside the ball.
 * Returns 0 on success, -1 on bad args/allocation failure. */
int wubu_manifold_head_init(wubu_manifold_head_t *h, int dim, int n_classes,
                            float R, unsigned seed);

void wubu_manifold_head_free(wubu_manifold_head_t *h);

/* logits[k] = -||log_map(x) - log_map(anchor_k)||^2 / (2 R^2).
 * x must be inside the ball. Returns 0 on success. */
int wubu_manifold_logits(const wubu_manifold_head_t *h, const float *x,
                         float *logits);

/* Softmax cross-entropy for `label`.
 * dtan:    if non-NULL, receives dL/du in the tangent space at x (size dim).
 *          Apply Riemannian SGD as u <- u - lr*dtan; x <- exp_map(u).
 * danchor: if non-NULL, ACCUMULATES dL/danchor (size n_classes*dim).
 * Returns the loss (>= 0). */
float wubu_manifold_ce(const wubu_manifold_head_t *h, const float *x, int label,
                       float *dtan, float *danchor);

/* Möbius low-rank adapter (fine-tuning):
 *   u  = log_map(h)
 *   u' = u + B (A u)
 *   out = exp_map(u')
 * A: [rank * dim], B: [dim * rank]. B == NULL or rank == 0 is the identity.
 * Returns 0 on success, -1 on bad args. */
int wubu_manifold_lora(const float *h, int dim, float R,
                       const float *A, const float *B, int rank, float *out);

/* Geodesic hierarchy loss (pre-training):
 *   loss = max(0, d(child, parent) - d(child, sibling) + margin)
 * All three points must be inside the ball. True Poincare distance is used,
 * not the tangent approximation, because the criterion is about hierarchy.
 * Returns the loss (0 when the child is already closer to its parent). */
float wubu_manifold_hierarchy_loss(const float *parent, const float *child,
                                   const float *sibling, int dim, float R,
                                   float margin);

/* Radius regulariser: keeps representations off the origin and off the
 * boundary, where hyperbolic gradients vanish.
 * Penalty is a squared hinge against the band [lo, hi] (radii, 0 < lo < hi < R).
 * The hinges sit 5% inward: a hinge exactly on the boundary only converges
 * asymptotically, so the band would never be strictly occupied.
 * If dtan != NULL, receives dPenalty/du in the tangent space at x.
 * Returns the penalty (0 inside the band). */
float wubu_manifold_radius_penalty(const float *x, int dim, float R,
                                   float lo, float hi, float *dtan);

/* Backward for the Mobius LoRA adapter.
 * Given dL/dout (ambient, size dim), writes dL/dA (rank*dim) and
 * dL/dB (dim*rank). Uses the exp-map Jacobian already in wubu_mobius.
 * Returns 0 on success. */
int wubu_manifold_lora_backward(const float *h, int dim, float R,
                                const float *A, const float *B, int rank,
                                const float *dout, float *dA, float *dB);

/* Self-check of the identities this module relies on. Returns 0 when all
 * internal cases pass. The full matrix is tools/test_manifold_train.c. */
int wubu_manifold_train_selftest(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_MANIFOLD_TRAIN_H */

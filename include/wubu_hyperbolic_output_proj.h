#ifndef WUBU_HYPERBOLIC_OUTPUT_PROJ_H
#define WUBU_HYPERBOLIC_OUTPUT_PROJ_H

/**
 * wubu_hyperbolic_output_proj.h — Hyperbolic output projection (lm_head).
 *
 * Replaces the Euclidean lm_head with a hyperbolic pipeline:
 *   hidden (Euclidean) → exp_map(·, R_hidden) → Poincaré ball
 *   → M⊗(·, R_hidden → R_logit) → Poincaré ball
 *   → log_map(·, R_logit) → Euclidean logits (for softmax CE)
 *
 * The M⊗ is implemented by wubu_mobius_linear, which does:
 *   log_map(x, R_in) → tangent → W @ tangent + b → exp_map(·, R_out)
 *
 * Default radii: R_hidden = 2.0f, R_logit = 5.0f.
 *
 * Memory strategy: forward saves h_ball ([N, D_MODEL], cheap) and optionally
 * l_ball ([N, V], expensive). If l_ball is NULL, backward recomputes it.
 * d_W ([V, D_MODEL]) is always accumulated — required for weight updates.
 */

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Forward: hyperbolic output projection
// ============================================================
// hidden:   [N, D]  — Euclidean hidden states (after final norm)
// N:        batch * sequence length
// D:        D_MODEL (hidden dimension)
// V:        vocab_size (output dimension)
// W:        [V, D]  — weight matrix (lm_head)
// b:        [V]     — bias (optional, pass NULL to omit)
// R_hidden: Poincaré ball radius for input projection (default: 2.0f)
// R_logit:  Poincaré ball radius for logit projection (default: 5.0f)
// logits:   [N, V]  — output Euclidean logits (for softmax CE)
// h_ball:   [N, D]  — saved hidden in Poincaré ball (optional for speed)
//                     If non-NULL, saved for backward to avoid recomputation.
// l_ball:   [N, V]  — saved logit in Poincaré ball (optional, can be large)
//                     If non-NULL, saved for backward to avoid recomputation.
//                     If NULL, backward will recompute it (memory-efficient).
void wubu_hyperbolic_output_proj_forward(
    const float *hidden, int N, int D, int V,
    const float *W, const float *b,
    float R_hidden, float R_logit,
    float *logits,
    float *h_ball,
    float *l_ball);

// ============================================================
// Backward: gradients through hyperbolic output projection
// ============================================================
// hidden:    [N, D]  — forward input (Euclidean)
// N, D, V:   same as forward
// W:         [V, D]  — forward weight
// b:         [V]     — forward bias (or NULL)
// R_hidden, R_logit: same as forward
// h_ball:    [N, D]  — saved from forward (or NULL to recompute)
// l_ball:    [N, V]  — saved from forward (or NULL to recompute)
// d_logits:  [N, V]  — upstream gradient w.r.t. Euclidean logits
//                       (e.g., from softmax CE: logits - one_hot(target))
// d_hidden:  [N, D]  — gradient w.r.t. hidden (ADDED TO, or pass NULL)
// d_W:       [V, D]  — gradient w.r.t. weight (ACCUMULATED, or pass NULL)
// d_b:       [V]     — gradient w.r.t. bias (ACCUMULATED, or pass NULL)
void wubu_hyperbolic_output_proj_backward(
    const float *hidden, int N, int D, int V,
    const float *W, const float *b,
    float R_hidden, float R_logit,
    const float *h_ball,
    const float *l_ball,
    const float *d_logits,
    float *d_hidden,
    float *d_W, float *d_b);

#ifdef __cplusplus
}
#endif

#endif // WUBU_HYPERBOLIC_OUTPUT_PROJ_H

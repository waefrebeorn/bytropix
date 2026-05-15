#ifndef WUBU_MOBIUS_LINEAR_H
#define WUBU_MOBIUS_LINEAR_H

/**
 * Möbius linear layer: M⊗(x) = exp_map(W · log_map(x) + b, R_out)
 *
 * A fully hyperbolic linear transformation:
 * 1. Map input x (in Poincaré ball of radius R_in) to tangent space at origin
 * 2. Apply Euclidean linear transformation (weight + bias)
 * 3. Map result back to Poincaré ball of radius R_out
 *
 * Note: R_in and R_out can differ, enabling cross-curvature transformations.
 *
 * For inputs at the origin (x=0), log_map(0)=0, so the layer reduces to
 * a standard Euclidean linear layer mapped to the ball.
 */

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Forward: y = exp_map(W @ log_map(x) + b, R_out)
// ============================================================
// x:      [N, D_in]  — points in Poincaré ball of radius R_in
// W:      [D_out, D_in] — weight matrix
// b:      [D_out] or NULL — bias (added in tangent space)
// R_in:   Poincaré ball radius for input
// R_out:  Poincaré ball radius for output
// output: [N, D_out] — result in Poincaré ball of radius R_out
void wubu_mobius_linear_forward(const float *x, int N, int D_in, int D_out,
                                const float *W, const float *b,
                                float R_in, float R_out,
                                float *output);

// ============================================================
// Backward: gradients through M⊗ layer
// ============================================================
// x:        [N, D_in]  — forward input
// tangent_x: [N, D_in] — saved log_map(x) from forward (or NULL to recompute)
// tangent_out: [N, D_out] — saved W@log_map(x)+b from forward (or NULL to recompute)
// output:   [N, D_out] — forward output
// d_output: [N, D_out] — upstream gradient
// W:        [D_out, D_in] — forward weight
// R_in:     input radius
// R_out:    output radius
// d_x:      [N, D_in] — gradient w.r.t. input (add to existing)
// d_W:      [D_out, D_in] — gradient w.r.t. weight (or NULL)
// d_b:      [D_out] — gradient w.r.t. bias (or NULL)
void wubu_mobius_linear_backward(const float *x, int N, int D_in, int D_out,
                                 const float *tangent_x,
                                 const float *tangent_out,
                                 const float *output,
                                 const float *d_output,
                                 const float *W,
                                 float R_in, float R_out,
                                 float *d_x,
                                 float *d_W, float *d_b);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MOBIUS_LINEAR_H
